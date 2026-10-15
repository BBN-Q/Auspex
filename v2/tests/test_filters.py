"""Offline contract tests for the V2 filter pipeline and VNA averaging group."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from auspex_v2.data import Axis, AuspexStore
from auspex_v2.filters import (Batch, Collector, ComplexAverager, DatasetWriter,
                                Passthrough, Pipeline, PipelineError, PortSchema)
from auspex_v2.instruments.drivers.simulated import SimulatedVNA


SCHEMA = PortSchema("<c8", 1, ("frequency",), ("Hz",), 64_000)


def make_pipeline(repeats=5):
    nodes = {"pass": Passthrough(SCHEMA), "average": ComplexAverager(SCHEMA, repeats=repeats),
             "raw": Collector(SCHEMA), "preview": Collector(SCHEMA, preview=True),
             "final": Collector(SCHEMA)}
    edges = [("$input", "out", "pass"), ("pass", "out", "average"),
             ("pass", "out", "raw"), ("average", "partial", "preview"),
             ("average", "final", "final")]
    return Pipeline(SCHEMA, nodes, edges), nodes


class FilterContractTest(unittest.TestCase):
    """Exercise graph validation, ordered windows, exact grids, and failures."""

    def test_five_simulated_vna_sweeps(self):
        """Plot labeled partial snapshots without confusing them with required results."""
        pipeline, nodes = make_pipeline()
        vna = SimulatedVNA("SIM::VNA")
        vna.connect()
        for index in range(5):
            frequency, trace = vna.get_trace()
            batch = Batch(None, index, 0, 0, trace, (Axis("frequency", frequency, "Hz"),))
            pipeline.push(batch)
            pipeline.end_window(index)
            self.assertEqual(len(nodes["preview"].batches), 1)
            self.assertEqual(nodes["preview"].batches[-1].count, index + 1)
            self.assertEqual(nodes["preview"].batches[-1].expected_count, 5)
            self.assertEqual(len(nodes["final"].batches), 1 if index == 4 else 0)
        pipeline.finish()
        self.assertEqual(len(nodes["raw"].batches), 5)
        self.assertEqual(len(nodes["final"].batches), 1)
        np.testing.assert_array_equal(nodes["final"].batches[0].data, trace)
        self.assertEqual(nodes["final"].batches[0].count, 5)
        vna.disconnect()

    def test_independent_group_sizes_on_two_branches(self):
        """Filters derive their own group boundaries from the same six windows."""
        nodes = {"average_two": ComplexAverager(SCHEMA, repeats=2, group_name="two"),
                 "average_three": ComplexAverager(SCHEMA, repeats=3, group_name="three"),
                 "results_two": Collector(SCHEMA), "results_three": Collector(SCHEMA)}
        graph = Pipeline(SCHEMA, nodes, [("$input", "out", "average_two"),
            ("$input", "out", "average_three"),
            ("average_two", "final", "results_two"),
            ("average_three", "final", "results_three")])
        for window in range(6):
            graph.push(Batch(None, window, 0, 0, np.array([window + 1j], dtype=np.complex64),
                             (Axis("frequency", [1.], "Hz"),)))
            graph.end_window(window)
        graph.finish()
        self.assertEqual([b.group for b in nodes["results_two"].batches],
                         ["two-0", "two-1", "two-2"])
        self.assertEqual([b.window for b in nodes["results_two"].batches], [1, 3, 5])
        self.assertEqual([b.group for b in nodes["results_three"].batches],
                         ["three-0", "three-1"])
        self.assertEqual([b.window for b in nodes["results_three"].batches], [2, 5])

    def test_completed_sweeps_and_final_average_are_published(self):
        """Persist only completed arrays through one writer-owned .auspex folder."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            with AuspexStore(path, mode="w") as store:
                nodes = {"average": ComplexAverager(SCHEMA, repeats=2),
                         "raw": DatasetWriter(SCHEMA, store, group="sweeps"),
                         "final": DatasetWriter(SCHEMA, store, group="averages", when="group")}
                pipeline = Pipeline(SCHEMA, nodes, [("$input", "out", "average"),
                    ("$input", "out", "raw"), ("average", "final", "final")])
                for index in range(2):
                    pipeline.push(Batch(None, index, 0, 0,
                        np.array([index + 1j], dtype=np.complex64),
                        (Axis("frequency", [6.1e9], "Hz"),)))
                    self.assertEqual(len(store.list_datasets()), index)
                    pipeline.end_window(index)
                pipeline.finish()
                self.assertEqual(len(store.list_datasets()), 3)
            with AuspexStore(path) as store:
                np.testing.assert_array_equal(
                    store.read_dataset("averages", "average-0_0001").data, [0.5 + 1j])

    def test_grid_mismatch_fails_without_corrupted_preview(self):
        """Never include a mismatched instrument-readback grid in an average."""
        pipeline, nodes = make_pipeline(repeats=2)
        for index, points in enumerate(([1., 2.], [1., 2.000001])):
            batch = Batch(None, index, 0, 0, np.array([1j, 2j], dtype=np.complex64),
                          (Axis("frequency", points, "Hz"),))
            if index == 0:
                pipeline.push(batch)
                pipeline.end_window(index)
            else:
                with self.assertRaisesRegex(PipelineError, "grids differ"):
                    pipeline.push(batch)
        self.assertEqual(len(nodes["preview"].batches), 1)
        self.assertEqual(len(nodes["final"].batches), 0)
        with self.assertRaisesRegex(PipelineError, "Pipeline failed"):
            pipeline.finish()

    def test_input_cannot_assign_a_filter_group(self):
        graph, _ = make_pipeline(repeats=2)
        with self.assertRaisesRegex(PipelineError, "Only the pipeline assigns groups"):
            graph.push(Batch("from-runner", 0, 0, 0,
                             np.array([1j], dtype=np.complex64),
                             (Axis("frequency", [1.], "Hz"),)))

    def test_incomplete_group_and_bad_order_fail(self):
        """Reject run end with an incomplete filter group and out-of-order windows."""
        pipeline, _ = make_pipeline(repeats=2)
        pipeline.push(Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                            (Axis("frequency", [1.], "Hz"),)))
        pipeline.end_window(0)
        with self.assertRaisesRegex(PipelineError, "incomplete"):
            pipeline.finish()
        other, _ = make_pipeline()
        with self.assertRaisesRegex(PipelineError, "Unexpected window"):
            other.push(Batch(None, 2, 0, 0, np.array([1j], dtype=np.complex64),
                             (Axis("frequency", [1.], "Hz"),)))

    def test_parameter_changes_only_after_group_completion(self):
        """Apply configuration revisions only after the active group has ended."""
        pipeline, _ = make_pipeline(repeats=2)
        batch = Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                      (Axis("frequency", [1.], "Hz"),))
        pipeline.push(batch)
        pipeline.end_window(0)
        with self.assertRaisesRegex(PipelineError, "Averaging repeats"):
            pipeline.update("average", {"repeats": 3})
        pipeline.push(Batch(None, 1, 0, 0, batch.data, batch.axes))
        pipeline.end_window(1)
        self.assertEqual(pipeline.update("average", {"repeats": 3}), 1)
        with self.assertRaisesRegex(PipelineError, "configured batch"):
            pipeline.push(Batch(None, 2, 0, 0, batch.data, batch.axes))

    def test_writer_retains_its_input_until_window_published(self):
        """A future transport may release its slot immediately after delivery."""
        with tempfile.TemporaryDirectory() as directory:
            with AuspexStore(Path(directory) / "run.auspex", mode="w") as store:
                writer = DatasetWriter(SCHEMA, store, group="raw")
                values = np.array([1j], dtype=np.complex64)
                points = np.array([6.1e9])
                writer.on_batch(Batch(None, 0, 0, 0, values,
                                      (Axis("frequency", points, "Hz"),)))
                values[0], points[0] = 9j, 1.0
                writer.on_window_end(0)
                saved = store.read_dataset("raw", "window_0000")
                self.assertEqual(saved.data[0], 1j)
                self.assertEqual(saved.axes[0].points, [6.1e9])

    def test_schema_counts_coordinates_toward_limit(self):
        schema = PortSchema("<c8", 1, ("frequency",), ("Hz",), 8)
        with self.assertRaisesRegex(PipelineError, "coordinates exceeds"):
            schema.check(Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                               (Axis("frequency", [1.], "Hz"),)))

    def test_invalid_graph_schema_and_preview_failure(self):
        """Reject bad graphs and data but isolate explicitly best-effort subscribers."""
        with self.assertRaisesRegex(PipelineError, "exactly one"):
            Pipeline(SCHEMA, {"node": Passthrough(SCHEMA)}, [])
        with self.assertRaisesRegex(PipelineError, "Incompatible"):
            Pipeline(SCHEMA, {"node": Collector(PortSchema("<f8", 1, ("frequency",), ("Hz",), 100))},
                     [("$input", "out", "node")])
        pipeline, nodes = make_pipeline(repeats=1)
        def broken(_):
            raise RuntimeError("viewer left")
        nodes["preview"].on_batch = broken
        batch = Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                      (Axis("frequency", [1.], "Hz"),))
        pipeline.push(batch)
        pipeline.end_window(0)
        pipeline.finish()
        self.assertEqual(len(nodes["final"].batches), 1)
        self.assertFalse(nodes["preview"].batches)
