"""Real worker/writer integration for optional, reconstructible live displays."""

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from auspex_v2.data import Axis, AuspexStore
from auspex_v2.filters import ManualPipeline, PipelineError, PortSchema
from auspex_v2.filters.definitions import (
    ComplexAverage, CompiledPipeline, PipelineDefinition, PipelineInput, Plot, WriteToFile,
)
from auspex_v2.filters.preview import decode_frame, request_preview
from auspex_v2.filters.preview_client import poll_run, read_json

SCHEMA = PortSchema("complex128", 1, ("frequency",), ("Hz",), 4096, value_name="S21")


def recipe(*, repeats=2, map_bytes=1024 * 1024):
    """Use partial averages for the trace, finalized rows for both writer and map."""
    source = PipelineInput("vna", SCHEMA)
    average = ComplexAverage(name="average", repeats=repeats)
    writer = WriteToFile(name="disk", group="measurement", dataset="response")
    trace = Plot(name="current_trace")
    image = Plot(name="bias_map", mode="map", max_bytes=map_bytes)
    return PipelineDefinition(inputs=[source], edges=[
        (source, average.sink), (average.final_average, writer.sink),
        (average.partial_average, trace.sink), (average.final_average, image.sink),
    ]).compile()


class LivePreviewTest(unittest.TestCase):
    """Display loss and display-specific failures must not invalidate saved data."""

    def test_detached_definition_and_validation(self):
        """Plot declarations survive JSON round trips without changing runtime tuple APIs."""
        compiled = recipe()
        restored = CompiledPipeline.from_document(compiled.document)
        self.assertEqual(restored.document, compiled.document)
        self.assertEqual(len(restored.runtime()), 6)
        plots = restored.plot_specs()
        self.assertEqual(plots["current_trace"].port, "partial")
        self.assertEqual(plots["bias_map"].writer, "disk")
        document = compiled.document
        plot = next(n for n in document["definition"]["nodes"] if n["name"] == "current_trace")
        plot["config"]["mode"] = "map"
        with self.assertRaisesRegex(PipelineError, "partial_average"):
            CompiledPipeline.from_document(document)
        with self.assertRaisesRegex(PipelineError, "display budget"):
            recipe(map_bytes=64 * 1024 * 1024)
        source = PipelineInput("source", SCHEMA)
        image = Plot(name="image", mode="map")
        disk = WriteToFile(name="disk")
        avg = ComplexAverage(name="avg")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, disk.sink), (source, avg.sink), (avg.final_average, image.sink)])
        with self.assertRaisesRegex(PipelineError, "writer on the same"):
            definition.compile()

    def test_late_attach_missing_cells_dropped_frames_and_final_cache(self):
        """Skipped GUI frames cannot lose accepted rows or finalized averaging values."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            frequency = np.array([1., 2., 4., 7.])
            biases = [0., 0.3, 0.8]
            expected = []
            with ManualPipeline(recipe(), store_path=path,
                                preview_sweep=Axis("bias", biases, "V")) as pipeline:
                for i, bias in enumerate(biases):
                    row = np.array(frequency + 1j * (bias + frequency))
                    expected.append(row + 0.5)
                    for repeat in range(2):
                        window = pipeline.submit(row + repeat,
                            axes=[Axis("frequency", frequency, "Hz")],
                            sweep={"bias": bias}, sweep_units={"bias": "V"})
                        pipeline._graph.wait_window(window)
                    if i == 0:
                        catalog = read_json(path / "preview.json")
                        endpoint = catalog["endpoint"]
                        # Attach only after acquisition has started, no prior subscription.
                        live, metadata = decode_frame(request_preview(endpoint, "bias_map"))
                        self.assertEqual(live.dimensions, ("bias", "frequency"))
                        self.assertEqual(live.data.shape, (3, 4))
                        np.testing.assert_allclose(live.data[0], expected[0])
                        self.assertTrue(np.all(np.isnan(live.data[1:].real)))
                        self.assertTrue(np.all(np.isnan(live.data[1:].imag)))
                        self.assertEqual(metadata["count"], 2)
                        self.assertEqual(metadata["publication"], "not acknowledged")
                    if i == 1:
                        # A burst followed by a pause must eventually flush its newest row.
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            live, metadata = decode_frame(request_preview(endpoint, "bias_map"))
                            if metadata["window"] >= window:
                                break
                            time.sleep(0.02)
                        self.assertGreaterEqual(metadata["window"], window)
                        np.testing.assert_allclose(live.data[:2], expected)
                    # No viewer for the final row; no mandatory display drainage.
            run = poll_run(path, wanted=["bias_map", "current_trace"])
            self.assertEqual(run["connection"], "closed")
            self.assertEqual(run["state"], "CLOSED")  # not a claim of experiment success
            image, metadata = run["frames"]["bias_map"]
            np.testing.assert_allclose(image.data, expected)
            self.assertEqual(metadata["publication"], "published")
            trace, metadata = run["frames"]["current_trace"]
            np.testing.assert_allclose(trace.data, expected[-1])
            self.assertEqual(metadata["count"], metadata["expected_count"])
            with AuspexStore(path) as store:
                saved = store.read_dataset("measurement", "response")
                np.testing.assert_allclose(saved.data, expected)
            self.assertFalse(poll_run(path, wanted=["bias_map"],
                             sequences={"bias_map": run["catalog"]["plots"]["bias_map"]["sequence"]})["frames"])

    def test_partial_trace_arrives_before_finalized_map_row(self):
        """An incomplete average is visible as a trace but never painted as final map data."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.auspex"
            with ManualPipeline(recipe(), store_path=path,
                                preview_sweep=Axis("bias", [0.], "V")) as pipeline:
                pipeline.submit(np.array([1+2j, 3+4j]), axes=[Axis("frequency", [1., 2.], "Hz")],
                                sweep={"bias": 0.}, sweep_units={"bias": "V"})
                endpoint = read_json(path / "preview.json")["endpoint"]
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    payload = request_preview(endpoint, "current_trace")
                    if payload:
                        break
                    time.sleep(0.01)
                self.assertTrue(payload)
                _, metadata = decode_frame(payload)
                self.assertEqual((metadata["count"], metadata["expected_count"]), (1, 2))
                self.assertFalse(request_preview(endpoint, "bias_map"))
                pipeline.submit(np.array([3+4j, 5+6j]), axes=[Axis("frequency", [1., 2.], "Hz")],
                                sweep={"bias": 0.}, sweep_units={"bias": "V"})
            final = poll_run(path, wanted=["bias_map"])["frames"]["bias_map"][0]
            np.testing.assert_array_equal(final.data, [[2+3j, 4+5j]])

    def test_cancelled_scan_keeps_preview_distinct_from_publication(self):
        """An acquired preview row survives ordinary close without becoming a saved scan."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancelled.auspex"
            pipeline = ManualPipeline(recipe(repeats=1), store_path=path,
                                      preview_sweep=Axis("bias", [0., 1.], "V"))
            try:
                window = pipeline.submit(np.ones(2, dtype=np.complex128),
                    axes=[Axis("frequency", [1., 2.], "Hz")],
                    sweep={"bias": 0.}, sweep_units={"bias": "V"})
                pipeline._graph.wait_window(window)
            finally:
                pipeline.cancel()
            run = poll_run(path, wanted=["bias_map"])
            frame, metadata = run["frames"]["bias_map"]
            self.assertEqual(metadata["publication"], "not acknowledged")
            self.assertTrue(np.all(np.isnan(frame.data[1])))
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [])

    def test_display_budget_failure_does_not_fail_publication(self):
        """A map may disable itself, while its required writer still saves every row."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.auspex"
            with ManualPipeline(recipe(repeats=1, map_bytes=8), store_path=path,
                                preview_sweep=Axis("bias", [0., 1.], "V")) as pipeline:
                for bias in [0., 1.]:
                    pipeline.submit(np.ones(4, dtype=np.complex128),
                        axes=[Axis("frequency", [1., 2., 3., 4.], "Hz")],
                        sweep={"bias": bias}, sweep_units={"bias": "V"})
            catalog = read_json(path / "preview.json")
            self.assertIn("budget", catalog["plots"]["bias_map"]["error"])
            with AuspexStore(path) as store:
                self.assertEqual(store.read_dataset("measurement", "response").data.shape, (2, 4))

    def test_changed_grid_and_duplicate_positions_disable_only_map(self):
        """Unsupported maps stay explicitly unsupported; primary files preserve actual data."""
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "irregular.auspex"
                with ManualPipeline(recipe(repeats=1), store_path=path) as pipeline:
                    for index in range(2):
                        grid = [1., 2.] if index == 0 or duplicate else [1., 3.]
                        pipeline.submit(np.ones(2, dtype=np.complex128),
                            axes=[Axis("frequency", grid, "Hz")],
                            sweep={"bias": 0. if duplicate else float(index)}, sweep_units={"bias": "V"})
                error = read_json(path / "preview.json")["plots"]["bias_map"]["error"]
                self.assertIn("Multiple finalized" if duplicate else "grid changed", error)
                with AuspexStore(path) as store:
                    saved = store.read_dataset("measurement", "response")
                    self.assertEqual(saved.data.shape, (2, 2))
                    if not duplicate:
                        self.assertEqual(saved.coordinates["frequency"].dimensions, ("bias", "frequency"))

    def test_preview_is_optional_when_endpoint_startup_fails(self):
        """An occupied descriptor path should warn, not turn plotting into a run gate."""
        # Startup failure injection is local to the store-owning process via a path collision.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.auspex"
            with AuspexStore(path, mode="w"):
                (path / "preview.tmp").mkdir()
            with ManualPipeline(recipe(repeats=1), store_path=path, existing_store=True,
                                preview_sweep=Axis("bias", [0.], "V")) as pipeline:
                pipeline.submit(np.ones(2, dtype=np.complex128),
                    axes=[Axis("frequency", [1., 2.], "Hz")],
                    sweep={"bias": 0.}, sweep_units={"bias": "V"})
            with AuspexStore(path) as store:
                self.assertEqual(store.read_dataset("measurement", "response").data.shape, (1, 2))
