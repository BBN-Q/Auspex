"""Round-trip flexible datasets and supervise the required process-backed writer."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from auspex_v2.data import AuspexStore, Axis, Coordinate, FORMAT_VERSION
from auspex_v2.filters import (Batch, FilterServer, NodeSpec, PipelineError, ResultSpec,
                               PortSchema, SweepPosition, WriterSpec)

SCHEMA = PortSchema("<c8", 1, ("frequency",), ("Hz",), 1024,
                    value_name="S21", value_unit="dimensionless")


def measurement(scan, window, bias, frequencies):
    return Batch(None, window, 0, 0,
                 np.asarray(frequencies, dtype=np.complex64) * 1j,
                 (Axis("frequency", np.asarray(frequencies, dtype=np.float64), "Hz"),),
                 scan=scan, sweep=(SweepPosition("bias", bias, "V"),))


class DatasetFileTest(unittest.TestCase):
    """Recover complete named arrays and coordinates with no manifest present."""

    def test_rectangular_dependent_and_ragged(self):
        """Recover shared, dependent, and ragged axes without an outer index."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            with AuspexStore(path, mode="w") as store:
                for name, lengths in (("shared", (2, 2)), ("dependent", (2, 2)),
                                      ("ragged", (2, 3))):
                    rows = [np.arange(n, dtype=np.float64) + i for i, n in enumerate(lengths)]
                    sweep = Coordinate("bias", np.array([0.0, 0.3]), ("point",), "V")
                    point = Coordinate("point", np.arange(2), ("point",))
                    if name == "ragged":
                        data = np.concatenate(rows).astype(np.complex64)
                        coords = (point, sweep, Coordinate("frequency", np.concatenate(rows),
                                                          ("sample",), "Hz"))
                        store.write_dataset("data", name, data, coordinates=coords,
                                            dimensions=("point",), offsets=[0, 2, 5],
                                            value_name="S21", value_unit="dimensionless")
                    else:
                        coords = (point, sweep, Coordinate("frequency", rows[0] if name == "shared"
                                                         else np.stack(rows),
                                                         ("frequency",) if name == "shared"
                                                         else ("point", "frequency"), "Hz"))
                        store.write_dataset("data", name, np.stack(rows).astype(np.complex64),
                                            coordinates=coords, dimensions=("point", "frequency"),
                                            value_name="S21", value_unit="dimensionless")
            (path / "manifest.json").unlink()
            for name in ("shared", "dependent", "ragged"):
                loaded = AuspexStore.read_file(path / "data" / f"{name}.dat")
                self.assertEqual(loaded.value_name, "S21")
                self.assertEqual(loaded.value_unit, "dimensionless")
                self.assertEqual(loaded.coordinates[1].unit, "V")
                if name == "ragged":
                    np.testing.assert_array_equal(loaded.trace(1), [1, 2, 3])
                elif name == "dependent":
                    self.assertEqual(loaded.coordinates[2].dimensions, ("point", "frequency"))


class StoreOwnerTest(unittest.TestCase):
    """Require durable publication from multiple logical writers before scan completion."""

    def graph(self, path, **kwargs):
        return FilterServer(SCHEMA, {"pass": NodeSpec("passthrough", SCHEMA)},
                            [("$input", "out", "pass")], {},
                            writers={"raw": WriterSpec("pass", "out", "data"),
                                     "copy": WriterSpec("pass", "out", "backup")},
                            store_path=path, **kwargs)

    def test_two_writers_two_scans_with_changing_and_ragged_grid(self):
        """Do not acknowledge scans until both destinations are self-contained."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_001.auspex"
            with self.graph(path, input_name="vna_s21",
                            run_metadata={"settings": {"source_power_dBm": -30}}) as graph:
                for window, bias, freq in ((0, 0., [1., 2.]), (1, .3, [1.1, 2.1])):
                    graph.push(measurement("sweep_000", window, bias, freq))
                    graph.end_window(window)
                graph.end_scan("sweep_000")
                graph.wait_scan("sweep_000")
                self.assertEqual(len(graph.published), 2)
                for window, freq in ((2, [3.]), (3, [3., 4., 5.])):
                    graph.push(measurement("reference", window, 0., freq))
                    graph.end_window(window)
                graph.end_scan("reference")
                graph.finish()
                graph.wait_finished()
            with AuspexStore(path) as store:
                self.assertEqual(len(store.list_datasets()), 4)
                self.assertEqual(store.read_run_metadata()["settings"]["source_power_dBm"], -30)
                result = store.read_dataset("data", "sweep_000")
                self.assertEqual(result.coordinates[-1].dimensions, ("bias", "frequency"))
                np.testing.assert_array_equal(result.coordinates[0].values, [0., .3])
                self.assertEqual(result.data.shape, (2, 2))
                reference = store.read_dataset("backup", "reference")
                self.assertEqual(reference.label, "reference")
                np.testing.assert_array_equal(reference.offsets, [0, 1, 4])
                np.testing.assert_array_equal(reference.trace(1), [3j, 4j, 5j])
            (path / "manifest.json").unlink()
            recovered = AuspexStore.read_file(path / "data" / "sweep_000.dat")
            self.assertEqual(recovered.source, "pass.out")
            self.assertEqual(recovered.input_name, "vna_s21")

    def test_completed_group_writes_one_average_after_two_windows(self):
        """Sparse group results do not require an emission at every window."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "average.auspex"
            with FilterServer(SCHEMA, {"avg": NodeSpec("complex_average", SCHEMA, {"repeats": 2})},
                    [("$input", "out", "avg")], {},
                    writers={"mean": WriterSpec("avg", "final")}, store_path=path) as graph:
                for window in range(2):
                    graph.push(measurement("sweep_000", window, 0., [1., 2.]))
                    graph.end_window(window)
                graph.end_scan("sweep_000")
                graph.wait_scan("sweep_000")
                graph.finish()
                graph.wait_finished()
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [("data", "sweep_000")])
                np.testing.assert_array_equal(store.read_dataset("data", "sweep_000").data, [[1j, 2j]])

    def test_writer_and_required_result_both_gate_scan_completion(self):
        """An adaptive decision result can coexist with durable scan publication."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.auspex"
            with FilterServer(SCHEMA, {"pass": NodeSpec("passthrough", SCHEMA)},
                    [("$input", "out", "pass")], {"trace": ResultSpec("pass", "out")},
                    writers={"raw": WriterSpec("pass", "out")}, store_path=path) as graph:
                graph.push(measurement("reference", 0, 0., [1.]))
                graph.end_window(0)
                graph.end_scan("reference")
                graph.wait_scan("reference")
                self.assertEqual(graph.receive("trace").scan, "reference")
                graph.finish()
                graph.wait_finished()
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [("data", "reference")])

    def test_two_sweep_parameters_form_named_dimensions(self):
        """Complete Cartesian scans yield N-dimensional named sweep coordinates."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.auspex"
            with FilterServer(SCHEMA, {"pass": NodeSpec("passthrough", SCHEMA)},
                    [("$input", "out", "pass")], {},
                    writers={"raw": WriterSpec("pass", "out")}, store_path=path) as graph:
                for window, (bias, power) in enumerate(((0., -30.), (0., -20.),
                                                        (1., -30.), (1., -20.))):
                    original = measurement("scan", window, bias, [1., 2.])
                    from dataclasses import replace
                    graph.push(replace(original, sweep=(SweepPosition("bias", bias, "V"),
                                                        SweepPosition("power", power, "dBm"))))
                    graph.end_window(window)
                graph.end_scan("scan")
                graph.finish()
                graph.wait_finished()
            with AuspexStore(path) as store:
                ds = store.read_dataset("data", "scan")
                self.assertEqual(ds.dimensions, ("bias", "power", "frequency"))
                self.assertEqual(ds.data.shape, (2, 2, 2))
                np.testing.assert_array_equal(ds.coordinates[1].values, [-30., -20.])

    def test_incomplete_group_at_scan_boundary_fails(self):
        """Never mix an unfinished filter-owned group across runner-owned scans."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.auspex"
            with FilterServer(SCHEMA, {"avg": NodeSpec("complex_average", SCHEMA, {"repeats": 2})},
                    [("$input", "out", "avg")], {},
                    writers={"mean": WriterSpec("avg", "final")}, store_path=path) as graph:
                graph.push(measurement("sweep_000", 0, 0., [1.]))
                graph.end_window(0)
                graph.end_scan("sweep_000")
                with self.assertRaisesRegex(PipelineError, "incomplete"):
                    graph.wait_scan("sweep_000")

    def test_store_owner_death_fails_supervisor(self):
        """A lost physical writer cannot count as successful publication."""
        with tempfile.TemporaryDirectory() as directory:
            with self.graph(Path(directory) / "crash.auspex") as graph:
                graph.processes["$store"].kill()
                with self.assertRaises(PipelineError):
                    graph.wait_window(0, timeout=2)
                self.assertTrue(graph.failed.is_set())

    def test_scan_capacity_and_file_version(self):
        """Bound staging and reject oversized scans instead of dropping data."""
        self.assertEqual(FORMAT_VERSION, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "limit.auspex"
            with FilterServer(SCHEMA, {"pass": NodeSpec("passthrough", SCHEMA)},
                    [("$input", "out", "pass")], {},
                    writers={"raw": WriterSpec("pass", "out", max_scan_bytes=1)},
                    store_path=path) as graph:
                graph.push(measurement("scan", 0, 0., [1.]))
                graph.end_window(0)
                with self.assertRaisesRegex(PipelineError, "capacity"):
                    graph.wait_window(0)
