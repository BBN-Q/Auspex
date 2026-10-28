"""Bounded scan assembly must preserve the existing self-contained dataset format."""

from contextlib import contextmanager
from itertools import product
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from auspex_v2.data import Axis, Coordinate, Dataset, AuspexStore
from auspex_v2.filters.core import Batch, PipelineError, PortSchema, SweepPosition
from auspex_v2.filters.scan import ScanAssembler

ROOT = Path(__file__).resolve().parents[2]
METADATA = {"label": "scan", "source": "filter.out", "input_name": "measurement"}


@contextmanager
def staged_scan(schema, rows):
    """Stage real windows and own their files/store without starting hardware or processes."""
    with tempfile.TemporaryDirectory(dir=ROOT / ".agents") as directory:
        assembler = ScanAssembler(schema, scratch_dir=directory)
        try:
            for window, (data, axes, sweep) in enumerate(rows):
                assembler.on_batch(Batch(None, window, 0, 0, data, axes, scan="scan", sweep=sweep))
                assembler.on_window_end(window)
            with AuspexStore(Path(directory) / "run.auspex", mode="w") as store:
                yield assembler, store
        finally:
            assembler.close()


class ScanAssemblyTest(unittest.TestCase):
    """Compare bounded publication to independent arrays through the ordinary save API."""

    def assert_equivalent(self, assembler, store, expected):
        """Require byte-identical encoding, plus recovery through the unchanged reader."""
        assembler.publish("scan", store, "actual", "scan", **METADATA)
        store.write_dataset("expected", "scan", expected.data, coordinates=expected.coordinates,
                            dimensions=expected.dimensions, offsets=expected.offsets,
                            value_name=assembler.schema.value_name, value_unit=assembler.schema.value_unit,
                            **METADATA)
        actual = store.path / "actual" / "scan.dat"
        self.assertEqual(actual.read_bytes(), (store.path / "expected" / "scan.dat").read_bytes())
        loaded = AuspexStore.read_file(actual)
        np.testing.assert_array_equal(loaded.data, expected.data)
        self.assertEqual(loaded.dimensions, expected.dimensions)
        self.assertEqual(loaded.label, "scan")
        self.assertEqual(assembler.rows, [])
        return loaded

    def test_cartesian_3d_sweep_with_shared_and_dependent_measurement_axes(self):
        """Preserve product order and dependent coordinates in a multidimensional result."""
        schema = PortSchema("complex64", 2, ("frequency", "time"), ("Hz", "s"), 1024,
                            value_name="response", value_unit="V")
        positions = list(product([0., 1.], [10., 20.], [-1., 1.]))
        data = [(np.arange(6).reshape(2, 3) + i * 1j).astype(np.complex64) for i in range(8)]
        times = [np.asarray([0., .2, .9]) + i * .01 for i in range(8)]
        rows = [(values, (Axis("frequency", np.asarray([4., 5.], dtype="f4" if i == 0 else "f8"), "Hz"),
                          Axis("time", times[i], "s")),
                 tuple(SweepPosition(name, value) for name, value in zip(("a", "b", "c"), position)))
                for i, (values, position) in enumerate(zip(data, positions))]
        expected = Dataset(np.stack(data).reshape(2, 2, 2, 2, 3), (), (
            Coordinate("a", np.asarray([0., 1.]), ("a",)),
            Coordinate("b", np.asarray([10., 20.]), ("b",)),
            Coordinate("c", np.asarray([-1., 1.]), ("c",)),
            Coordinate("frequency", np.asarray([4., 5.], dtype="f4"), ("frequency",), "Hz"),
            Coordinate("time", np.stack(times).reshape(2, 2, 2, 3), ("a", "b", "c", "time"), "s")),
            dimensions=("a", "b", "c", "frequency", "time"))
        with staged_scan(schema, rows) as (assembler, store):
            loaded = self.assert_equivalent(assembler, store, expected)
            self.assertEqual(loaded.coordinates["frequency"].values.dtype, np.dtype("f4"))

    def test_joint_path_does_not_enumerate_hypothetical_cartesian_product(self):
        """A diagonal path has N acquired points, not N squared combinations."""
        count = 512
        schema = PortSchema("float32", 0, (), (), 64)
        rows = [(np.asarray(i, dtype="f4"), (), (SweepPosition("x", i), SweepPosition("y", i * 10)))
                for i in range(count)]
        expected = Dataset(np.arange(count, dtype="f4"), (), (
            Coordinate("point", np.arange(count), ("point",)),
            Coordinate("x", np.arange(count), ("point",)),
            Coordinate("y", np.arange(count) * 10, ("point",))), dimensions=("point",))
        with staged_scan(schema, rows) as (assembler, store), patch(
                "auspex_v2.filters.scan.product", side_effect=AssertionError("Enumerated a missing grid")):
            self.assert_equivalent(assembler, store, expected)

    def test_repeated_and_out_of_order_positions_stay_point_indexed(self):
        """Matching grid cardinality alone must not reorder data or hide repeated positions."""
        schema = PortSchema("float64", 0, (), (), 64)
        for positions in (((0, 0), (1, 1), (0, 1), (1, 0)),
                          ((0, 0), (0, 1), (1, 0), (1, 0))):
            with self.subTest(positions=positions):
                rows = [(np.asarray(float(i)), (), (SweepPosition("x", x), SweepPosition("y", y)))
                        for i, (x, y) in enumerate(positions)]
                expected = Dataset(np.arange(4, dtype="f8"), (), (
                    Coordinate("point", np.arange(4), ("point",)),
                    Coordinate("x", np.asarray([p[0] for p in positions]), ("point",)),
                    Coordinate("y", np.asarray([p[1] for p in positions]), ("point",))), dimensions=("point",))
                with staged_scan(schema, rows) as (assembler, store):
                    self.assert_equivalent(assembler, store, expected)

    def test_ragged_empty_rows_and_coordinate_dtype_promotion(self):
        """Keep empty rows and flatten changing grids without a scan-sized concatenation."""
        schema = PortSchema("complex64", 1, ("frequency",), ("Hz",), 1024)
        lengths = (0, 2, 0, 3)
        data = [np.arange(n, dtype="f4").astype("c8") + i * 1j for i, n in enumerate(lengths)]
        axes = [np.arange(n, dtype=dtype) for n, dtype in zip(lengths, ("f4", "i4", ">f8", "f8"))]
        rows = [(values, (Axis("frequency", points, "Hz"),), (SweepPosition("bias", i // 2),))
                for i, (values, points) in enumerate(zip(data, axes))]
        expected = Dataset(np.concatenate(data), (), (
            Coordinate("point", np.arange(4), ("point",)),
            Coordinate("bias", np.asarray([0, 0, 1, 1]), ("point",)),
            Coordinate("frequency", np.concatenate(axes), ("sample",), "Hz")),
            offsets=np.asarray([0, 0, 2, 2, 5]), dimensions=("point",))
        with staged_scan(schema, rows) as (assembler, store), patch(
                "auspex_v2.filters.scan.product", side_effect=AssertionError("Ragged grid classification")):
            loaded = self.assert_equivalent(assembler, store, expected)
            self.assertEqual(loaded.trace(0).size, 0)
            self.assertEqual(loaded.trace(2).size, 0)

    def test_zero_length_uniform_traces_and_non_native_value_dtype(self):
        """Zero-byte sections and NumPy's historical native-endian assembly both round-trip."""
        for length, dtype in ((0, "f4"), (2, ">f4")):
            with self.subTest(length=length, dtype=dtype):
                schema = PortSchema(dtype, 1, ("time",), ("s",), 1024)
                values = [np.arange(length, dtype=dtype) + i for i in range(2)]
                # Addition can normalize byte order; enforce the actual input schema.
                values = [v.astype(dtype) for v in values]
                rows = [(v, (Axis("time", np.arange(length), "s"),), ()) for v in values]
                expected = Dataset(np.stack(values), (), (
                    Coordinate("point", np.arange(2), ("point",)),
                    Coordinate("time", np.arange(length), ("time",), "s")),
                    dimensions=("point", "time"))
                with staged_scan(schema, rows) as (assembler, store):
                    self.assert_equivalent(assembler, store, expected)

    def test_payload_copy_failure_leaves_no_publication_and_can_retry(self):
        """A second-pass I/O failure cleans the final temporary file, not staged windows."""
        schema = PortSchema("float64", 0, (), (), 64)
        rows = [(np.asarray(float(i)), (), ()) for i in range(3)]
        with staged_scan(schema, rows) as (assembler, store):
            store.write_dataset("previous", "complete", np.asarray(7.))
            read = assembler._read_window
            calls = 0

            def failing_read(file):
                nonlocal calls
                calls += 1
                if calls == 5:  # Three layout reads, then failure after the first payload window.
                    raise OSError("Staged disk read failed")
                return read(file)

            with patch.object(assembler, "_read_window", side_effect=failing_read):
                with self.assertRaisesRegex(OSError, "Staged disk read failed"):
                    assembler.publish("scan", store, "actual", "scan")
            self.assertEqual(store.list_datasets(), [("previous", "complete")])
            self.assertEqual(list((store.path / "actual").iterdir()), [])
            self.assertEqual(len(assembler.rows), 3)
            assembler.publish("scan", store, "actual", "scan")
            np.testing.assert_array_equal(store.read_dataset("actual", "scan").data, [0., 1., 2.])

    def test_manifest_failure_keeps_a_complete_but_unlisted_file(self):
        """Retain existing post-rename failure semantics and direct-file recovery."""
        schema = PortSchema("float64", 0, (), (), 64)
        with staged_scan(schema, [(np.asarray(3.), (), ())]) as (assembler, store):
            with patch.object(store, "_publish_manifest", side_effect=OSError("Manifest unavailable")):
                with self.assertRaisesRegex(OSError, "Manifest unavailable"):
                    assembler.publish("scan", store, "actual", "scan")
            self.assertEqual(store.list_datasets(), [])
            file = store.path / "actual" / "scan.dat"
            np.testing.assert_array_equal(AuspexStore.read_file(file).data, [3.])
            self.assertEqual(list(file.parent.iterdir()), [file])
            self.assertEqual(len(assembler.rows), 1)

    def test_variable_rank_two_shapes_are_rejected_without_publication(self):
        schema = PortSchema("float64", 2, ("x", "y"), (None, None), 1024)
        rows = [(np.zeros((n, 2)), (Axis("x", np.arange(n)), Axis("y", np.arange(2))), ())
                for n in (1, 2)]
        with staged_scan(schema, rows) as (assembler, store):
            with self.assertRaisesRegex(PipelineError, "rank-one"):
                assembler.publish("scan", store, "actual", "scan")
            self.assertEqual(store.list_datasets(), [])

    def test_publication_memory_is_window_bounded_for_all_layouts(self):
        """Measure allocations during real disk assembly, excluding staging and later full reads."""
        count, length = 96, 8192
        schema = PortSchema("complex64", 1, ("frequency",), ("Hz",), 256 * 1024)
        for kind in ("shared", "dependent", "ragged"):
            with self.subTest(kind=kind):
                def rows():
                    for i in range(count):
                        n = length + (i % 2 if kind == "ragged" else 0)
                        frequency = np.arange(n, dtype="f8") + (i if kind == "dependent" else 0)
                        yield (np.full(n, i + 1j, dtype="c8"), (Axis("frequency", frequency, "Hz"),),
                               (SweepPosition("bias", i),))

                with staged_scan(schema, rows()) as (assembler, store):
                    self.assertGreater(assembler.total_bytes, 12 * 1024**2 - 1)
                    tracemalloc.start()
                    try:
                        with (patch("auspex_v2.filters.scan.np.stack", side_effect=AssertionError("Stacked scan")),
                              patch("auspex_v2.filters.scan.np.concatenate", side_effect=AssertionError("Concatenated scan")),
                              patch.object(store, "write_dataset", side_effect=AssertionError("Whole-array writer"))):
                            assembler.publish("scan", store, "actual", "scan")
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    self.assertLess(peak, 8 * 1024**2, f"{kind}: publication allocated {peak} bytes")
                    loaded = store.read_dataset("actual", "scan")
                    self.assertEqual(loaded.data.dtype, np.dtype("complex64"))
                    if kind == "ragged":
                        self.assertEqual(len(loaded.offsets), count + 1)
                        np.testing.assert_array_equal(loaded.trace(95), np.full(length + 1, 95 + 1j))
                    else:
                        self.assertEqual(loaded.data.shape, (count, length))
                        np.testing.assert_array_equal(loaded.data[95], np.full(length, 95 + 1j))
                    scratch = Path(assembler.scratch.name)
                self.assertFalse(scratch.exists())
