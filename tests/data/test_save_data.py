"""Round-trip and overwrite contracts for the standalone regular-array helper."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from auspex_v2.data import Axis, AuspexStore, save_data, inspect_folder


class SaveDataTest(unittest.TestCase):
    """Use actual stores and readers to check the manual saving workflow."""

    def test_complex_map_axis_order_units_comments_and_browser_discovery(self):
        """Keyword order, not alphabetical order, defines the stored dimensions."""
        with tempfile.TemporaryDirectory() as directory:
            z = np.arange(12).reshape(3, 4) * (1 + 2j)
            bias, frequency = [2., 0., -1.], [1., 2., 4., 8.]
            path = save_data(Path(directory) / "manual", z,
                             z_bias=bias, a_frequency=Axis("a_frequency", frequency, "GHz"),
                             comments="Manual scan\nComplex transmission: Δf")
            self.assertEqual(path, Path(directory) / "manual.auspex")
            self.assertTrue(path.is_dir())
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [("data", "data")])
                result = store.read_dataset("data", "data")
                np.testing.assert_array_equal(result.data, z)
                self.assertEqual(result.dimensions, ("z_bias", "a_frequency"))
                np.testing.assert_array_equal(result.coordinates["z_bias"].values, bias)
                np.testing.assert_array_equal(result.coordinates["a_frequency"].values, frequency)
                self.assertEqual(result.axes["a_frequency"].unit, "GHz")
                self.assertIsNone(result.axes["z_bias"].unit)
                self.assertEqual(store.read_run_metadata(),
                                 {"comments": "Manual scan\nComplex transmission: Δf"})
            self.assertTrue(inspect_folder(path).entries[0].plottable)

    def test_real_trace_and_scalar(self):
        """Lists work like arrays; scalar data needs no axes and suffixes stay stable."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.auspex"
            self.assertEqual(save_data(path, [3., 4.], time=Axis("time", [0., 0.5], "s")), path)
            with AuspexStore(path) as store:
                result = store.read_dataset("data", "data")
                np.testing.assert_array_equal(result.data, [3., 4.])
                self.assertEqual(result.axes["time"].unit, "s")
            self.assertFalse((path / "metadata.json").exists())
            scalar_path = save_data(Path(directory) / "scalar", 2 + 3j)
            with AuspexStore(scalar_path) as store:
                result = store.read_dataset("data", "data")
                self.assertEqual(result.data.shape, ())
                self.assertEqual(result.data.item(), 2 + 3j)

    def test_force_write_replaces_whole_store(self):
        """Overwrite is explicit and removes stale datasets and previous comments."""
        with tempfile.TemporaryDirectory() as directory:
            path = save_data(Path(directory) / "scan", [1, 2], x=[0, 1], comments="old")
            with AuspexStore(path, mode="a") as store:
                store.write_dataset("other", "trace", [9], [Axis("t", [0])])
            with self.assertRaises(FileExistsError):
                save_data(path, [5, 6], x=[0, 1])
            with AuspexStore(path) as store:
                self.assertEqual(len(store.list_datasets()), 2)
                np.testing.assert_array_equal(store.read_dataset("data", "data").data, [1, 2])
            self.assertEqual(save_data(path, [7], force_write=True, time=[2]), path)
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [("data", "data")])
                result = store.read_dataset("data", "data")
                np.testing.assert_array_equal(result.data, [7])
                self.assertEqual(result.dimensions, ("time",))
            self.assertFalse((path / "other").exists())
            self.assertFalse((path / "metadata.json").exists())
            save_data(path, [8], force_write=True, time=[3], comments="replacement")
            with AuspexStore(path) as store:
                self.assertEqual(store.read_run_metadata(), {"comments": "replacement"})

    def test_invalid_inputs_leave_no_new_folder_and_preserve_existing_store(self):
        """Validation must precede filesystem changes, especially forced replacement."""
        cases = [
            (np.ones((2, 3)), {"x": [0, 1]}),
            ([1, 2], {"x": [0]}),
            ([1, 2], {"x": [0, 1], "y": [2, 3]}),
            ([1, 2], {"x": Axis("other", [0, 1])}),
            ([1, 2], {"x": Axis("x", [0, 1], 3)}),
            ([1, 2], {"x": [0, np.nan]}),
            ([1, 2], {"x": [0, 1j]}),
            ([1, 2], {"x": [[0, 1]]}),
            ([1, 2], {"bad/name": [0, 1]}),
            (["a", "b"], {"x": [0, 1]}),
            ([[1], [2, 3]], {"x": [0, 1]}),
            ([1, 2], {"x": [0, 1], "comments": 42}),
            ([1, 2], {"x": [0, 1], "comments": "a" * (1024 * 1024)}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            existing = save_data(Path(directory) / "existing", [9], x=[3], comments="keep")
            new = Path(directory) / "new.auspex"
            for data, kwargs in cases:
                with self.subTest(kwargs=tuple(kwargs), shape=np.shape(data) if not isinstance(data, list) else "list"):
                    for path in (new, existing):
                        with self.assertRaises((TypeError, ValueError)):
                            save_data(path, data, force_write=True, **kwargs)
                    self.assertFalse(new.exists())
                    with AuspexStore(existing) as store:
                        np.testing.assert_array_equal(store.read_dataset("data", "data").data, [9])
                        self.assertEqual(store.read_run_metadata(), {"comments": "keep"})

    def test_force_write_respects_writer_and_rejects_unrelated_directory(self):
        """Explicit overwrite must not bypass live ownership or erase arbitrary folders."""
        with tempfile.TemporaryDirectory() as directory:
            path = save_data(Path(directory) / "scan", [1], x=[0])
            with AuspexStore(path, mode="a"):
                with self.assertRaisesRegex(RuntimeError, "Another writer"):
                    save_data(path, [2], x=[0], force_write=True)
            with AuspexStore(path) as store:
                np.testing.assert_array_equal(store.read_dataset("data", "data").data, [1])
            unrelated = Path(directory) / "unrelated.auspex"
            unrelated.mkdir()
            marker = unrelated / "notes.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                save_data(unrelated, [2], x=[0], force_write=True)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
