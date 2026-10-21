"""Offline V2 .auspex storage tests; no V1 imports or hardware."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from auspex_v2.data import Axis, AuspexStore, FORMAT_VERSION


class DataStoreTest(unittest.TestCase):
    """Verify versioned publication, coordinates, and exclusive writes."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "run.auspex"

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_multiple_groups_and_complex_trace(self):
        """Keep actual coordinates and complex dtype across a fresh reader."""
        frequency = [4.0e9, 4.1e9, 4.25e9]
        trace = np.array([1+2j, 3-4j, 5+0j], dtype=np.complex128)
        with AuspexStore(self.path, mode="w") as store:
            store.write_dataset("vna", "s21", trace, [Axis("frequency", frequency, "Hz")])
            store.write_dataset("other", "power", np.array([[1, 2], [3, 4]]),
                                [Axis("time", [0, 1], "s"), Axis("channel", [1, 2])])
            self.assertEqual(len(store.list_datasets()), 2)
        with AuspexStore(self.path) as store:
            self.assertEqual(len(store.list_datasets()), 2)
            loaded = store.read_dataset("vna", "s21")
            np.testing.assert_array_equal(loaded.data, trace)
            self.assertEqual(loaded.axes[0].points, frequency)
            self.assertEqual(loaded.axes[0].unit, "Hz")
            np.testing.assert_array_equal(store.read_dataset("other", "power").data, [[1, 2], [3, 4]])
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.assertEqual(manifest["format_version"], FORMAT_VERSION)
        self.assertTrue((self.path / "vna" / "s21.dat").exists())

    def test_reopen_append_and_exclusive_writer(self):
        """Allow sequential writes but reject concurrent writers and duplicates."""
        with AuspexStore(self.path, mode="w") as writer:
            with self.assertRaises(RuntimeError):
                AuspexStore(self.path, mode="a")
            writer.write_dataset("g", "x", [1, 2], [Axis("n", [0, 1])])
            with self.assertRaises(FileExistsError):
                writer.write_dataset("g", "x", [3, 4], [Axis("n", [0, 1])])
        with AuspexStore(self.path, mode="a") as writer:
            writer.write_dataset("g", "y", np.array(7.0), [])
        with AuspexStore(self.path) as reader:
            self.assertEqual(reader.read_dataset("g", "y").data.item(), 7.0)
            self.assertEqual(len(reader.list_datasets()), 2)
            with self.assertRaises(PermissionError):
                reader.write_dataset("g", "z", [1], [Axis("n", [0])])

    def test_unpublished_files_are_invisible_and_not_overwritten(self):
        """A crashed array write must not become a readable dataset."""
        with AuspexStore(self.path, mode="w") as store:
            (self.path / "g").mkdir()
            (self.path / "g" / "orphan.dat").write_bytes(b"old")
            self.assertEqual(store.list_datasets(), [])
            with self.assertRaises(FileExistsError):
                store.write_dataset("g", "orphan", [1], [Axis("n", [0])])
            with self.assertRaises(KeyError):
                store.read_dataset("g", "orphan")

    def test_reject_invalid_data_and_paths(self):
        """Reject ambiguous coordinates, object data, and traversal names before publishing."""
        with AuspexStore(self.path, mode="w") as store:
            cases = [lambda: store.write_dataset("../g", "x", [1], [Axis("n", [0])]),
                     lambda: store.write_dataset("g", "../x", [1], [Axis("n", [0])]),
                     lambda: store.write_dataset("g", "x", [1, 2], [Axis("n", [0])]),
                     lambda: store.write_dataset("g", "x", [1], [Axis("n", [float("nan")])]),
                     lambda: store.write_dataset("g", "x", np.array([object()]), [Axis("n", [0])])]
            for case in cases:
                with self.assertRaises((ValueError, TypeError)):
                    case()
            self.assertEqual(store.list_datasets(), [])

    def test_reject_version_and_truncated_array(self):
        """Fail clearly on an unknown version or a damaged published array."""
        with AuspexStore(self.path, mode="w") as store:
            store.write_dataset("g", "x", [1, 2], [Axis("n", [0, 1])])
        (self.path / "g" / "x.dat").write_bytes(b"bad")
        with AuspexStore(self.path) as store:
            with self.assertRaises(ValueError):
                store.read_dataset("g", "x")
        manifest = self.path / "manifest.json"
        content = json.loads(manifest.read_text())
        content["format_version"] = FORMAT_VERSION + 1
        manifest.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            AuspexStore(self.path)
