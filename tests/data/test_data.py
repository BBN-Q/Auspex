"""Offline V2 .auspex storage tests; no V1 imports or hardware."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from auspex_v2.data import Axis, Coordinate, Dataset, AuspexStore, FORMAT_VERSION


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

    def test_metadata_sequences_support_names_without_changing_tuple_behavior(self):
        """Names and positions return the same objects; collections retain ordinary read-only sequence behavior."""
        axes = [Axis("bias", [0, 1], "V"), Axis("frequency", [4, 5, 6], "Hz")]
        coordinates = [Coordinate(a.name, np.asarray(a.points), (a.name,), a.unit) for a in axes]
        dataset = Dataset(np.zeros((2, 3)), axes, coordinates)
        for collection, originals in ((dataset.axes, axes), (dataset.coordinates, coordinates)):
            self.assertIs(collection["frequency"], originals[1])
            self.assertIs(collection["frequency"], collection[1])
            self.assertIs(collection[-1], collection[1])
            self.assertIs(collection[np.int64(1)], collection[1])
            self.assertEqual(len(collection), 2)
            self.assertEqual([item.name for item in collection], ["bias", "frequency"])
            self.assertEqual(collection[:1], (originals[0],))
            self.assertEqual(collection[::-1], tuple(reversed(originals)))
            self.assertEqual(collection, tuple(originals))
            with self.assertRaises(KeyError) as missing:
                collection["missing"]
            self.assertEqual(missing.exception.args, ("missing",))
            with self.assertRaises(KeyError):
                collection["Frequency"]
            for index in (2, -3):
                with self.assertRaises(IndexError):
                    collection[index]
            with self.assertRaises(TypeError):
                collection[0] = originals[1]
            with self.assertRaises(AttributeError):
                collection.append(originals[0])
        axes.clear()
        coordinates.clear()
        self.assertEqual(len(dataset.axes), 2)  # Snapshot the collection, not its numerical payloads.
        self.assertEqual(len(dataset.coordinates), 2)
        scalar = Dataset(np.asarray(1.0), ())
        self.assertEqual(scalar.axes, ())
        self.assertEqual(scalar.coordinates, ())
        with self.assertRaises(KeyError):
            scalar.coordinates["frequency"]

    def test_duplicate_metadata_names_are_rejected_per_collection(self):
        """Reject ambiguous lookup even for directly constructed datasets before serialization."""
        axis = Axis("frequency", [1, 2], "Hz")
        coordinate = Coordinate("frequency", np.asarray([1, 2]), ("frequency",), "Hz")
        with self.assertRaisesRegex(ValueError, "Duplicate metadata name"):
            Dataset(np.zeros((2, 2)), (axis, axis))
        with self.assertRaisesRegex(ValueError, "Duplicate metadata name"):
            Dataset(np.zeros(2), (axis,), (coordinate, coordinate))
        dataset = Dataset(np.zeros(2), (axis,), (coordinate,))
        self.assertIs(dataset.axes["frequency"], axis)  # Sharing a name across the two collections is normal.
        self.assertIs(dataset.coordinates["frequency"], coordinate)

    def test_named_lookup_round_trips_regular_saved_data_and_standalone_files(self):
        """Both file readers expose positional and named access without changing format version 2."""
        values = np.arange(6).reshape(2, 3)
        with AuspexStore(self.path, mode="w") as store:
            store.write_dataset("g", "trace", values,
                                [Axis("bias", [0, 1], "V"), Axis("frequency", [4, 5, 6], "Hz")])
        with AuspexStore(self.path) as store:
            loaded = store.read_dataset("g", "trace")
        direct = AuspexStore.read_file(self.path / "g" / "trace.dat")
        for dataset in (loaded, direct):
            self.assertIsInstance(dataset.axes["frequency"], Axis)
            self.assertIsInstance(dataset.coordinates["frequency"], Coordinate)
            self.assertIs(dataset.axes["frequency"], dataset.axes[1])
            self.assertIs(dataset.coordinates["frequency"], dataset.coordinates[1])
            self.assertEqual(dataset.axes["frequency"].unit, "Hz")
            np.testing.assert_array_equal(dataset.coordinates["frequency"].values, [4, 5, 6])
            np.testing.assert_array_equal(dataset.data, values)
        self.assertEqual(AuspexStore.read_file_header(self.path / "g" / "trace.dat")["format_version"], 2)

    def test_named_lookup_preserves_dependent_and_ragged_coordinate_semantics(self):
        """Dependent/sample coordinates stay available by name without becoming fictitious separable axes."""
        frequencies = np.asarray([[4, 5, 6], [4.1, 5.1, 6.1]])
        with AuspexStore(self.path, mode="w") as store:
            store.write_dataset("g", "dependent", np.ones((2, 3)), dimensions=("bias", "frequency"),
                coordinates=[Coordinate("bias", np.asarray([0, 1]), ("bias",), "V"),
                             Coordinate("frequency", frequencies, ("bias", "frequency"), "Hz"),
                             Coordinate("temperature", np.asarray([10, 11]), ("bias",), "K")])
            store.write_dataset("g", "ragged", np.arange(5), dimensions=("point",), offsets=np.asarray([0, 2, 5]),
                coordinates=[Coordinate("point", np.asarray([0, 1]), ("point",)),
                             Coordinate("frequency", np.asarray([4, 5, 4.1, 5.1, 6.1]), ("sample",), "Hz")])
        with AuspexStore(self.path) as store:
            dependent = store.read_dataset("g", "dependent")
            ragged = store.read_dataset("g", "ragged")
        self.assertEqual([a.name for a in dependent.axes], ["bias"])
        self.assertIs(dependent.coordinates["frequency"], dependent.coordinates[1])
        self.assertEqual(dependent.coordinates["frequency"].dimensions, ("bias", "frequency"))
        np.testing.assert_array_equal(dependent.coordinates["frequency"].values, frequencies)
        self.assertEqual(dependent.coordinates["temperature"].unit, "K")
        for dataset in (dependent, ragged):
            with self.assertRaises(KeyError):
                dataset.axes["frequency"]
        self.assertEqual(ragged.coordinates["frequency"].dimensions, ("sample",))
        self.assertIs(ragged.axes["point"], ragged.axes[0])
        np.testing.assert_array_equal(ragged.trace(1), [2, 3, 4])

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
