"""Real-store lookup and loaded Dataset contracts for the manual open helper."""

from datetime import date
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from auspex_v2.data import Axis, AuspexStore, Coordinate, Dataset, open_data, save_data


class OpenDataTest(unittest.TestCase):
    """Check resolution and selection without guessing a folder or dataset."""

    def test_explicit_path_loads_detached_complex_dataset_without_gui(self):
        """Loading needs neither Qt nor an open store after the helper returns."""
        with tempfile.TemporaryDirectory() as directory:
            values = np.arange(6).reshape(2, 3) * (1 + 2j)
            path = save_data(Path(directory) / "manual", values,
                             bias=[-1., 1.], frequency=Axis("frequency", [2., 3., 5.], "Hz"))
            with patch.dict("sys.modules", {"PySide6": None, "pyqtgraph": None}):
                loaded = open_data(path=path)
            self.assertIsInstance(loaded, Dataset)
            self.assertEqual(loaded.dimensions, ("bias", "frequency"))
            self.assertEqual(loaded.axes["frequency"].unit, "Hz")
            np.testing.assert_array_equal(loaded.data, values)
            with AuspexStore(path, mode="a"):
                loaded.data[0, 0] = 99
            np.testing.assert_array_equal(open_data(path=str(path)).data, values)

    def test_number_lookup_dates_zero_padding_and_nonrecursive_search(self):
        """Only immediate folders with the exact numeric suffix count as matches."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_data(root / "prefix_with_underscores_007", [7], x=[0])
            save_data(root / "2024-02-29" / "dated_7", [29], x=[0])
            today = date.today().isoformat()
            save_data(root / today / "today_007", [42], x=[0])
            save_data(root / "other" / "nested_7", [-1], x=[0])
            (root / "not_a_folder_7.auspex").write_text("ignored", encoding="utf-8")
            save_data(root / "not_numbered_7_extra", [-2], x=[0])
            np.testing.assert_array_equal(open_data(path=root, number=7).data, [7])
            np.testing.assert_array_equal(open_data(path=root, number=7, date=None).data, [7])
            np.testing.assert_array_equal(open_data(path=root, number=7, date="2024-02-29").data, [29])
            np.testing.assert_array_equal(open_data(path=root, number=7, date="today").data, [42])
            np.testing.assert_array_equal(open_data(path=root, number=7, date=today).data, [42])
            with self.assertRaises(FileNotFoundError):
                open_data(path=root, number=8)
            with self.assertRaises(FileNotFoundError):
                open_data(path=root, number=7, date="1999-01-01")
            save_data(root / "another_7", [8], x=[0])
            with self.assertRaisesRegex(ValueError, "Ambiguous folder.*another_7.*prefix_with_underscores_007"):
                open_data(path=root, number=7)

    def test_dataset_selection_and_ambiguity(self):
        """Partial selectors are allowed only when they identify a single dataset."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiple.auspex"
            with AuspexStore(path, mode="w") as store:
                store.write_dataset("a", "trace", [1], [Axis("t", [0])])
                store.write_dataset("b", "trace", [2], [Axis("t", [0])])
                store.write_dataset("b", "other", [3], [Axis("t", [0])])
            for selectors in ({}, {"dataset": "trace"}, {"group": "b"}):
                with self.subTest(selectors=selectors):
                    with self.assertRaisesRegex(ValueError, "Ambiguous dataset.*a/trace.*b/trace.*b/other"):
                        open_data(path=path, **selectors)
            for selectors, expected in (({"group": "a"}, 1), ({"dataset": "other"}, 3),
                                        ({"group": "b", "dataset": "trace"}, 2)):
                np.testing.assert_array_equal(open_data(path=path, **selectors).data, [expected])
            with self.assertRaisesRegex(FileNotFoundError, "No matching dataset.*available:"):
                open_data(path=path, group="missing")
            empty = Path(directory) / "empty.auspex"
            with AuspexStore(empty, mode="w"):
                pass
            with self.assertRaisesRegex(FileNotFoundError, "available: \\(none\\)"):
                open_data(path=empty)

    def test_argument_errors_do_not_invoke_picker(self):
        """Bad lookup arguments fail before importing GUI code or guessing paths."""
        with patch("auspex_v2.data.manual._pick_data_folder", side_effect=AssertionError("unexpected picker")):
            with self.assertRaisesRegex(ValueError, "number requires path"):
                open_data(number=7)
            with self.assertRaisesRegex(ValueError, "date requires number"):
                open_data(date="today")
            for number in (-1, True, 1.5, "7"):
                with self.subTest(number=number), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    open_data(path="unused", number=number)
            for invalid_date in ("20240229", "2024-2-29", "2023-02-29", "../2024-02-29", 2024):
                with self.subTest(date=invalid_date), self.assertRaises(ValueError):
                    open_data(path="unused", number=7, date=invalid_date)

    def test_picker_selection_and_cancellation(self):
        """Only the dialog boundary is replaced; loading still reads an actual store."""
        with tempfile.TemporaryDirectory() as directory:
            path = save_data(Path(directory) / "selected", [5], t=[1])
            with patch("auspex_v2.data.manual._pick_data_folder", return_value=str(path)) as picker:
                np.testing.assert_array_equal(open_data().data, [5])
                picker.assert_called_once_with()
            with patch("auspex_v2.data.manual._pick_data_folder", return_value=""):
                self.assertIsNone(open_data())
            with patch.dict("sys.modules", {"PySide6": None}):
                with self.assertRaisesRegex(ImportError, r"requires auspex-v2\[gui\].*pass path="):
                    open_data()

    def test_ragged_dataset_passes_through_existing_reader(self):
        """The loading helper must not impose the regular-only save helper's limits."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ragged.auspex"
            with AuspexStore(path, mode="w") as store:
                store.write_dataset("g", "ragged", [1., 2., 3.], dimensions=("point",),
                                    coordinates=(Coordinate("point", [0, 1], ("point",)),
                                                 Coordinate("t", [0, 1, 0], ("sample",), "s")),
                                    offsets=[0, 2, 3])
            loaded = open_data(path=path)
            np.testing.assert_array_equal(loaded.trace(0), [1, 2])
            np.testing.assert_array_equal(loaded.offsets, [0, 2, 3])

    @unittest.skipUnless(importlib.util.find_spec("PySide6"), "install auspex-v2[gui] for picker test")
    def test_qt_picker_adapter(self):
        """Exercise QApplication reuse and the real Qt dialog entry point offscreen."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        from auspex_v2.data.manual import _pick_data_folder
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value="") as dialog:
            self.assertEqual(_pick_data_folder(), "")
            dialog.assert_called_once()
            self.assertIs(QtWidgets.QApplication.instance(), app)
