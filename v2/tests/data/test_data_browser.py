"""Read-only discovery and optional offscreen DataBrowser tests."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from auspex_v2.data.browser import inspect_folder
from auspex_v2.data import AuspexStore, Axis, Coordinate
from . import gen_test_data


def populate(path, *, extra=False):
    """Create one published run with 1D, multidimensional and ragged values."""
    with AuspexStore(path, mode="w") as store:
        store.write_run_metadata({"experiment": {"name": "resonance", "settings": {"power": -30}}})
        store.write_dataset("data", "trace", np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
                            [Axis("frequency", [4.1e9, 4.13e9, 4.3e9], "Hz")],
                            value_name="S21", value_unit="dimensionless")
        store.write_dataset("data", "scan", np.ones((2, 3), dtype=np.float64),
                            [Axis("bias", [0., 0.2], "V"), Axis("frequency", [4.1, 4.3, 4.8], "GHz")])
        store.write_dataset("data", "ragged", np.array([1, 2, 3], dtype=np.float32),
                            coordinates=(Coordinate("point", [0, 1], ("point",)),
                                         Coordinate("frequency", [1, 2, 3], ("sample",), "Hz")),
                            dimensions=("point",), offsets=[0, 1, 3])
        if extra:
            store.write_dataset("other", "voltage", np.array([0.1, 0.2]),
                                [Axis("time", [1., 1.25], "s")], value_name="voltage", value_unit="V")


class BrowserDiscoveryTest(unittest.TestCase):
    """Inspect published headers without loading the full numerical arrays."""

    def test_supported_versus_unplottable(self):
        """Keep ragged and 2D files visible but refuse to represent them as 1D."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            populate(path)
            contents = inspect_folder(path)
            self.assertEqual(contents.metadata["experiment"]["settings"]["power"], -30)
            entries = {e.name: e for e in contents.entries}
            self.assertTrue(entries["trace"].plottable)
            self.assertFalse(entries["scan"].plottable)
            self.assertFalse(entries["ragged"].plottable)
            self.assertEqual(entries["trace"].header["value_name"], "S21")
            self.assertEqual(entries["trace"].header["arrays"]["value"]["shape"], [3])


class BrowserDemoGeneratorTest(unittest.TestCase):
    """Keep synthetic V2 demo runs reproducible and independent of GUI startup."""

    def test_two_noisy_runs_complex_and_ramsey_are_reusable(self):
        """The visual demo receives two real stores, not missing import paths."""
        with tempfile.TemporaryDirectory() as directory:
            resonance, ramsey = gen_test_data.generate_test_data(directory)
            self.assertEqual([resonance.name, ramsey.name], ["resonances.auspex", "ramsey.auspex"])
            with AuspexStore(resonance) as store:
                self.assertEqual(set(store.list_datasets()),
                                 {("traces", "three_lorentzians"), ("traces", "complex_s21")})
                trace = store.read_dataset("traces", "complex_s21")
                magnitude = store.read_dataset("traces", "three_lorentzians")
                self.assertEqual(trace.data.dtype, np.dtype("complex128"))
                self.assertGreater(np.std(trace.data.imag), 0.01)
                self.assertGreater(np.std(np.diff(magnitude.data)), 0.001)
                self.assertEqual(trace.axes[0].unit, "Hz")
                saved = magnitude.data.copy()
            with AuspexStore(ramsey) as store:
                sequence = store.read_dataset("qubit", "ramsey_beating")
                self.assertEqual(sequence.axes[0].name, "delay")
                self.assertGreater(np.std(sequence.data), 0.01)
            self.assertEqual(gen_test_data.generate_test_data(directory), (resonance, ramsey))
            with AuspexStore(resonance) as store:
                np.testing.assert_array_equal(store.read_dataset("traces", "three_lorentzians").data, saved)
            (ramsey / "metadata.json").unlink()
            with self.assertRaisesRegex(ValueError, "not the expected demo"):
                gen_test_data.generate_test_data(directory)


@unittest.skipUnless(importlib.util.find_spec("PySide6") and importlib.util.find_spec("pyqtgraph"),
                     "install auspex-v2[gui] for browser UI tests")
class BrowserGuiTest(unittest.TestCase):
    """Verify folder discovery, real coordinates, tab lifetime and manual refresh."""

    def test_visual_demo_entry_point_generates_both_runs(self):
        """Exercise the moved visual script's generator and utility import paths."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        script = Path(__file__).with_name("visual_data_browser.py")
        spec = importlib.util.spec_from_file_location("visual_data_browser", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            paths = module.generate_test_data(directory)
            self.assertEqual([path.name for path in paths], ["resonances.auspex", "ramsey.auspex"])
            self.assertEqual(module.generate_test_data(directory), paths)

    def test_multiple_folders_tabs_and_metadata(self):
        """Closing plots must not close their store or erase published inventory."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets, QtGui
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
        import data_browser as browser

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        browser.dark_theme(app)
        self.assertEqual(app.palette().color(QtGui.QPalette.ColorRole.Window).name(), "#202632")
        with tempfile.TemporaryDirectory() as directory:
            a, b = (Path(directory) / "one.auspex", Path(directory) / "two.auspex")
            populate(a)
            populate(b, extra=True)
            window = browser.DataBrowser([a, b])
            try:
                self.assertEqual(window.tree.topLevelItemCount(), 2)
                root_a, root_b = window.tree.topLevelItem(0), window.tree.topLevelItem(1)
                items_a = {root_a.child(0).child(i).data(0, browser.ROLE_KEY)[3]: root_a.child(0).child(i)
                           for i in range(root_a.child(0).childCount())}
                for name in ("scan", "ragged"):
                    self.assertIn("plotting not yet supported", items_a[name].text(0))
                    items_a[name].setSelected(True)  # still selectable for metadata inspection
                    window._open_item(items_a[name], 0)
                    self.assertEqual(window.tabs.count(), 0)
                window._open_item(items_a["trace"], 0)
                self.assertEqual(window.tabs.count(), 1)
                tab = window.tabs.widget(0)
                np.testing.assert_array_equal(tab.curve.xData, [4.1e9, 4.13e9, 4.3e9])
                np.testing.assert_allclose(tab.curve.yData, np.abs([1+2j, 3+4j, 5+6j]), rtol=1e-6)
                tab.component.setCurrentText("Imaginary")
                np.testing.assert_array_equal(tab.curve.yData, [2., 4., 6.])
                window.tree.setCurrentItem(items_a["trace"])
                self.assertEqual(window.metadata.topLevelItemCount(), 3)
                self.assertEqual(window.metadata.topLevelItem(2).text(0), "dataset header")
                window._open_item(items_a["trace"], 0)  # focus existing tab, never duplicate
                self.assertEqual(window.tabs.count(), 1)
                other = next(root_b.child(0).child(i) for i in range(root_b.child(0).childCount())
                             if root_b.child(0).child(i).data(0, browser.ROLE_KEY)[3] == "trace")
                window._open_item(other, 0)
                self.assertEqual(window.tabs.count(), 2)
                original_tab = tab
                with AuspexStore(a, mode="a") as store:
                    store.write_dataset("new_group", "later", np.array([1., 2.]),
                                        [Axis("time", [0., 1.], "s")])
                window.refresh()
                self.assertIs(window.open_tabs[(a, "data", "trace")], original_tab)
                self.assertEqual(window.tree.topLevelItem(0).childCount(), 2)
                menu = window._folder_menu(a)
                self.assertEqual(menu.actions()[0].text(), "Close all datasets")
                menu.actions()[0].trigger()
                self.assertEqual(window.tabs.count(), 1)
                self.assertEqual(window._key(window.tree.currentItem()),
                                 ("dataset", str(b), "data", "trace"))
                self.assertNotIn((a, "data", "trace"), window.open_tabs)
                self.assertEqual(window.tree.topLevelItemCount(), 2)
                window.close_tab(0)
                self.assertFalse(window.open_tabs)
            finally:
                window.close()

    def test_tab_and_dataset_selection_stay_in_sync(self):
        """Tabs select tree items; selecting an open item focuses its tab and metadata."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
        import data_browser as browser

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "runs"
            parent.mkdir()
            a, b = parent / "a.auspex", parent / "b.auspex"
            populate(a)
            populate(b)
            window = browser.DataBrowser([parent])
            try:
                a_key = ("dataset", str(a), "data", "trace")
                b_key = ("dataset", str(b), "data", "trace")
                scan_key = ("dataset", str(a), "data", "scan")
                window._open_item(window.tree_items[a_key], 0)
                window._open_item(window.tree_items[b_key], 0)
                self.assertEqual(window._key(window.tree.currentItem()), b_key)
                window.tabs.setCurrentIndex(0)
                self.assertEqual(window._key(window.tree.currentItem()), a_key)
                self.assertEqual(window.metadata.topLevelItem(2).text(0), "dataset header")
                window.tree.setCurrentItem(window.tree_items[b_key])
                self.assertEqual(window.tabs.currentWidget().key, (b, "data", "trace"))
                window.tree.setCurrentItem(window.tree_items[scan_key])
                self.assertEqual(window.tabs.currentWidget().key, (b, "data", "trace"))
                self.assertEqual(window._key(window.tree.currentItem()), scan_key)
                window.tabs.setCurrentIndex(0)
                window.close_tab(0)
                self.assertEqual(window._key(window.tree.currentItem()), b_key)
                self.assertEqual(window.tabs.currentWidget().key, (b, "data", "trace"))
                window.refresh_button.click()
                self.assertEqual(window._key(window.tree.currentItem()), b_key)
            finally:
                window.close()

    def test_multiple_watched_folders_refresh_independently(self):
        """One refresh discovers direct children in every watched OS directory."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
        import data_browser as browser

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            populate(first / "a.auspex")
            populate(second / "b.auspex")
            window = browser.DataBrowser([first, second])
            try:
                self.assertEqual(window.tree.topLevelItemCount(), 2)
                self.assertEqual([window.tree.topLevelItem(i).childCount() for i in range(2)], [1, 1])
                populate(second / "c.auspex")
                window.refresh_button.click()
                self.assertEqual([window.tree.topLevelItem(i).childCount() for i in range(2)], [1, 2])
                window.remove_directory(first)
                self.assertEqual(window.tree.topLevelItemCount(), 1)
                self.assertEqual(window.tree.topLevelItem(0).childCount(), 2)
                self.assertTrue((first / "a.auspex" / "manifest.json").exists())
            finally:
                window.close()

    def test_watch_children_refresh_button_and_remove_without_deleting(self):
        """Scan only direct runs; remove a watched OS folder and its plots, not files."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
        import data_browser as browser

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "runs"
            parent.mkdir()
            a = parent / "one.auspex"
            b = Path(directory) / "separate.auspex"
            nested = parent / "nested"
            nested.mkdir()
            populate(a)
            populate(b)
            populate(nested / "deep.auspex")
            window = browser.DataBrowser()
            try:
                window.add_directory(parent)
                self.assertEqual(window.tree.topLevelItemCount(), 1)
                watched = window.tree.topLevelItem(0)
                self.assertEqual(watched.data(0, browser.ROLE_KEY)[0], "directory")
                self.assertEqual(watched.childCount(), 1)  # no recursive traversal
                window.tree.setCurrentItem(watched.child(0))
                window._remove_selected_folder()
                self.assertIn(parent, window.watched_directories)  # removing a child must not remove its parent
                window.add_run(b)
                window.add_run(a)  # also pin a; do not display it twice
                self.assertEqual(window.tree.topLevelItemCount(), 2)
                watched = window.tree.topLevelItem(0)
                a_node = watched.child(0)
                trace = next(a_node.child(0).child(i) for i in range(a_node.child(0).childCount())
                             if a_node.child(0).child(i).data(0, browser.ROLE_KEY)[3] == "trace")
                window._open_item(trace, 0)
                b_node = window.tree.topLevelItem(1)
                other = next(b_node.child(0).child(i) for i in range(b_node.child(0).childCount())
                             if b_node.child(0).child(i).data(0, browser.ROLE_KEY)[3] == "trace")
                window._open_item(other, 0)
                self.assertEqual(window.tabs.count(), 2)
                populate(parent / "later.auspex")
                window.refresh_button.click()
                watched = window.tree.topLevelItem(0)
                self.assertEqual(watched.childCount(), 2)
                self.assertEqual(window.tabs.count(), 2)
                with AuspexStore(a, mode="a") as store:
                    store.write_dataset("new_group", "latest", np.array([1., 2.]),
                                        [Axis("time", [0., 1.], "s")])
                file_menu = window.menuBar().actions()[0].menu()
                refresh = next(action for action in file_menu.actions() if "Refresh all" in action.text())
                refresh.trigger()
                a_node = next(window.tree.topLevelItem(0).child(i)
                              for i in range(window.tree.topLevelItem(0).childCount())
                              if window.tree.topLevelItem(0).child(i).data(0, browser.ROLE_KEY)[1] == str(a))
                self.assertEqual(a_node.childCount(), 2)
                self.assertEqual(window.tabs.count(), 2)  # already open plots stay unchanged
                menu = window._directory_menu(parent)
                self.assertEqual(menu.actions()[0].text(), "Remove folder")
                menu.actions()[0].trigger()
                self.assertEqual(window.tabs.count(), 1)  # a closed even though directly pinned
                self.assertNotIn(parent, window.watched_directories)
                self.assertEqual(window.tree.topLevelItemCount(), 2)  # pinned a and separate b remain
                self.assertTrue((a / "manifest.json").exists())
                self.assertTrue((parent / "later.auspex" / "manifest.json").exists())
                direct_a = next(window.tree.topLevelItem(i)
                                for i in range(window.tree.topLevelItemCount())
                                if window.tree.topLevelItem(i).data(0, browser.ROLE_KEY)[1] == str(a))
                window.tree.setCurrentItem(direct_a)
                window._remove_selected_folder()
                self.assertEqual(window.tree.topLevelItemCount(), 1)
                self.assertEqual(window.tabs.count(), 1)
                self.assertTrue((a / "manifest.json").exists())
            finally:
                window.close()
