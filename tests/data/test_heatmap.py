"""Correctness and offscreen integration checks for rectilinear 2D displays."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from auspex_v2.data import Axis, AuspexStore


@unittest.skipUnless(importlib.util.find_spec("PySide6") and importlib.util.find_spec("pyqtgraph"),
                     "install auspex-v2[gui] for heatmap tests")
class HeatmapTest(unittest.TestCase):
    """Exercise real stored arrays through the browser, not a fake plotting backend."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
        import data_browser
        cls.browser = data_browser
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_complex_modes_mesh_scales_and_draggable_cuts(self):
        """Non-square, descending/nonuniform axes expose transposition and cut errors."""
        x, y = np.array([2., 3., 7., 10.]), np.array([5., 1., -2.])
        z = np.arange(12).reshape(3, 4) - 4 + 1j * np.arange(12, 24).reshape(3, 4)
        original = z.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.auspex"
            with AuspexStore(path, mode="w") as store:
                store.write_dataset("map", "response", z,
                                    [Axis("bias", y, "V"), Axis("frequency", x, "Hz")],
                                    value_name="S21", value_unit="V")
            window = self.browser.DataBrowser([path])
            try:
                item = window.tree.topLevelItem(0).child(0).child(0)
                window._open_item(item, 0)
                self.assertEqual(window.tabs.count(), 1)
                tab = window.tabs.widget(0)
                self.assertEqual(tab.component.currentText(), "Magnitude / Phase")
                self.assertEqual(tab.colormap.currentText(), "plasma")
                palettes = [tab.colormap.itemText(i) for i in range(tab.colormap.count())]
                self.assertFalse(any(name.startswith(("CET", "PAL")) for name in palettes))
                self.assertIn("viridis", palettes)
                self.assertEqual(len(tab.panels), 2)
                np.testing.assert_allclose(tab.panels[0][1].z, np.abs(z))
                np.testing.assert_allclose(tab.panels[1][1].z, np.angle(z))
                np.testing.assert_allclose(tab.panels[1][2].levels(), [-np.pi, np.pi])
                np.testing.assert_allclose(tab.mesh_x[0], [1.5, 2.5, 5, 8.5, 11.5])
                np.testing.assert_allclose(tab.mesh_y[:, 0], [7, 3, -0.5, -3.5])
                self.assertFalse(tab.x_cut.isChecked())
                self.assertFalse(tab.x_cut.isEnabled())
                for mode in ("Real", "Imaginary", "Magnitude", "Phase",
                             "Real / Imaginary", "All components"):
                    tab.component.setCurrentText(mode)
                    expected = {"Real": [z.real], "Imaginary": [z.imag],
                                "Magnitude": [np.abs(z)], "Phase": [np.angle(z)],
                                "Real / Imaginary": [z.real, z.imag],
                                "All components": [z.real, z.imag, np.abs(z), np.angle(z)]}[mode]
                    self.assertEqual(len(tab.panels), len(expected))
                    for (_, mesh, _), values in zip(tab.panels, expected):
                        np.testing.assert_allclose(mesh.z, values)
                tab.component.setCurrentText("Imaginary")
                self.assertFalse(tab.lines)
                tab.x_cut.setChecked(True)
                tab.y_cut.setChecked(True)
                tab.lines["X"].setValue(4.6)
                tab.lines["Y"].setValue(6.1)
                self.assertEqual(tab.lines["X"].value(), 5)
                self.assertEqual(tab.lines["Y"].value(), 7)
                np.testing.assert_array_equal(tab.cut_curves["X"][1].xData, x)
                np.testing.assert_array_equal(tab.cut_curves["X"][1].yData, z.imag[0])
                np.testing.assert_array_equal(tab.cut_curves["Y"][1].xData, y)
                np.testing.assert_array_equal(tab.cut_curves["Y"][1].yData, z.imag[:, 2])
                tab.component.setCurrentText("Phase")
                np.testing.assert_allclose(tab.cut_curves["X"][1].yData, np.angle(z)[0])
                tab.colormap.setCurrentText("viridis")
                for _, mesh, bar in tab.panels:
                    self.assertEqual(mesh.getColorMap().name, "viridis")
                tab.component.setCurrentText("All components")
                self.assertFalse(tab.lines)
                tab.component.setCurrentText("Real")
                self.assertEqual(set(tab.lines), {"X", "Y"})
                np.testing.assert_allclose(tab.panels[0][2].levels(), [-7, 7])
                tab.x_cut.setChecked(False)
                tab.y_cut.setChecked(False)
                self.assertFalse(tab.lines)
                np.testing.assert_array_equal(tab.dataset.data, original)
                window.close_tab(0)
                self.assertIsNone(tab.dataset)
                self.assertIsNone(tab.mesh_x)
            finally:
                window.close()

    def test_real_heatmap_and_coordinate_contract(self):
        """Real inputs remain single-panel; unsupported grids fail explicitly."""
        from src.data_heatmap import HeatmapTab, coordinate_edges, color_levels
        from auspex_v2.data import Dataset, Coordinate
        for bad in ([], [1, 1], [0, 2, 1], [0, np.nan]):
            with self.assertRaises(ValueError):
                coordinate_edges(bad)
        np.testing.assert_array_equal(coordinate_edges([3]), [2.5, 3.5])
        self.assertEqual(color_levels(np.zeros((2, 2)), "Real"), (-1, 1))
        self.assertEqual(color_levels(np.array([np.nan]), "Magnitude"), (0, 1))
        dataset = Dataset(np.array([[1., -2.], [3., 4.]]), axes=(),
                          coordinates=(Coordinate("y", [0, 1], ("y",)),
                                       Coordinate("x", [2, 4], ("x",))), dimensions=("y", "x"))
        tab = HeatmapTab(None, dataset)
        try:
            self.assertEqual(tab.component.count(), 1)
            self.assertEqual(len(tab.panels), 1)
            np.testing.assert_array_equal(tab.panels[0][1].z, dataset.data)
            self.assertTrue(tab.x_cut.isEnabled())
            self.assertFalse(tab.lines)
        finally:
            tab.release()
            tab.close()
        missing = Dataset(np.ones((2, 2)), axes=(), dimensions=("y", "x"))
        with self.assertRaisesRegex(ValueError, "No one-dimensional coordinate"):
            HeatmapTab(None, missing)
