"""Offline 2D heatmaps on stored rectilinear coordinates; no interpolation."""

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg


MODES = {
    "Real": ("Real",),
    "Imaginary": ("Imaginary",),
    "Magnitude": ("Magnitude",),
    "Phase": ("Phase",),
    "Real / Imaginary": ("Real", "Imaginary"),
    "Magnitude / Phase": ("Magnitude", "Phase"),
    "All components": ("Real", "Imaginary", "Magnitude", "Phase"),
}


def coordinate_edges(values):
    """Use midpoint cell boundaries, extrapolating half a spacing at each end."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Heatmap coordinates must be nonempty and finite")
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    delta = np.diff(values)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("Heatmap coordinates must be strictly monotonic")
    return np.r_[values[0] - delta[0] / 2, values[:-1] + delta / 2,
                 values[-1] + delta[-1] / 2]


def component_values(data, component):
    """Derive display values without modifying the stored complex array."""
    return {"Real": np.real, "Imaginary": np.imag,
            "Magnitude": np.abs, "Phase": np.angle}[component](data)


def color_levels(values, component):
    """Keep phase comparable across views and signed components centered on zero."""
    finite = values[np.isfinite(values)]
    if component == "Phase":
        return (-np.pi, np.pi)
    maximum = float(np.max(np.abs(finite))) if finite.size else 0.0
    if component in ("Real", "Imaginary"):
        return (-maximum, maximum) if maximum else (-1.0, 1.0)
    return (0.0, maximum or 1.0)


def axis_label(coordinate):
    return coordinate.name + (f" ({coordinate.unit})" if coordinate.unit else "")


class HeatmapTab(QtWidgets.QWidget):
    """Own one loaded 2D dataset, component heatmaps and optional snapped cuts."""

    def __init__(self, key, dataset):
        """Validate dimension coordinates before constructing interactive views."""
        super().__init__()
        if dataset.data.ndim != 2 or dataset.offsets is not None:
            raise ValueError("Heatmaps require non-ragged 2D data")
        self.key, self.dataset = key, dataset
        self.coordinates = []
        for dim, size in zip(dataset.dimensions, dataset.data.shape):
            matches = [c for c in dataset.coordinates
                       if c.dimensions == (dim,) and np.shape(c.values) == (size,)]
            matches.sort(key=lambda c: c.name != dim)
            if not matches:
                raise ValueError(f"No one-dimensional coordinate for {dim}")
            self.coordinates.append(matches[0])
        self.y, self.x = [np.asarray(c.values) for c in self.coordinates]
        self.mesh_x, self.mesh_y = np.meshgrid(coordinate_edges(self.x), coordinate_edges(self.y))
        self.component = QtWidgets.QComboBox()
        if np.iscomplexobj(dataset.data):
            self.component.addItems(MODES)
            self.component.setCurrentText("Magnitude / Phase")
        else:
            self.component.addItem("Real")
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(sorted(name for name in pg.colormap.listMaps()
                                     if not name.startswith(("CET", "PAL"))))
        self.colormap.setCurrentText("plasma")
        self.x_cut = QtWidgets.QCheckBox("Z versus X (fixed Y)")
        self.y_cut = QtWidgets.QCheckBox("Z versus Y (fixed X)")
        controls = QtWidgets.QHBoxLayout()
        for widget in (QtWidgets.QLabel("Component:"), self.component,
                       QtWidgets.QLabel("Colormap:"), self.colormap, self.x_cut, self.y_cut):
            controls.addWidget(widget)
        controls.addStretch()
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("#171b24")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.graphics, 1)
        self.indices = {"X": len(self.y) // 2, "Y": len(self.x) // 2}
        self.panels, self.lines, self.cut_curves = [], {}, {}
        self.component.currentIndexChanged.connect(self.redraw)
        self.colormap.currentIndexChanged.connect(self.update_colormap)
        self.x_cut.toggled.connect(self.redraw)
        self.y_cut.toggled.connect(self.redraw)
        self.redraw()

    def redraw(self, *_):
        """Rebuild only display objects; retain cut positions across mode changes."""
        self.graphics.clear()
        self.panels, self.lines, self.cut_curves = [], {}, {}
        components = MODES[self.component.currentText()]
        single = len(components) == 1
        self.x_cut.setEnabled(single)
        self.y_cut.setEnabled(single)
        cmap = pg.colormap.get(self.colormap.currentText())
        columns = 2 if len(components) == 4 else 1
        for index, component in enumerate(components):
            row, column = divmod(index, columns)
            plot = self.graphics.addPlot(row=row, col=column * 2)
            units = "rad" if component == "Phase" else self.dataset.value_unit or ""
            plot.setTitle(f"{component}({self.dataset.value_name})" + (f" ({units})" if units else ""))
            plot.setLabel("bottom", axis_label(self.coordinates[1]))
            plot.setLabel("left", axis_label(self.coordinates[0]))
            values = component_values(self.dataset.data, component)
            levels = color_levels(values, component)
            mesh = pg.PColorMeshItem(self.mesh_x, self.mesh_y, values,
                                     colorMap=cmap, levels=levels, enableAutoLevels=False)
            plot.addItem(mesh)
            bar = pg.ColorBarItem(values=levels, colorMap=cmap, interactive=False)
            self.graphics.addItem(bar, row=row, col=column * 2 + 1)
            self.panels.append((plot, mesh, bar))
        if single:
            for direction, checkbox, positions, angle in (
                    ("X", self.x_cut, self.y, 0), ("Y", self.y_cut, self.x, 90)):
                if not checkbox.isChecked():
                    continue
                line = pg.InfiniteLine(pos=positions[self.indices[direction]], angle=angle,
                                       movable=True, bounds=(float(min(positions)), float(max(positions))),
                                       pen=pg.mkPen("#00e5ff" if direction == "X" else "#ff9e40", width=2))
                self.panels[0][0].addItem(line)
                self.lines[direction] = line
                cut = self.graphics.addPlot(row=len(self.cut_curves) + 1, col=0)
                coordinate = self.coordinates[1 if direction == "X" else 0]
                cut.setLabel("bottom", axis_label(coordinate))
                cut.setLabel("left", f"{components[0]}({self.dataset.value_name})", units=units)
                self.cut_curves[direction] = (cut, cut.plot(pen=line.pen))
                line.sigPositionChanged.connect(lambda _, d=direction: self.update_cut(d))
                self.update_cut(direction)

    def update_cut(self, direction):
        """Snap the movable line to a measured row/column, never interpolate Z."""
        line = self.lines[direction]
        positions = self.y if direction == "X" else self.x
        index = int(np.argmin(np.abs(positions - line.value())))
        self.indices[direction] = index
        line.blockSignals(True)
        line.setValue(float(positions[index]))
        line.blockSignals(False)
        values = component_values(self.dataset.data, self.component.currentText())
        plot, curve = self.cut_curves[direction]
        curve.setData(self.x if direction == "X" else self.y,
                      values[index, :] if direction == "X" else values[:, index])
        fixed = self.coordinates[0 if direction == "X" else 1]
        plot.setTitle(f"{fixed.name} = {positions[index]:.6g} {fixed.unit or ''}")

    def update_colormap(self, *_):
        """Apply one palette to every panel without resetting view or cut positions."""
        cmap = pg.colormap.get(self.colormap.currentText())
        for _, mesh, bar in self.panels:
            mesh.setColorMap(cmap)
            bar.setColorMap(cmap)

    def update_dataset(self, dataset):
        """Refresh compatible live arrays while preserving zoom, mode and cut choices."""
        if dataset.dimensions != self.dataset.dimensions or dataset.data.ndim != 2:
            raise ValueError("Live map dimensions changed")
        coordinates = [next(c for c in dataset.coordinates if c.dimensions == (dim,))
                       for dim in dataset.dimensions]
        y, x = [np.asarray(c.values) for c in coordinates]
        mesh_x, mesh_y = np.meshgrid(coordinate_edges(x), coordinate_edges(y))
        self.dataset, self.coordinates = dataset, coordinates
        self.y, self.x, self.mesh_x, self.mesh_y = y, x, mesh_x, mesh_y
        for component, (_, mesh, bar) in zip(MODES[self.component.currentText()], self.panels):
            values = component_values(dataset.data, component)
            levels = color_levels(values, component)
            mesh.setData(mesh_x, mesh_y, values)
            mesh.setLevels(levels)
            bar.setLevels(levels)
        for direction, line in self.lines.items():
            positions = y if direction == "X" else x
            line.setBounds((float(min(positions)), float(max(positions))))
            self.update_cut(direction)

    def release(self):
        """Release numerical arrays and graphics when the owning tab closes."""
        self.graphics.clear()
        self.panels, self.lines, self.cut_curves = [], {}, {}
        self.dataset = None
        self.coordinates = []
        self.x = self.y = self.mesh_x = self.mesh_y = None
