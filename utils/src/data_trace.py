"""Shared offline/live one-dimensional data renderer."""

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg

BACKGROUND = "#171b24"
INK = "#e7eef8"


class DataTab(QtWidgets.QWidget):
    """Hold one validated 1D dataset and plot against its actual coordinates."""

    def __init__(self, key, dataset):
        """Keep the offline coordinate/component controls usable for live updates too."""
        super().__init__()
        if dataset.data.ndim != 1 or dataset.offsets is not None:
            raise ValueError("Only non-ragged 1D datasets can be plotted")
        self.key = key
        self.dataset = dataset
        axis = dataset.dimensions[0]
        self.coordinates = [c for c in dataset.coordinates
                            if c.dimensions == (axis,) and np.asarray(c.values).shape == dataset.data.shape]
        if not self.coordinates:
            raise ValueError("The 1D dataset has no matching numeric coordinate")
        # Prefer a physical sweep coordinate over the synthetic 'point' index.
        self.coordinates.sort(key=lambda c: c.name == "point")
        self.x_axis = QtWidgets.QComboBox()
        for coordinate in self.coordinates:
            label = coordinate.name + (f" ({coordinate.unit})" if coordinate.unit else "")
            self.x_axis.addItem(label)
        self.x_axis.currentIndexChanged.connect(self.redraw)
        self.component = QtWidgets.QComboBox()
        if np.iscomplexobj(dataset.data):
            self.component.addItems(["Magnitude", "Real", "Imaginary", "Phase"])
        else:
            self.component.addItem("Value")
        self.component.currentIndexChanged.connect(self.redraw)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("X coordinate:"))
        controls.addWidget(self.x_axis)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Component:"))
        controls.addWidget(self.component)
        controls.addStretch()
        self.plot = pg.PlotWidget(background=BACKGROUND)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        for side in ("left", "bottom"):
            self.plot.getAxis(side).setPen(pg.mkPen(INK))
            self.plot.getAxis(side).setTextPen(pg.mkPen(INK))
        self.curve = self.plot.plot(pen=pg.mkPen("#79d4ff", width=2), connect="finite")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.plot, 1)
        self.redraw()

    def redraw(self, *_):
        """Select only a display component; never alter the loaded dataset."""
        if self.dataset is None:
            return
        coordinate = self.coordinates[self.x_axis.currentIndex()]
        component = self.component.currentText()
        values = self.dataset.data
        if component == "Magnitude":
            values = np.abs(values)
        elif component == "Real":
            values = np.real(values)
        elif component == "Imaginary":
            values = np.imag(values)
        elif component == "Phase":
            values = np.angle(values)
        label = self.dataset.value_name
        if component == "Magnitude":
            label = f"|{label}|"
        elif component != "Value":
            label = f"{component}({label})"
        units = "rad" if component == "Phase" else self.dataset.value_unit or ""
        self.plot.setLabel("bottom", coordinate.name, units=coordinate.unit or "")
        self.plot.setLabel("left", label, units=units)
        self.curve.setData(np.asarray(coordinate.values), np.asarray(values))

    def update_dataset(self, dataset):
        """Update a compatible trace without recreating axes, controls or view state."""
        if dataset.dimensions != self.dataset.dimensions or dataset.data.ndim != 1:
            raise ValueError("Live trace dimensions changed")
        coordinates = [c for c in dataset.coordinates
                       if c.dimensions == (dataset.dimensions[0],)]
        if [c.name for c in coordinates] != [c.name for c in self.coordinates]:
            raise ValueError("Live trace coordinate names changed")
        self.dataset = dataset
        self.coordinates = coordinates
        self.redraw()

    def release(self):
        """Drop all plot and dataset references when the user closes this tab."""
        self.curve.clear()
        self.plot.clear()
        self.dataset = None
        self.coordinates = []


