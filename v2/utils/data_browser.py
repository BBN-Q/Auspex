"""Read-only V2 .auspex browser: 1D pyqtgraph plots and nested JSON metadata.

Run with: python v2/utils/data_browser.py [OS_FOLDER | RUN.auspex ...]
Requires the optional auspex-v2[gui] dependencies. No instrument or server access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from auspex_v2.data.browser import inspect_folder
from auspex_v2.data import AuspexStore

BACKGROUND = "#171b24"
INK = "#e7eef8"
MUTED = QtGui.QColor("#8996a9")
ROLE_KEY = QtCore.Qt.ItemDataRole.UserRole


class DataTab(QtWidgets.QWidget):
    """Hold one validated 1D dataset and plot against its actual coordinates."""

    def __init__(self, key, dataset):
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

    def release(self):
        """Drop all plot and dataset references when the user closes this tab."""
        self.curve.clear()
        self.plot.clear()
        self.dataset = None
        self.coordinates = []


class DataBrowser(QtWidgets.QMainWindow):
    """Browse several published stores without changing their files or data."""

    def __init__(self, paths=()):
        super().__init__()
        self.setWindowTitle("Auspex V2 — DataBrowser (read only)")
        self.resize(1390, 790)
        self.folders = {}
        self.folder_errors = {}
        self.watched_directories = {}  # OS folder -> direct .auspex children
        self.watch_errors = {}
        self.direct_runs = {}  # Individually added .auspex paths
        self.open_tabs = {}
        self.tree_items = {}
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabel("Available datasets")
        self.tree.itemDoubleClicked.connect(self._open_item)
        self.tree.currentItemChanged.connect(self._selected)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.metadata = QtWidgets.QTreeWidget()
        self.metadata.setHeaderLabels([".auspex metadata", "Value"])
        self.metadata.setAlternatingRowColors(True)
        self.metadata.setColumnWidth(0, 190)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.add_folder_button = QtWidgets.QPushButton("Add folder…")
        self.add_folder_button.clicked.connect(self._choose_directory)
        left_controls = QtWidgets.QHBoxLayout()
        left_controls.addWidget(self.add_folder_button)
        left_controls.addWidget(self.refresh_button)
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addLayout(left_controls)
        left_layout.addWidget(self.tree, 1)
        left = QtWidgets.QWidget()
        left.setLayout(left_layout)
        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.addWidget(self.metadata)
        split.setSizes([310, 740, 350])
        self.setCentralWidget(split)
        self._menu()
        self.statusBar().showMessage("Add an OS folder or a single .auspex run; double-click a 1D dataset to plot it.")
        for path in paths:
            try:
                self.open_folder(path)
            except (ValueError, OSError, KeyError) as exc:
                self.statusBar().showMessage(f"Could not open {path}: {exc}")

    def _menu(self):
        """Offer explicit folder selection, snapshot refresh, and browser-only removal."""
        menu = self.menuBar().addMenu("&File")
        add_directory = menu.addAction("&Add folder containing .auspex runs…")
        add_directory.setShortcut("Ctrl+O")
        add_directory.triggered.connect(self._choose_directory)
        add_run = menu.addAction("Add single .&auspex run…")
        add_run.triggered.connect(self._choose_run)
        refresh = menu.addAction("&Refresh all folders and datasets")
        refresh.setShortcut("Ctrl+R")
        refresh.triggered.connect(self.refresh)
        menu.addSeparator()
        close_folder = menu.addAction("&Remove selected folder/run")
        close_folder.triggered.connect(self._remove_selected_folder)
        menu.addSeparator()
        menu.addAction("&Quit", self.close)

    def _choose_directory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add a folder containing .auspex runs")
        if path:
            try:
                self.add_directory(path)
            except (ValueError, OSError, KeyError) as exc:
                QtWidgets.QMessageBox.warning(self, "Cannot add folder", str(exc))

    def _choose_run(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add a single .auspex run")
        if path:
            try:
                self.add_run(path)
            except (ValueError, OSError, KeyError) as exc:
                QtWidgets.QMessageBox.warning(self, "Cannot add run", str(exc))

    def open_folder(self, path):
        """Accept either a folder containing runs or one direct .auspex run."""
        path = Path(path).expanduser().resolve()
        if path.suffix == ".auspex":
            self.add_run(path)
        else:
            self.add_directory(path)

    def add_directory(self, path):
        """Watch only direct .auspex child directories of one OS folder."""
        path = Path(path).expanduser().resolve()
        if not path.is_dir() or path.suffix == ".auspex":
            raise ValueError(f"Select a folder containing .auspex runs: {path}")
        self.watched_directories.setdefault(path, ())
        self.refresh(select=("directory", str(path), None, None))

    def add_run(self, path):
        """Keep a single explicitly selected .auspex run available independently."""
        path = Path(path).expanduser().resolve()
        if path.suffix != ".auspex":
            raise ValueError(f"Select a .auspex run: {path}")
        inspect_folder(path)  # Direct selection should fail clearly if it is not a V2 store.
        self.direct_runs.setdefault(path, None)
        self.refresh(select=("folder", str(path), None, None))

    def refresh(self, _checked=False, *, select=None):
        """Rescan every OS folder and run; preserve tabs unless a run disappears."""
        selected = select if select is not None else self._key(self.tree.currentItem())
        failures = []
        active = set(self.direct_runs)
        for directory in self.watched_directories:
            try:
                children = tuple(sorted((child.resolve() for child in directory.iterdir()
                                         if child.suffix == ".auspex" and child.is_dir()), key=str))
                self.watched_directories[directory] = children
                self.watch_errors.pop(directory, None)
                active.update(children)
            except OSError as exc:
                self.watched_directories[directory] = ()
                self.watch_errors[directory] = str(exc)
                failures.append(f"{directory}: {exc}")
        previous = set(self.folders) | set(self.folder_errors)
        for path in previous - active:
            self._close_run_tabs(path)
        folders = {}
        errors = {}
        for path in sorted(active):
            try:
                folders[path] = inspect_folder(path)
            except (ValueError, OSError, KeyError) as exc:
                errors[path] = str(exc)
                failures.append(f"{path}: {exc}")
        self.folders = folders
        self.folder_errors = errors
        self._rebuild_tree(select=selected)
        self.statusBar().showMessage("; ".join(failures) if failures else
                                     "All folders and datasets refreshed (open plots unchanged).")

    def _rebuild_tree(self, *, select=None):
        """Display watched OS folders, their runs, and separately opened runs."""
        self.tree.blockSignals(True)
        self.tree_items.clear()
        self.tree.clear()
        target = None

        def add_run(parent, path):
            nonlocal target
            root = self._item(parent, path.name, ("folder", str(path), None, None))
            root.setToolTip(0, str(path))
            contents = self.folders.get(path)
            if contents is None:
                root.setText(0, path.name + "  [unreadable]")
                root.setForeground(0, MUTED)
                root.setToolTip(0, self.folder_errors.get(path, "Could not read run"))
            else:
                groups = {}
                for entry in contents.entries:
                    group = groups.get(entry.group)
                    if group is None:
                        group = self._item(root, entry.group, ("group", str(path), entry.group, None))
                        groups[entry.group] = group
                    key = (path, entry.group, entry.name)
                    suffix = "  [open]" if key in self.open_tabs else ""
                    if not entry.plottable:
                        suffix += "  [plotting not yet supported]" if not entry.error else "  [unreadable]"
                    item = self._item(group, entry.name + suffix,
                                      ("dataset", str(path), entry.group, entry.name))
                    if not entry.plottable:
                        item.setForeground(0, MUTED)
                        item.setToolTip(0, entry.reason)
                    if select == self._key(item):
                        target = item
                for group in groups.values():
                    group.setExpanded(True)
            root.setExpanded(True)
            if select == self._key(root):
                target = root

        for directory, children in self.watched_directories.items():
            parent = self._item(None, f"{directory.name}  [OS folder]",
                                ("directory", str(directory), None, None))
            parent.setToolTip(0, str(directory))
            if directory in self.watch_errors:
                parent.setForeground(0, MUTED)
                parent.setToolTip(0, self.watch_errors[directory])
            for path in children:
                add_run(parent, path)
            parent.setExpanded(True)
            if select == self._key(parent):
                target = parent
        watched_paths = {run for children in self.watched_directories.values() for run in children}
        for path in self.direct_runs:
            if path not in watched_paths:
                add_run(None, path)
        self.tree.blockSignals(False)
        if target is None and self.tabs.currentWidget() is not None:
            tab_key = self.tabs.currentWidget().key
            target = self.tree_items.get(("dataset", str(tab_key[0]), tab_key[1], tab_key[2]))
        if target is not None:
            self.tree.setCurrentItem(target)
        elif self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._selected(self.tree.currentItem())

    def _item(self, parent, label, key):
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(0, ROLE_KEY, key)
        self.tree_items[key] = item
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        return item

    @staticmethod
    def _key(item):
        return tuple(item.data(0, ROLE_KEY)) if item else None

    def _entry(self, key):
        return next(entry for entry in self.folders[Path(key[1])].entries
                    if (entry.group, entry.name) == (key[2], key[3]))

    def _open_item(self, item, _column):
        """Open a single validated 1D dataset once; unsupported datasets stay listed."""
        key = self._key(item)
        if key is None or key[0] != "dataset":
            return
        entry = self._entry(key)
        if not entry.plottable:
            self.statusBar().showMessage(entry.reason, 6000)
            return
        tab_key = (Path(key[1]), key[2], key[3])
        if tab_key in self.open_tabs:
            self.tabs.setCurrentWidget(self.open_tabs[tab_key])
            self.tree.setCurrentItem(item)
            return
        try:
            with AuspexStore(tab_key[0]) as store:
                dataset = store.read_dataset(tab_key[1], tab_key[2])
            if dataset.data.ndim != 1 or dataset.offsets is not None:
                raise ValueError("Plotting not yet supported for this dataset")
            tab = DataTab(tab_key, dataset)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.statusBar().showMessage(f"Could not open {entry.name}: {exc}", 10000)
            return
        self.open_tabs[tab_key] = tab
        index = self.tabs.addTab(tab, f"{tab_key[0].name} / {tab_key[1]}/{tab_key[2]}")
        self.tabs.setTabToolTip(index, f"{tab_key[0]}/{tab_key[1]}/{tab_key[2]}")
        self.tabs.setCurrentIndex(index)
        self.tree.setCurrentItem(item)
        item.setText(0, entry.name + "  [open]")
        self._selected(item)

    def close_tab(self, index, *, rebuild=True):
        """Remove a plot and release its loaded numerical arrays, not its folder."""
        tab = self.tabs.widget(index)
        if tab is None:
            return
        key = tab.key
        self.tabs.removeTab(index)
        self.open_tabs.pop(key, None)
        tab.release()
        tab.deleteLater()
        if rebuild:
            current = self.tabs.currentWidget()
            selected = (("dataset", str(current.key[0]), current.key[1], current.key[2])
                        if current is not None else ("dataset", str(key[0]), key[1], key[2]))
            self._rebuild_tree(select=selected)

    def _close_run_tabs(self, path):
        """Close a run's tabs without rebuilding the inventory during bulk changes."""
        for index in reversed(range(self.tabs.count())):
            if self.tabs.widget(index).key[0] == path:
                self.close_tab(index, rebuild=False)

    def close_all_datasets(self, path):
        """Close only this run's tabs; leave its dataset list available."""
        path = Path(path).expanduser().resolve()
        self._close_run_tabs(path)
        current = self.tabs.currentWidget()
        selected = (("dataset", str(current.key[0]), current.key[1], current.key[2])
                    if current is not None else ("folder", str(path), None, None))
        self._rebuild_tree(select=selected)

    def remove_directory(self, path):
        """Unwatch an OS folder and close its tabs without deleting anything."""
        path = Path(path).expanduser().resolve()
        for run in self.watched_directories.pop(path, ()):
            self._close_run_tabs(run)
        self.watch_errors.pop(path, None)
        self.refresh(select=None)

    def remove_run(self, path):
        """Remove a directly added run and close its tabs without deleting files."""
        path = Path(path).expanduser().resolve()
        self._close_run_tabs(path)
        self.direct_runs.pop(path, None)
        self.refresh(select=None)

    def _remove_selected_folder(self):
        """Never interpret a watched run selection as removing its entire OS folder."""
        item = self.tree.currentItem()
        while item and self._key(item)[0] not in ("directory", "folder"):
            item = item.parent()
        if item is None:
            return
        kind, path, _, _ = self._key(item)
        if kind == "directory":
            self.remove_directory(path)
        elif Path(path) in self.direct_runs:
            self.remove_run(path)
        else:
            self.statusBar().showMessage("Select the OS folder to remove it; watched runs cannot be removed individually.",
                                         7000)

    def _folder_menu(self, path):
        """Offer folder-scoped tab closure without removing the run from the tree."""
        menu = QtWidgets.QMenu(self)
        menu.addAction("Close all datasets", lambda: self.close_all_datasets(path))
        return menu

    def _directory_menu(self, path):
        """Expose unwatching separately from folder-scoped dataset tab closure."""
        menu = QtWidgets.QMenu(self)
        menu.addAction("Remove folder", lambda: self.remove_directory(path))
        return menu

    def _context_menu(self, point):
        item = self.tree.itemAt(point)
        if item is None:
            return
        kind, raw_path, _, _ = self._key(item)
        if kind == "directory":
            menu = self._directory_menu(Path(raw_path))
        elif kind == "folder":
            menu = self._folder_menu(Path(raw_path))
        else:
            return
        menu.exec(self.tree.viewport().mapToGlobal(point))

    def _tab_changed(self, index):
        """Selecting a tab highlights its dataset even across watched OS folders."""
        tab = self.tabs.widget(index)
        if tab is None:
            return
        key = ("dataset", str(tab.key[0]), tab.key[1], tab.key[2])
        item = self.tree_items.get(key)
        if item is not None and self.tree.currentItem() is not item:
            self.tree.setCurrentItem(item)

    def _selected(self, item, _previous=None):
        """Show metadata and focus an already open tab for selected datasets."""
        self.metadata.clear()
        key = self._key(item)
        if key is None:
            return
        if key[0] == "dataset":
            tab = self.open_tabs.get((Path(key[1]), key[2], key[3]))
            if tab is not None and self.tabs.currentWidget() is not tab:
                self.tabs.setCurrentWidget(tab)
        path = Path(key[1])
        if key[0] == "directory":
            record = {"OS folder": {"path": str(path),
                                    "auspex runs": [str(p) for p in self.watched_directories[path]],
                                    "error": self.watch_errors.get(path)}}
        elif path not in self.folders:
            record = {"run": {"path": str(path),
                              "error": self.folder_errors.get(path, "Run unavailable")}}
        else:
            contents = self.folders[path]
            record = {"run metadata": contents.metadata if contents.metadata_error is None
                      else {"error": contents.metadata_error},
                      "manifest": contents.manifest}
            if key[0] == "dataset":
                entry = self._entry(key)
                record["dataset header"] = entry.header if entry.error is None else {"error": entry.error}
        for name, value in record.items():
            self._metadata_item(self.metadata.invisibleRootItem(), name, value)
        self.metadata.expandToDepth(0)

    def _metadata_item(self, parent, name, value):
        """Show nested JSON as expandable nodes, not one hard-to-read text blob."""
        item = QtWidgets.QTreeWidgetItem(parent, [str(name)])
        if isinstance(value, dict):
            for key, child in value.items():
                self._metadata_item(item, key, child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._metadata_item(item, f"[{index}]", child)
        else:
            item.setText(1, json.dumps(value, ensure_ascii=False))
        return item


def dark_theme(app):
    """Apply a standalone dark theme without affecting other Auspex programs."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    for role, color in ((QtGui.QPalette.ColorRole.Window, "#202632"),
                        (QtGui.QPalette.ColorRole.WindowText, INK),
                        (QtGui.QPalette.ColorRole.Base, BACKGROUND),
                        (QtGui.QPalette.ColorRole.AlternateBase, "#293241"),
                        (QtGui.QPalette.ColorRole.Text, INK),
                        (QtGui.QPalette.ColorRole.Button, "#344052"),
                        (QtGui.QPalette.ColorRole.ButtonText, INK),
                        (QtGui.QPalette.ColorRole.Highlight, "#337ba3"),
                        (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff")):
        palette.setColor(role, QtGui.QColor(color))
    app.setPalette(palette)
    pg.setConfigOptions(antialias=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Browse and plot 1D V2 .auspex datasets")
    parser.add_argument("folders", nargs="*", help="OS folders to watch or individual V2 .auspex runs")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    dark_theme(app)
    window = DataBrowser(args.folders)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
