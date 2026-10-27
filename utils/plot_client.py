"""One detached live-plot window. Run: python utils/plot_client.py [RUN.auspex ...]."""

import argparse
from pathlib import Path
import queue
import threading

from PySide6 import QtCore, QtWidgets

from auspex_v2.experiment.client import ExperimentServerClient
from auspex_v2.filters.preview_client import poll_run
from src.data_trace import DataTab
from src.data_heatmap import HeatmapTab
from src.plot_client_model import PlotCatalog
from data_browser import dark_theme, add_json

ROLE = QtCore.Qt.ItemDataRole.UserRole


class PlotPane(QtWidgets.QWidget):
    """Keep display controls/view state while replacing only compatible live arrays."""

    def __init__(self, key):
        super().__init__()
        self.key = key
        self.renderer = None
        self.layout = QtWidgets.QVBoxLayout(self)
        self.notice = QtWidgets.QLabel("Waiting for a display snapshot (not saved data).")
        self.layout.addWidget(self.notice)

    def update_frame(self, dataset):
        if self.renderer is None:
            renderer = HeatmapTab if dataset.data.ndim == 2 else DataTab
            self.renderer = renderer(self.key, dataset)
            self.layout.addWidget(self.renderer, 1)
        else:
            self.renderer.update_dataset(dataset)

    def release(self):
        if self.renderer is not None:
            self.renderer.release()


def collect_updates(paths, socket_path, discover, wanted, sequences, stop, output):
    """Network/filesystem work stays off Qt; a bounded daemon has no acquisition role."""
    errors = []
    if discover:
        try:
            client = ExperimentServerClient(socket_path, timeout=0.5)
            paths = set(paths) | {Path(run["path"]) for run in client.list_runs()}
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"ExperimentServer unavailable: {exc}")
    runs = []
    for path in paths:
        if stop.is_set():
            return
        try:
            run = poll_run(path, wanted=wanted.get(str(path), ()), sequences=sequences.get(str(path)))
            if run is not None:
                runs.append(run)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    if not stop.is_set():
        output.put((runs, errors))


class PlotClient(QtWidgets.QMainWindow):
    """A session's newest-first plot catalog, tabbed displays and selected metadata."""

    def __init__(self, paths=(), *, socket_path=None, discover=True, auto_poll=True):
        """Attach without ownership; closing this window never sends run commands."""
        super().__init__()
        self.setWindowTitle("Auspex V2 — Live plots (read only)")
        self.resize(1450, 850)
        self.paths = {Path(p).expanduser().resolve() for p in paths}
        self.socket_path = socket_path
        self.discover = discover
        self.catalog = PlotCatalog()
        self.open_tabs = {}
        self.stop = threading.Event()
        self.replies = queue.Queue(maxsize=1)
        self.worker = None
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Runs / plots — newest first"])
        self.tree.currentItemChanged.connect(self.selected)
        self.tree.itemDoubleClicked.connect(lambda item, _: self.open_plot(item.data(0, ROLE)))
        self.close_button = QtWidgets.QPushButton("Close plot")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.remove_selected)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(self.close_button)
        left_layout.addWidget(self.tree)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self.tab_selected)
        self.metadata = QtWidgets.QTreeWidget()
        self.metadata.setHeaderLabels(["Metadata", "Value"])
        splitter = QtWidgets.QSplitter()
        for widget in (left, self.tabs, self.metadata):
            splitter.addWidget(widget)
        splitter.setSizes([280, 850, 320])
        self.setCentralWidget(splitter)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        if auto_poll:
            self.timer.start(500)
            self.tick()

    def tick(self):
        """Coalesce polling work; never queue GUI updates behind a slow connection."""
        try:
            runs, errors = self.replies.get_nowait()
            self.apply_updates(runs)
            if errors:
                prefix = "64-plot cap: new plots declined. " if self.catalog.rejected else ""
                self.statusBar().showMessage(prefix + "; ".join(errors), 3000)
        except queue.Empty:
            pass
        if self.worker is not None and self.worker.is_alive():
            return
        wanted, sequences = {}, {}
        for key, record in self.catalog.plots.items():
            path = record["run"]["path"]
            if key in self.open_tabs:
                wanted.setdefault(path, []).append(key[1])
            if record.get("frame") is not None:
                sequences.setdefault(path, {})[key[1]] = record["frame"][1]["sequence"]
        self.worker = threading.Thread(target=collect_updates,
            args=(set(self.paths), self.socket_path, self.discover, wanted, sequences, self.stop, self.replies), daemon=True)
        self.worker.start()

    def apply_updates(self, runs):
        """Update terminal state before admissions, preserving focus and explicit removals."""
        by_id = {run["id"]: run for run in runs}
        for key, record in self.catalog.plots.items():
            if key[0] in by_id:
                record["state"] = by_id[key[0]]["state"]
        for run in sorted(runs, key=lambda r: r["order"]):
            self.paths.add(Path(run["path"]))
            for name, spec in run["catalog"]["plots"].items():
                key = (run["id"], name)
                record = {"state": run["state"], "order": run["order"], "run": run,
                          "spec": spec, "frame": None, "error": None}
                admitted, evicted = self.catalog.admit(key, record)
                if evicted is not None:
                    self._close_key(evicted)
                if not admitted:
                    continue
                record = self.catalog.plots[key]
                record.update(state=run["state"], run=run, spec=spec,
                              error=spec.get("error") or run["errors"].get(name))
                if name in run["frames"]:
                    record["frame"] = run["frames"][name]
                if key not in self.open_tabs and record.get("new", True):
                    record["new"] = False
                    self.open_plot(key, focus=False)
                if key in self.open_tabs:
                    self._update_pane(key)
        self.rebuild_tree()
        if self.catalog.rejected:
            self.statusBar().showMessage("64-plot cap: new plots declined while all retained plots are active.")
        else:
            self.statusBar().clearMessage()
        if self.tree.currentItem() is not None:
            self.selected(self.tree.currentItem())
        else:
            self.tab_selected(self.tabs.currentIndex())

    def _update_pane(self, key):
        """Retain the last good display if a new frame is incompatible or disconnected."""
        record, pane = self.catalog.plots[key], self.open_tabs[key]
        if record["frame"] is not None:
            dataset, metadata = record["frame"]
            if getattr(pane, "sequence", None) != metadata["sequence"]:
                try:
                    pane.update_frame(dataset)
                    pane.sequence = metadata["sequence"]
                except (ValueError, KeyError, StopIteration) as exc:
                    record["error"] = str(exc)
        pane.notice.setText(f"{record['state']} • {record['run']['connection']} • "
                            + (record["error"] or "Preview only; run/publication status shown in metadata."))

    def open_plot(self, key, *, focus=True):
        """Explicit opening focuses a plot; automatic admission preserves inspection focus."""
        if key not in self.catalog.plots:
            return
        if key in self.open_tabs:
            if focus:
                self.tabs.setCurrentWidget(self.open_tabs[key])
            return
        pane = PlotPane(key)
        self.open_tabs[key] = pane
        record = self.catalog.plots[key]
        self.tabs.addTab(pane, f"{record['run']['name']} — {key[1]}")
        self._update_pane(key)
        if focus:
            self.tabs.setCurrentWidget(pane)

    def _close_key(self, key):
        pane = self.open_tabs.pop(key, None)
        if pane is not None:
            self.tabs.removeTab(self.tabs.indexOf(pane))
            pane.release()
            pane.deleteLater()

    def close_tab(self, index):
        """Unsubscribe but retain the listing and last frame for explicit reopening."""
        pane = self.tabs.widget(index)
        if pane is not None:
            self._close_key(pane.key)
            self.rebuild_tree()

    def remove_selected(self):
        item = self.tree.currentItem()
        key = item.data(0, ROLE) if item is not None else None
        if key in self.catalog.plots:
            self._close_key(key)
            self.catalog.remove(key)
            self.rebuild_tree()

    def rebuild_tree(self):
        """Keep closed listings and selected metadata while ordering runs newest first."""
        current = self.tree.currentItem()
        selected = current.data(0, ROLE) if current is not None else None
        self.tree.blockSignals(True)
        self.tree.clear()
        roots = {}
        for key, record in sorted(self.catalog.plots.items(), key=lambda p: p[1]["order"], reverse=True):
            run = record["run"]
            if key[0] not in roots:
                root = QtWidgets.QTreeWidgetItem([f"{run['name']} — {Path(run['path']).stem} [{record['state']}]"])
                self.tree.addTopLevelItem(root)
                roots[key[0]] = root
                root.setData(0, ROLE, ("run", key[0]))
                root.setExpanded(True)
            item = QtWidgets.QTreeWidgetItem(roots[key[0]], [key[1] + (" [open]" if key in self.open_tabs else "")])
            item.setData(0, ROLE, key)
            if key == selected:
                self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)
        if self.tree.currentItem() is None:
            self.close_button.setEnabled(False)
            self.metadata.clear()

    def selected(self, item, _previous=None):
        """Inspect run/plot metadata; only an already-open plot changes tab focus."""
        key = item.data(0, ROLE) if item is not None else None
        self.close_button.setEnabled(key in self.catalog.plots)
        if isinstance(key, tuple) and key[0] == "run":
            record = next((r for k, r in self.catalog.plots.items() if k[0] == key[1]), None)
            self.metadata.clear()
            if record is not None:
                add_json(self.metadata.invisibleRootItem(), "Run metadata", record["run"]["metadata"])
                add_json(self.metadata.invisibleRootItem(), "Run status", record["run"]["status"])
            return
        if key not in self.catalog.plots:
            return
        if key in self.open_tabs:
            self.tabs.setCurrentWidget(self.open_tabs[key])
        record = self.catalog.plots[key]
        self.metadata.clear()
        for label, data in (("Run metadata", record["run"]["metadata"]),
                            ("Run status (authoritative when available)", record["run"]["status"]),
                            ("Plot source / schema", record["spec"]),
                            ("Display snapshot", record["frame"][1] if record["frame"] else None),
                            ("Display diagnostic", record["error"])):
            add_json(self.metadata.invisibleRootItem(), label, data)
        self.metadata.expandToDepth(0)

    def tab_selected(self, index):
        """Synchronize tree highlighting and metadata with the inspected tab."""
        pane = self.tabs.widget(index)
        if pane is None:
            return
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, ROLE) == pane.key:
                self.tree.setCurrentItem(item)
                self.selected(item)
                return
            iterator += 1

    def closeEvent(self, event):
        self.stop.set()
        self.timer.stop()
        for key in list(self.open_tabs):
            self._close_key(key)
        event.accept()


def main(argv=None):
    """Attach a viewer without starting services or taking ownership of acquisition."""
    parser = argparse.ArgumentParser(description="Read-only live Auspex plots")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional explicit run folders")
    parser.add_argument("--socket", help="ExperimentServer socket (default: local service)")
    parser.add_argument("--no-discovery", action="store_true", help="Only follow explicit run folders")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dark_theme(app)
    window = PlotClient(args.paths, socket_path=args.socket, discover=not args.no_discovery)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
