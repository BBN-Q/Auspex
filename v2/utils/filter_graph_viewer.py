"""Read-only, dark-themed viewer for a locally running V2 FilterServer.

Usage: python v2/utils/filter_graph_viewer.py /private/dir/filter.sock
Install the optional GUI dependencies with: pip install -e './v2[gui]'
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from auspex_v2.filters.inspection import fetch_snapshot

BACKGROUND = QtGui.QColor("#171b24")
BOX = QtGui.QColor("#293241")
SINK = QtGui.QColor("#243d41")
PREVIEW = QtGui.QColor("#363246")
INK = QtGui.QColor("#edf1f7")
MUTED = QtGui.QColor("#b5c3d5")
EDGE = QtGui.QColor("#a5bbd2")
ACCENT = QtGui.QColor("#77d7ff")


class NodeItem(QtWidgets.QGraphicsObject):
    """Selectable filter, acquisition, or sink box with labeled ports."""

    WIDTH = 190

    def __init__(self, key, title, subtitle, kind, inputs, outputs):
        super().__init__()
        self.key = key
        self.kind = kind
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.height = 67 + 20 * max(len(self.inputs), len(self.outputs), 1)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.labels = []
        metrics = QtGui.QFontMetricsF(QtGui.QFont())
        for text, x, y in [(title, 12, 8), (subtitle, 12, 30)]:
            label = QtWidgets.QGraphicsSimpleTextItem(
                metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, self.WIDTH - 24), self)
            label.setToolTip(text)
            label.setBrush(INK if y == 8 else MUTED)
            label.setPos(x, y)
            self.labels.append(label)
        for side, ports in (("in", self.inputs), ("out", self.outputs)):
            for index, port in enumerate(ports):
                label = QtWidgets.QGraphicsSimpleTextItem(port, self)
                label.setBrush(MUTED)
                y = 53 + index * 20
                label.setPos(13 if side == "in" else self.WIDTH - 13 - label.boundingRect().width(), y)
                self.labels.append(label)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.WIDTH, self.height)

    def port_pos(self, direction, port):
        """Return a scene-coordinate anchor for an explicitly named port."""
        ports = self.inputs if direction == "in" else self.outputs
        index = ports.index(port)
        x = 0 if direction == "in" else self.WIDTH
        return self.mapToScene(QtCore.QPointF(x, 64 + index * 20))

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        fill = PREVIEW if self.kind == "preview" else SINK if self.kind in ("result", "writer") else BOX
        painter.setBrush(fill)
        painter.setPen(QtGui.QPen(ACCENT if self.isSelected() else QtGui.QColor("#526378"),
                                  2.5 if self.isSelected() else 1))
        painter.drawRoundedRect(self.boundingRect(), 8, 8)
        painter.setBrush(ACCENT if self.isSelected() else EDGE)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for direction, ports in (("in", self.inputs), ("out", self.outputs)):
            for index, _ in enumerate(ports):
                x = 0 if direction == "in" else self.WIDTH
                painter.drawEllipse(QtCore.QPointF(x, 64 + index * 20), 3, 3)


class EdgeItem(QtWidgets.QGraphicsPathItem):
    """Selectable curved connection with a generous invisible click target."""

    def __init__(self, source, destination, preview):
        start = source
        end = destination
        path = QtGui.QPainterPath(start)
        bend = min(85, max(30, (end.x() - start.x()) * 0.45))
        path.cubicTo(start.x() + bend, start.y(), end.x() - bend, end.y(), end.x(), end.y())
        super().__init__(path)
        self.preview = preview
        pen = QtGui.QPen(MUTED if preview else EDGE, 2)
        if preview:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setZValue(0)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        tangent = path.pointAtPercent(0.99)
        angle = math.atan2(end.y() - tangent.y(), end.x() - tangent.x())
        tip = QtCore.QPointF(end.x() - 5 * math.cos(angle), end.y() - 5 * math.sin(angle))
        arrow = QtGui.QPolygonF([tip,
            QtCore.QPointF(tip.x() - 11 * math.cos(angle - 0.45), tip.y() - 11 * math.sin(angle - 0.45)),
            QtCore.QPointF(tip.x() - 11 * math.cos(angle + 0.45), tip.y() - 11 * math.sin(angle + 0.45))])
        marker = QtWidgets.QGraphicsPolygonItem(arrow, self)
        marker.setBrush(MUTED if preview else EDGE)
        marker.setPen(QtCore.Qt.PenStyle.NoPen)
        self.key = None

    def shape(self):
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self.path())

    def paint(self, painter, option, widget=None):
        pen = QtGui.QPen(ACCENT if self.isSelected() else MUTED if self.preview else EDGE,
                         3 if self.isSelected() else 2)
        if self.preview:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.drawPath(self.path())


class GraphView(QtWidgets.QGraphicsView):
    """Zoomable diagram with middle-button panning that preserves left-click selection."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(BACKGROUND)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pan_start = None

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if 0.05 < self.transform().m11() * factor < 3.5:
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.pan_start = event.position()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan_start is not None:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.pan_start = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class SnapshotWorker(QtCore.QObject):
    """Perform socket I/O outside the GUI thread without polling automatically."""

    finished = QtCore.Signal(object, object)

    def __init__(self, socket_path):
        super().__init__()
        self.socket_path = socket_path

    @QtCore.Slot()
    def fetch(self):
        try:
            self.finished.emit(fetch_snapshot(self.socket_path), None)
        except Exception as exc:
            self.finished.emit(None, str(exc))


def layout_positions(snapshot):
    """Place a DAG in stable topological columns with upstream-ordered rows."""
    ids = ["$input", *snapshot["nodes"],
           *(f"result:{key}" for key in snapshot["results"]),
           *(f"writer:{key}" for key in snapshot["writers"])]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate graph node IDs")
    parents = {key: [] for key in ids}
    for edge in snapshot["edges"]:
        if edge["source"] not in parents or edge["target"] not in parents:
            raise ValueError("Unknown graph node")
        parents[edge["target"]].append(edge["source"])
    depth = {"$input": 0}
    while len(depth) < len(ids):
        available = [key for key in ids if key not in depth and parents[key] and
                     all(parent in depth for parent in parents[key])]
        if not available:
            raise ValueError("Cycle or unreachable graph node")
        for key in available:
            depth[key] = 1 + max(depth[parent] for parent in parents[key])
    columns = {}
    for key, level in depth.items():
        columns.setdefault(level, []).append(key)
    positions = {}
    for level in sorted(columns):
        def ordering(key):
            return (sum(positions[parent][1] for parent in parents[key]) / len(parents[key])
                    if parents[key] else 0, key)
        for row, key in enumerate(sorted(columns[level], key=ordering)):
            positions[key] = (level * 300, row * 155)
    return positions


class ViewerWindow(QtWidgets.QMainWindow):
    """Display one read-only graph snapshot and manually refresh it on request."""

    requested = QtCore.Signal()

    def __init__(self, socket_path):
        super().__init__()
        self.socket_path = socket_path
        self.snapshot = None
        self.topology = None
        self.node_items = {}
        self.setWindowTitle("Auspex V2 — Filter Graph (read only)")
        self.resize(1180, 720)
        self.status = QtWidgets.QLabel("Not connected — no snapshot")
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.fit_button = QtWidgets.QPushButton("Fit graph")
        self.fit_button.clicked.connect(self.fit_graph)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.status, 1)
        bar.addWidget(self.fit_button)
        bar.addWidget(self.refresh_button)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.selectionChanged.connect(self.show_details)
        self.view = GraphView(self.scene)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlainText("Click a node or connection to inspect its snapshot.")
        split = QtWidgets.QSplitter()
        split.addWidget(self.view)
        split.addWidget(self.details)
        split.setSizes([830, 350])
        self.legend = QtWidgets.QLabel(
            "Solid: required    - - -: lossy preview    •    Click to inspect; wheel to zoom; middle-drag to pan")
        self.legend.setWordWrap(False)
        self.legend.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Fixed)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(bar)
        layout.addWidget(split, 1)
        layout.addWidget(self.legend)
        root = QtWidgets.QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.thread = QtCore.QThread(self)
        self.worker = SnapshotWorker(socket_path)
        self.worker.moveToThread(self.thread)
        self.requested.connect(self.worker.fetch)
        self.worker.finished.connect(self.received)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        QtCore.QTimer.singleShot(0, self.refresh)

    def refresh(self):
        """Ask once for an immutable snapshot; never edit filter state."""
        if not self.refresh_button.isEnabled():
            return
        self.refresh_button.setEnabled(False)
        self.status.setText(f"Fetching snapshot from {self.socket_path} …")
        self.requested.emit()

    @QtCore.Slot(object, object)
    def received(self, snapshot, error):
        self.refresh_button.setEnabled(True)
        if error is not None:
            self.status.setText(f"Disconnected — last snapshot retained. {error}")
            return
        try:
            self.show_snapshot(snapshot)
        except (KeyError, ValueError, TypeError) as exc:
            self.status.setText(f"Invalid snapshot — last view retained. {exc}")

    def show_snapshot(self, snapshot):
        """Retain geometry on parameter-only refreshes, redraw only changed topology."""
        positions = layout_positions(snapshot)
        topology = (tuple(sorted((name, spec["kind"]) for name, spec in snapshot["nodes"].items())),
                    tuple(sorted((name, spec["preview"]) for name, spec in snapshot["results"].items())),
                    tuple(sorted((name, spec["group"]) for name, spec in snapshot["writers"].items())),
                    tuple(sorted((e["source"], e["port"], e["target"], e["target_port"], e["preview"])
                                 for e in snapshot["edges"])))
        if topology != self.topology:
            self.draw_graph(snapshot, positions)
            self.topology = topology
        self.snapshot = snapshot
        self.status.setText(f"Snapshot {snapshot['captured_at']}  •  {snapshot['lifecycle']}  •  "
                            f"{self.socket_path}  •  Read only (not live)")
        self.show_details()

    def draw_graph(self, snapshot, positions):
        """Draw boxed ports and directed connections for a validated snapshot DAG."""
        self.scene.clear()
        self.node_items.clear()
        defs = {"$input": (snapshot["input"]["name"], "Acquisition", "input", [], ["out"])}
        for name, spec in snapshot["nodes"].items():
            defs[name] = (name, spec["kind"], "filter", spec["inputs"], spec["outputs"])
        for name, spec in snapshot["results"].items():
            kind = "preview" if spec["preview"] else "result"
            defs[f"result:{name}"] = (name, kind.title(), kind, ["in"], [])
        for name, spec in snapshot["writers"].items():
            defs[f"writer:{name}"] = (name, f"Writer → {spec['group']}", "writer", ["in"], [])
        for key, (title, subtitle, kind, inputs, outputs) in defs.items():
            item = NodeItem(key, title, subtitle, kind, inputs, outputs)
            item.setPos(*positions[key])
            self.scene.addItem(item)
            self.node_items[key] = item
        for edge in snapshot["edges"]:
            source = self.node_items[edge["source"]].port_pos("out", edge["port"])
            target = self.node_items[edge["target"]].port_pos("in", edge["target_port"])
            item = EdgeItem(source, target, edge["preview"])
            item.key = edge
            self.scene.addItem(item)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-32, -32, 32, 32))
        self.fit_graph()

    def fit_graph(self):
        if self.scene.items():
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def show_details(self):
        """Show configured values and schemas, without displaying measurement arrays."""
        if self.snapshot is None:
            return
        selected = self.scene.selectedItems()
        if not selected:
            self.details.setPlainText("Select a box or connection. This is a static snapshot, not a live view.")
            return
        item = selected[0]
        if isinstance(item, EdgeItem):
            edge = item.key
            source = edge["source"]
            schema = (self.snapshot["input"]["schema"] if source == "$input"
                      else self.snapshot["nodes"][source]["schema"])
            info = {"connection": edge, "schema": schema}
        elif isinstance(item, NodeItem):
            key = item.key
            info = (self.snapshot["input"] if key == "$input" else
                    self.snapshot["results"][key[7:]] if key.startswith("result:") else
                    self.snapshot["writers"][key[7:]] if key.startswith("writer:") else
                    self.snapshot["nodes"][key])
            info = {"name": key, **info}
        else:
            return
        self.details.setPlainText(json.dumps(info, indent=2, ensure_ascii=False))

    def closeEvent(self, event):
        self.thread.quit()
        self.thread.wait(4000)  # requests have a three-second socket timeout
        super().closeEvent(event)


def dark_theme(app):
    """Apply a dark, high-contrast palette to this standalone process."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    for role, color in ((QtGui.QPalette.ColorRole.Window, "#202632"),
                        (QtGui.QPalette.ColorRole.WindowText, "#edf1f7"),
                        (QtGui.QPalette.ColorRole.Base, "#171b24"),
                        (QtGui.QPalette.ColorRole.AlternateBase, "#293241"),
                        (QtGui.QPalette.ColorRole.Text, "#edf1f7"),
                        (QtGui.QPalette.ColorRole.Button, "#344052"),
                        (QtGui.QPalette.ColorRole.ButtonText, "#edf1f7"),
                        (QtGui.QPalette.ColorRole.Highlight, "#337ba3"),
                        (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff")):
        palette.setColor(role, QtGui.QColor(color))
    app.setPalette(palette)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a running Auspex V2 filter graph")
    parser.add_argument("socket", help="FilterServer's explicit inspect_socket path")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    dark_theme(app)
    window = ViewerWindow(args.socket)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
