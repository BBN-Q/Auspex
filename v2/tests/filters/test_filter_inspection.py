"""Read-only local inspection endpoint and optional dark GUI smoke tests."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest

import numpy as np

from auspex_v2.data import Axis
from auspex_v2.filters import Batch, FilterServer, NodeSpec, PipelineError, PortSchema, ResultSpec
from auspex_v2.filters.inspection import fetch_snapshot

SCHEMA = PortSchema("<c8", 1, ("frequency",), ("Hz",), 256,
                    value_name="S21", value_unit="dimensionless")


def graph(path):
    return FilterServer(SCHEMA, {"avg": NodeSpec("complex_average", SCHEMA, {"repeats": 2})},
                        [("$input", "out", "avg")],
                        {"mean": ResultSpec("avg", "final"),
                         "preview": ResultSpec("avg", "partial", preview=True)},
                        input_name="vna_s21", inspect_socket=path)


class InspectionTest(unittest.TestCase):
    """Verify bounded local snapshots contain acknowledged config, not acquisition data."""

    def test_socket_snapshot_and_lifecycle(self):
        """A viewer can attach independently and reconnect while the graph runs."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter.sock"
            with graph(path) as server:
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                before = fetch_snapshot(path)
                self.assertEqual(before["snapshot_version"], 1)
                self.assertEqual(before["input"]["name"], "vna_s21")
                self.assertEqual(before["nodes"]["avg"]["config"],
                                 {"repeats": 2, "group_name": "average"})
                self.assertEqual(before["nodes"]["avg"]["revision"], 0)
                before["nodes"]["avg"]["config"]["repeats"] = 99
                self.assertEqual(fetch_snapshot(path)["nodes"]["avg"]["config"]["repeats"], 2)
                self.assertEqual(len(before["edges"]), 3)
                self.assertTrue(next(e for e in before["edges"]
                                     if e["target"] == "result:preview")["preview"])
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(path))
                    client.sendall(b'{"version":1,"op":"CONFIGURE"}\n')
                    self.assertFalse(json.loads(client.makefile("rb").readline())["ok"])
                with self.assertRaisesRegex(PipelineError, "between groups"):
                    server.push(Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                                      (Axis("frequency", [1.], "Hz"),), scan="sweep"))
                    server.end_window(0)
                    server.configure("avg", {"repeats": 3})
                self.assertEqual(fetch_snapshot(path)["nodes"]["avg"]["config"]["repeats"], 2)
                server.push(Batch(None, 1, 0, 0, np.array([1j], dtype=np.complex64),
                                  (Axis("frequency", [1.], "Hz"),), scan="sweep"))
                server.end_window(1)
                self.assertEqual(server.configure("avg", {"repeats": 3}), 1)
                updated = fetch_snapshot(path)
                self.assertEqual(updated["nodes"]["avg"]["config"],
                                 {"repeats": 3, "group_name": "average"})
                self.assertEqual(updated["nodes"]["avg"]["revision"], 1)
                self.assertNotIn("data", json.dumps(updated))
                server.cancel()
            self.assertFalse(path.exists())
            with self.assertRaises(OSError):
                fetch_snapshot(path)

    def test_requires_explicit_private_directory_and_rejects_socket_collision(self):
        """Never bind a read-only configuration endpoint in a shared directory."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter.sock"
            with graph(path):
                with self.assertRaises(FileExistsError):
                    graph(path)
            os.chmod(directory, 0o755)
            with self.assertRaisesRegex(ValueError, "private directory"):
                graph(path)


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "install auspex-v2[gui] for GUI tests")
class GraphViewerTest(unittest.TestCase):
    """Exercise the optional viewer with an offscreen Qt application."""

    def test_dark_scene_selection_and_manual_refresh(self):
        """Redraw on topology change but preserve geometry for parameter refreshes."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets, QtGui, QtCore, QtTest
        script = Path(__file__).resolve().parents[2] / "utils" / "filter_graph_viewer.py"
        spec = importlib.util.spec_from_file_location("filter_graph_viewer", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        module.dark_theme(app)
        self.assertEqual(app.palette().color(QtGui.QPalette.ColorRole.Window).name(), "#202632")
        snapshot = {"captured_at": "time-one", "lifecycle": "RUNNING",
                    "input": {"name": "vna_s21", "schema": {"dtype": "<c8"}},
                    "nodes": {"avg": {"kind": "complex_average", "config": {"repeats": 2},
                                       "revision": 0, "schema": {"dtype": "<c8"},
                                       "inputs": ["in"], "outputs": ["partial", "final"]}},
                    "edges": [{"source": "$input", "port": "out", "target": "avg",
                               "target_port": "in", "preview": False},
                              {"source": "avg", "port": "partial", "target": "result:plot",
                               "target_port": "in", "preview": True},
                              {"source": "avg", "port": "final", "target": "writer:raw",
                               "target_port": "in", "preview": False}],
                    "results": {"plot": {"node": "avg", "port": "partial", "preview": True}},
                    "writers": {"raw": {"node": "avg", "port": "final", "group": "data", "prefix": ""}}}
        module.fetch_snapshot = lambda path: snapshot
        window = module.ViewerWindow("/no/socket")
        try:
            deadline = time.monotonic() + 3
            while window.snapshot is None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertIsNotNone(window.snapshot)
            self.assertEqual(len(window.node_items), 4)
            self.assertGreater(window.node_items["writer:raw"].pos().x(), window.node_items["avg"].pos().x())
            original = window.node_items["avg"]
            window.show()
            app.processEvents()
            self.assertLessEqual(window.legend.height(), window.legend.fontMetrics().height() + 8)
            self.assertGreater(window.view.height(), 5 * window.legend.height())
            click = window.view.mapFromScene(original.sceneBoundingRect().center())
            QtTest.QTest.mouseClick(window.view.viewport(), QtCore.Qt.MouseButton.LeftButton,
                                    pos=click)
            self.assertTrue(original.isSelected())
            self.assertIn('"repeats": 2', window.details.toPlainText())
            original.setSelected(False)
            edge = next(item for item in window.scene.items() if isinstance(item, module.EdgeItem))
            edge.setSelected(True)
            self.assertIn('"connection"', window.details.toPlainText())
            edge.setSelected(False)
            original.setSelected(True)
            second = json.loads(json.dumps(snapshot))
            second["nodes"]["avg"]["config"]["repeats"] = 4
            second["captured_at"] = "time-two"
            module.fetch_snapshot = lambda path: second
            window.refresh_button.click()
            deadline = time.monotonic() + 3
            while window.snapshot["captured_at"] != "time-two" and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertIs(window.node_items["avg"], original)
            self.assertIn('"repeats": 4', window.details.toPlainText())
            module.fetch_snapshot = lambda path: (_ for _ in ()).throw(OSError("offline"))
            window.refresh_button.click()
            deadline = time.monotonic() + 3
            while "Disconnected" not in window.status.text() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertIn("Disconnected", window.status.text())
            self.assertIs(window.node_items["avg"], original)
        finally:
            window.close()
