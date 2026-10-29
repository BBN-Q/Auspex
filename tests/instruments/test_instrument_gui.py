"""Optional offscreen smoke test of the inventory-only operator window."""

import importlib.util
import os
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "install auspex-v2[gui] to test Qt")
class InstrumentGuiTest(unittest.TestCase):
    """Verify GUI actions do not open instrument sessions or access settings."""

    def test_connection_actions_and_clear_gating(self):
        """Show actions for each state and gate clearing until fully disconnected."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from pyqtgraph.Qt import QtGui, QtWidgets
        from auspex_v2.instruments.server import gui

        class FakeClient:
            """Track GUI requests with one mutable simulated device."""

            calls = []
            state = "DISCONNECTED"
            present = True
            status_error = False
            live_owner = None
            run = None
            released = []

            def __init__(self, socket_path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def request(self, op, **kwargs):
                self.calls.append(op)
                if op == "RELEASE_LIVE":
                    self.released.append(kwargs)
                    type(self).live_owner = None
                elif op == "CONNECT":
                    type(self).state = "CONNECTED"
                elif op == "DISCONNECT" or op == "DISCONNECT_ALL":
                    type(self).state = "DISCONNECTED"
                elif op in ("TERMINATE", "TERMINATE_ALL"):
                    type(self).state = "TERMINATED"
                elif op == "RECOVER":
                    type(self).state = "DISCONNECTED"
                elif op in ("CLEAR_INSTRUMENTS", "DEREGISTER"):
                    type(self).present = False
                if op == "STATUS":
                    if self.status_error:
                        raise ConnectionError("offline")
                    return {"run": self.run, "instruments": {"src": {
                        "driver": "SimulatedSource", "physical": "SIM::SRC",
                        "state": self.state, "worker_running": self.state == "CONNECTED",
                        "live_sessions": int(self.live_owner is not None),
                        "live_owner": self.live_owner, "locked": False}} if self.present else {}}
                return None

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        gui.dark_theme(app)
        self.assertEqual(app.palette().color(QtGui.QPalette.ColorRole.Window).name(), "#202632")
        self.assertEqual(app.palette().color(QtGui.QPalette.ColorRole.Base).name(), "#171b24")

        def until(predicate):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                app.processEvents()
                if predicate():
                    return
                time.sleep(0.01)
            self.fail("GUI request did not finish")

        def button(window, label):
            panel = window.table.cellWidget(0, 5)
            buttons = panel.findChildren(QtWidgets.QPushButton)
            return next(item for item in buttons if item.text() == label)

        with patch.object(gui, "InstrumentServerClient", FakeClient), \
             patch.object(gui.QtWidgets.QMessageBox, "question",
                          return_value=QtWidgets.QMessageBox.StandardButton.Yes) as confirm:
            window = gui.InstrumentWindow()
            try:
                self.assertEqual(window.windowTitle(), "Auspex V2 — Instrument Server")
                self.assertTrue(window.table.alternatingRowColors())
                self.assertIn("#ff8c95", window.terminate_all.styleSheet())
                until(lambda: window.status is not None and not window.pending)
                self.assertTrue(window.clear_instruments.isEnabled())
                FakeClient.run = {"run_id": "1abc4de" + "0" * 25, "name": "vna_sweep_003", "orphaned": True}
                window._request("STATUS")
                until(lambda: not window.pending)
                self.assertIn("vna_sweep_003 [1abc4de]", window.domain.text())
                self.assertNotIn(FakeClient.run["run_id"], window.domain.text())
                FakeClient.run = None
                window._request("STATUS")
                until(lambda: not window.pending)
                button(window, "Connect").click()
                until(lambda: FakeClient.state == "CONNECTED" and not window.pending)
                self.assertIn("Last action: CONNECT src", window.message.text())
                window._request("STATUS")
                until(lambda: not window.pending)
                self.assertIn("Last action: CONNECT src", window.message.text())
                FakeClient.live_owner = {"id": "abc1234", "label": "notebook", "revoking": False}
                window._request("STATUS")
                until(lambda: not window.pending)
                self.assertIn("notebook [abc1234]", window.table.item(0, 4).text())
                self.assertFalse(button(window, "Disconnect").isEnabled())
                confirm.return_value = QtWidgets.QMessageBox.StandardButton.No
                button(window, "Release live ownership…").click()
                self.assertEqual(FakeClient.released, [])
                self.assertIn("notebook [abc1234]", confirm.call_args.args[2])
                confirm.return_value = QtWidgets.QMessageBox.StandardButton.Yes
                button(window, "Release live ownership…").click()
                until(lambda: not window.pending)
                self.assertEqual(FakeClient.released, [{"name": "src", "session_id": "abc1234"}])
                self.assertEqual(FakeClient.state, "CONNECTED")
                self.assertTrue(button(window, "Disconnect").isEnabled())
                FakeClient.status_error = True
                window._request("STATUS")
                until(lambda: not window.pending)
                self.assertIn("status failed: offline", window.connection.text())
                self.assertIn("Last action: RELEASE_LIVE src", window.message.text())
                FakeClient.status_error = False
                window._request("STATUS")
                until(lambda: not window.pending)
                self.assertFalse(window.clear_instruments.isEnabled())
                self.assertFalse(button(window, "Clear").isEnabled())
                button(window, "Disconnect").click()
                until(lambda: FakeClient.state == "DISCONNECTED" and not window.pending)
                button(window, "Terminate").click()  # disconnected devices can be quarantined
                until(lambda: FakeClient.state == "TERMINATED" and not window.pending)
                self.assertFalse(window.clear_instruments.isEnabled())
                self.assertFalse(button(window, "Disconnect").isEnabled())
                button(window, "Recover").click()
                until(lambda: FakeClient.state == "DISCONNECTED" and not window.pending)
                self.assertEqual(confirm.call_count, 2)  # only the two live-release prompts
                button(window, "Clear").click()
                until(lambda: not FakeClient.present and not window.pending)
                self.assertEqual(window.table.rowCount(), 0)
                self.assertEqual(confirm.call_count, 3)
                self.assertEqual(set(FakeClient.calls), {"STATUS", "CONNECT", "DISCONNECT",
                                                        "TERMINATE", "RECOVER", "DEREGISTER", "RELEASE_LIVE"})
            finally:
                window.close()
