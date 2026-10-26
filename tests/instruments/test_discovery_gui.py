"""Offscreen GUI discovery against real local IPC and a fake VISA backend."""

import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from .test_discovery import VisaManager, VNA, SERIAL, UNKNOWN


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "install auspex-v2[gui] to test Qt")
class DiscoveryGuiTest(unittest.TestCase):
    """Exercise real discovery widgets without constructing or connecting hardware drivers."""

    def setUp(self):
        """Keep the Qt client, instrument service and fake VISA resources in one test scope."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from pyqtgraph.Qt import QtWidgets
        from auspex_v2.instruments.server import gui, discovery
        from auspex_v2.instruments.server.service import Server

        self.gui = gui
        self.qt = QtWidgets
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        gui.dark_theme(self.app)
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.home = patch.dict(os.environ, {"AUSPEX_HOME": self.tmp.name})
        self.home.start()
        self.manager = VisaManager()
        self.backend = patch.object(discovery, "resource_manager", return_value=self.manager)
        self.backend.start()
        self.server = Server(Path(self.tmp.name) / "instrument.sock", quiet=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.window = gui.InstrumentWindow(self.server.path)
        self.window.show()
        self.until(lambda: self.window.status is not None and not self.window.pending)
        self.window.timer.stop()
        self.window.message.setText("Previous main action")
        self.window.add_instruments.click()
        self.discovery = self.window.discovery

    def tearDown(self):
        self.manager.release.set()
        self.window.close()
        self.app.processEvents()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.backend.stop()
        self.home.stop()
        self.tmp.cleanup()

    def until(self, predicate):
        """Pump Qt until an asynchronous UI/IPC transition is visible."""
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail("GUI request did not finish")

    def enumerate(self):
        self.discovery.enumerate.click()
        self.until(lambda: self.discovery.pending_action is None)

    def row(self, resource):
        return next(row for row, candidate in enumerate(self.discovery.candidates)
                    if candidate["resource"] == resource)

    def button(self, resource, name):
        panel = self.discovery.table.cellWidget(self.row(resource), 3)
        return panel.findChild(self.qt.QPushButton, name)

    def combo(self, resource):
        return self.discovery.table.cellWidget(self.row(resource), 2)

    def test_selection_registration_cancellation_and_inventory_refresh(self):
        """Unknown rows require a driver; registration prompts, leaves disconnected and refreshes inventory."""
        self.assertTrue(self.discovery.table.alternatingRowColors())
        self.assertFalse(self.discovery.identify_all.isEnabled())
        self.enumerate()
        self.assertEqual(self.discovery.table.rowCount(), 3)
        self.assertEqual(self.manager.queries, [])
        self.assertEqual(self.combo(VNA).currentText(), "— unknown —")
        self.assertFalse(self.button(VNA, "register").isEnabled())
        self.button(VNA, "identify").click()
        self.until(lambda: self.discovery.pending_action is None)
        self.assertEqual(self.combo(VNA).currentData(), "AgilentN5230A")
        self.assertTrue(self.button(VNA, "register").isEnabled())
        self.assertEqual(set(self.combo(VNA).itemData(i) for i in range(1, self.combo(VNA).count())),
                         set(self.gui.DRIVERS))
        with patch.object(self.qt.QInputDialog, "getText", return_value=("cancelled", False)) as prompt:
            self.button(VNA, "register").click()
        self.assertEqual(prompt.call_args.kwargs["text"], "AgilentN5230A")
        self.assertFalse(self.discovery.candidates[self.row(VNA)]["registered"])
        self.combo(VNA).setCurrentText("AgilentE8363C")  # Explicit selection is authoritative.
        with patch.object(self.qt.QInputDialog, "getText", return_value=("bench_vna", True)):
            self.button(VNA, "register").click()
        self.until(lambda: self.window.status["instruments"].get("bench_vna") is not None
                   and not self.window.pending)
        spec = self.window.status["instruments"]["bench_vna"]
        self.assertEqual(spec["driver"], "AgilentE8363C")
        self.assertEqual(spec["state"], "DISCONNECTED")
        self.assertFalse(spec["worker_running"])
        self.assertFalse(self.button(VNA, "identify").isEnabled())
        self.assertFalse(self.button(VNA, "register").isEnabled())
        self.assertFalse(self.combo(VNA).isEnabled())
        self.assertEqual(self.window.table.rowCount(), 1)
        self.assertEqual(self.window.message.text(), "Previous main action")
        self.assertEqual(self.manager.queries, [(VNA, "*IDN?")])

    def test_identify_all_confirmation_skips_serial_and_reports_failures(self):
        """Bulk probing is confirmed and serial remains an explicit per-row action."""
        self.enumerate()
        with patch.object(self.qt.QMessageBox, "question", return_value=self.qt.QMessageBox.StandardButton.No):
            self.discovery.identify_all.click()
        self.assertEqual(self.manager.queries, [])
        with patch.object(self.qt.QMessageBox, "question", return_value=self.qt.QMessageBox.StandardButton.Yes):
            self.discovery.identify_all.click()
        self.until(lambda: self.discovery.pending_action is None and not self.discovery.queue)
        self.assertEqual(self.manager.queries, [(VNA, "*IDN?"), (UNKNOWN, "*IDN?")])
        self.assertIn("no response", self.discovery.table.item(self.row(UNKNOWN), 1).text())
        self.assertFalse(self.button(UNKNOWN, "register").isEnabled())
        self.combo(UNKNOWN).setCurrentText("YokogawaGS200")
        self.assertTrue(self.button(UNKNOWN, "register").isEnabled())
        self.button(SERIAL, "identify").click()
        self.until(lambda: self.discovery.pending_action is None)
        self.assertEqual(self.manager.queries[-1], (SERIAL, "*IDN?"))
        self.assertEqual(self.window.message.text(), "Previous main action")

    def test_manual_selection_survives_identify_and_ambiguity_stays_unknown(self):
        """Hints never silently replace an explicit choice or select an ambiguous match."""
        self.enumerate()
        self.combo(VNA).setCurrentText("YokogawaGS200")
        self.button(VNA, "identify").click()
        self.until(lambda: self.discovery.pending_action is None)
        self.assertEqual(self.combo(VNA).currentData(), "YokogawaGS200")
        self.enumerate()
        with patch.object(self.gui.DRIVERS["AgilentE8363C"], "IDN_STRING", "N5230A"):
            self.button(VNA, "identify").click()
            self.until(lambda: self.discovery.pending_action is None)
        self.assertIsNone(self.combo(VNA).currentData())
        self.assertIn("AgilentE8363C", self.combo(VNA).toolTip())
        self.assertIn("AgilentN5230A", self.combo(VNA).toolTip())
        self.assertFalse(self.button(VNA, "register").isEnabled())

    def test_probe_keeps_qt_responsive_and_close_stops_remaining_probes(self):
        """Hiding a busy sub-window never waits on VISA or starts the next bulk probe."""
        self.enumerate()
        self.manager.entered = threading.Event()
        with patch.object(self.qt.QMessageBox, "question", return_value=self.qt.QMessageBox.StandardButton.Yes):
            self.discovery.identify_all.click()
        self.until(lambda: self.manager.entered.is_set())
        self.assertFalse(self.discovery.enumerate.isEnabled())
        tick = []
        self.gui.QtCore.QTimer.singleShot(0, lambda: tick.append(True))
        self.until(lambda: bool(tick))
        self.window._request("STATUS")
        self.until(lambda: not self.window.pending)
        self.assertIsNotNone(self.discovery.pending_action)
        start = time.monotonic()
        self.discovery.close()
        self.assertLess(time.monotonic() - start, .5)
        self.assertFalse(self.discovery.isVisible())
        self.manager.release.set()
        self.until(lambda: self.discovery.pending_action is None)
        self.assertEqual(self.manager.queries, [(VNA, "*IDN?")])
        self.assertEqual(self.window.status["instruments"], {})
        self.window.add_instruments.click()
        self.assertIs(self.window.discovery, self.discovery)
        self.assertTrue(self.discovery.enumerate.isEnabled())

    def test_server_failures_do_not_replace_main_feedback_or_register_a_row(self):
        """Registration conflicts and discovery errors stay local and preserve an actionable row."""
        from auspex_v2.instruments.server import InstrumentServerClient

        self.enumerate()
        self.combo(VNA).setCurrentText("AgilentN5230A")
        with InstrumentServerClient(self.server.path) as client:
            client.register("taken", "SimulatedSource", "SIM::taken")
        with patch.object(self.qt.QInputDialog, "getText", return_value=("taken", True)):
            self.button(VNA, "register").click()
        self.until(lambda: self.discovery.pending_action is None)
        self.assertIn("REGISTER failed", self.discovery.message.text())
        self.assertTrue(self.button(VNA, "register").isEnabled())
        self.assertFalse(self.discovery.candidates[self.row(VNA)]["registered"])
        with patch.object(self.manager, "list_resources", side_effect=RuntimeError("enumeration unavailable")):
            self.enumerate()
        self.assertEqual(self.discovery.table.rowCount(), 0)
        self.assertIn("enumeration unavailable", self.discovery.message.text())
        self.assertEqual(self.window.message.text(), "Previous main action")

    def test_run_lock_gates_discovery(self):
        """Polling ownership disables all discovery actions during a run and restores them afterward."""
        from auspex_v2.instruments.server import InstrumentServerClient

        self.enumerate()
        with InstrumentServerClient(self.server.path) as client:
            with client.begin_run("locked"):
                self.window._request("STATUS")
                self.until(lambda: not self.window.pending)
                self.assertFalse(self.discovery.enumerate.isEnabled())
                self.assertFalse(self.button(VNA, "identify").isEnabled())
            self.window._request("STATUS")
            self.until(lambda: not self.window.pending)
        self.assertTrue(self.discovery.enumerate.isEnabled())
        self.assertTrue(self.button(VNA, "identify").isEnabled())
