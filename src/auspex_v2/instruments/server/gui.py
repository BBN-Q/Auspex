"""Optional Qt inventory, discovery and connection client for the instrument server."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, QT_LIB

from .client import InstrumentServerClient
from ..drivers.registry import DRIVERS
from ..drivers.addresses import normalize_visa_address
from .discovery import hidden_reason, bulk_identifiable, filter_summary
from ...paths import home_note

if QT_LIB != "PySide6":
    raise RuntimeError("The instrument GUI requires PySide6 (install auspex-v2[gui])")


class RequestWorker(QtCore.QObject):
    """Keep synchronous server requests off the GUI event loop."""

    finished = QtCore.Signal(str, object)

    def __init__(self, socket_path):
        super().__init__()
        self.socket_path = socket_path

    @QtCore.Slot(str, object)
    def perform(self, op, kwargs):
        """Use a short-lived, session-free client for one serialized request."""
        try:
            with InstrumentServerClient(self.socket_path) as client:
                result = client.request(op, **kwargs)
            self.finished.emit(op, (True, result))
        except Exception as exc:
            self.finished.emit(op, (False, str(exc)))


class DiscoveryWindow(QtWidgets.QWidget):
    """Discover VISA candidates and explicitly register them without connecting drivers."""

    requested = QtCore.Signal(str, object)
    registered = QtCore.Signal()

    def __init__(self, socket_path, parent):
        """Keep discovery requests and messages independent of the main inventory window."""
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Auspex V2 — Add instruments")
        self.resize(1600, 600)
        self.candidates = []
        self.pending_action = None
        self.queue = []
        self.available = False
        self.closed = False
        self.enumerate = QtWidgets.QPushButton("Enumerate VISA")
        self.enumerate.clicked.connect(self._enumerate)
        self.identify_all = QtWidgets.QPushButton("Identify all")
        self.identify_all.clicked.connect(self._identify_all)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.enumerate)
        header.addWidget(self.identify_all)
        header.addStretch()
        self.show_serial = QtWidgets.QCheckBox("Show Serial Resources")
        self.show_prologix = QtWidgets.QCheckBox("Show Prologix GPIB Resources")
        self.hide_permission_denied = QtWidgets.QCheckBox("Hide permission denied resources")
        self.hide_permission_denied.setChecked(True)
        filters = QtWidgets.QHBoxLayout()
        for checkbox in (self.show_serial, self.show_prologix, self.hide_permission_denied):
            checkbox.toggled.connect(self._apply_filters)
            filters.addWidget(checkbox)
        self.whitelist_warning = None
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["VISA resource", "Identity / result", "Driver", "Actions"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.message = QtWidgets.QLabel("Enumerate candidates, then identify or choose a driver to register.")
        self.message.setWordWrap(True)
        warning = QtWidgets.QLabel("Identify sends *IDN?; Identify all probes visible rows, skipping serial/Prologix. Registration does not connect.")
        warning.setStyleSheet("color: #ffd18a")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(filters)
        layout.addWidget(self.table)
        layout.addWidget(self.summary)
        layout.addWidget(warning)
        layout.addWidget(self.message)
        self.thread = QtCore.QThread(self)
        self.worker = RequestWorker(socket_path)
        self.worker.moveToThread(self.thread)
        self.requested.connect(self.worker.perform)
        self.worker.finished.connect(self._completed)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self._apply_filters()

    def _filters(self):
        return {"show_serial": self.show_serial.isChecked(), "show_prologix": self.show_prologix.isChecked(),
                "hide_permission_denied": self.hide_permission_denied.isChecked()}

    def _apply_filters(self):
        """Hide rows without discarding their identities, driver choices or in-flight row indices."""
        for row, candidate in enumerate(self.candidates):
            self.table.setRowHidden(row, hidden_reason(candidate, **self._filters()) is not None)
        text = filter_summary(self.candidates, **self._filters())
        if self.whitelist_warning:
            text += " Warning: " + self.whitelist_warning
        self.summary.setText(text)
        self._update_actions()

    def set_status(self, status):
        """Use inventory polling for advisory gating; the server still validates every action."""
        instruments = status["instruments"] if status else {}
        self.available = status is not None and status["run"] is None and all(
            not spec["live_sessions"] and not spec.get("busy") for spec in instruments.values())
        registered = {spec["physical"] for spec in instruments.values()}
        for row, candidate in enumerate(self.candidates):
            if normalize_visa_address(candidate["resource"]) in registered:
                candidate["registered"] = True
                self.table.item(row, 1).setText("Already registered")
        self._update_actions()

    def _update_actions(self):
        """Disable edits during requests and require an explicit driver before registration."""
        enabled = self.available and self.pending_action is None
        self.enumerate.setEnabled(enabled)
        self.identify_all.setEnabled(enabled and any(
            bulk_identifiable(c) and not hidden_reason(c, **self._filters()) for c in self.candidates))
        for row, candidate in enumerate(self.candidates):
            available = enabled and not candidate["registered"]
            combo = self.table.cellWidget(row, 2)
            combo.setEnabled(available)
            panel = self.table.cellWidget(row, 3)
            panel.findChild(QtWidgets.QPushButton, "identify").setEnabled(available)
            panel.findChild(QtWidgets.QPushButton, "register").setEnabled(
                available and combo.currentData() is not None)

    def _render(self):
        """Preserve explicit driver choices across probe results and table refreshes."""
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.candidates))
        for row, candidate in enumerate(self.candidates):
            kind = " [Prologix]" if candidate.get("prologix") else " [serial]" if candidate["serial"] else ""
            resource = candidate["resource"] + kind + (" [whitelisted]" if candidate.get("whitelisted") else "")
            detail = ("Already registered" if candidate["registered"] else
                      candidate["error"] or candidate["identity"] or "Not identified")
            for col, text in enumerate((resource, detail)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, col, item)
            combo = QtWidgets.QComboBox()
            combo.addItem("— unknown —", None)
            for name in sorted(DRIVERS):
                combo.addItem(name, name)
            combo.setCurrentIndex(max(0, combo.findData(candidate.get("selected_driver"))))
            combo.setToolTip("Suggestions: " + (", ".join(candidate["drivers"]) or "unknown"))
            combo.currentIndexChanged.connect(lambda _index, r=row: self._select_driver(r))
            self.table.setCellWidget(row, 2, combo)
            panel = QtWidgets.QWidget()
            buttons = QtWidgets.QHBoxLayout(panel)
            buttons.setContentsMargins(0, 0, 0, 0)
            identify = QtWidgets.QPushButton("Identify")
            identify.setObjectName("identify")
            identify.clicked.connect(lambda _checked=False, r=row: self._identify(r))
            register = QtWidgets.QPushButton("Register")
            register.setObjectName("register")
            register.clicked.connect(lambda _checked=False, r=row: self._register(r))
            buttons.addWidget(identify)
            buttons.addWidget(register)
            self.table.setCellWidget(row, 3, panel)
        self._apply_filters()

    def _select_driver(self, row):
        self.candidates[row]["selected_driver"] = self.table.cellWidget(row, 2).currentData()
        self.candidates[row]["manual_driver"] = True
        self._update_actions()

    def _request(self, op, row=None, **kwargs):
        if self.pending_action is not None or not self.available or self.closed:
            return
        self.pending_action = (op, row)
        self.message.setText(f"{op}: {kwargs.get('resource', kwargs.get('name', 'VISA candidates'))}…")
        self._update_actions()
        self.requested.emit(op, kwargs)

    def _enumerate(self):
        # Invalidate the old snapshot even if a refresh fails.
        self.candidates = []
        self._render()
        self._request("ENUMERATE", interface="visa", include_metadata=True)

    def _identify(self, row):
        self._request("IDENTIFY", row, interface="visa", resource=self.candidates[row]["resource"])

    def _identify_all(self):
        """Confirm once, then send one background probe at a time with visible progress."""
        if QtWidgets.QMessageBox.question(self, "Confirm identification",
                "Send *IDN? to visible, unregistered candidates? Serial and Prologix resources are skipped.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.queue = [row for row, c in enumerate(self.candidates)
                      if bulk_identifiable(c) and not hidden_reason(c, **self._filters())]
        self._next_probe()

    def _next_probe(self):
        while self.queue and self.available and not self.closed:
            row = self.queue.pop(0)
            candidate = self.candidates[row]
            if bulk_identifiable(candidate) and not hidden_reason(candidate, **self._filters()):
                self._identify(row)
                return
        self.queue = []

    def _register(self, row):
        """Require a name for an explicit driver choice; never identify or connect implicitly."""
        driver = self.table.cellWidget(row, 2).currentData()
        if driver is None:
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Register instrument", "Inventory name:", text=driver)
        if not accepted or not name.strip():
            return
        self._request("REGISTER", row, name=name.strip(), driver=driver,
                      resource=self.candidates[row]["resource"])

    @QtCore.Slot(str, object)
    def _completed(self, op, outcome):
        """Update only discovery feedback, and stop a bulk probe on transport/ownership errors."""
        _, row = self.pending_action
        self.pending_action = None
        ok, value = outcome
        if not ok:
            self.queue = []
            self.message.setText(f"{op} failed: {value}")
        elif op == "ENUMERATE":
            self.candidates = value["candidates"]
            self.whitelist_warning = value["whitelist_warning"]
            self.summary.setToolTip("Server whitelist: " + value["whitelist_path"])
            self.message.setText(f"Enumerated {len(self.candidates)} VISA candidates.")
        elif op == "IDENTIFY":
            candidate = self.candidates[row]
            candidate.update(value)
            if not candidate.get("manual_driver"):
                candidate["selected_driver"] = value["drivers"][0] if len(value["drivers"]) == 1 else None
            self.message.setText(f"Identify {candidate['resource']}: {value['error'] or value['identity']}")
        elif op == "REGISTER":
            self.candidates[row]["registered"] = True
            self.message.setText("Registered as DISCONNECTED. Connect from the main inventory window.")
            self.registered.emit()
        self._render()
        self._next_probe()

    def closeEvent(self, event):
        """Hide immediately and stop further bulk probes; an already dispatched request may finish."""
        self.closed = True
        self.queue = []
        super().closeEvent(event)

    def shutdown(self):
        """Let the owning main window join the worker before destroying Qt objects."""
        self.close()
        self.thread.quit()
        self.thread.wait()


class InstrumentWindow(QtWidgets.QWidget):
    """Show inventory, connection and ownership without instrument data access."""

    requested = QtCore.Signal(str, object)

    def __init__(self, socket_path=None):
        """Build the inventory window and start its serialized background client."""
        super().__init__()
        self.setWindowTitle("Auspex V2 — Instrument Server")
        self.resize(1400, 520)
        self.pending = False
        self.pending_action = None
        self.status = None
        self.socket_path = socket_path
        self.discovery = None
        self.refresh_after_pending = False
        self.closing = False
        self.reported_registration_errors = {}

        self.connection = QtWidgets.QLabel("Connecting…")
        self.domain = QtWidgets.QLabel("Domain: unknown")
        self.refresh = QtWidgets.QPushButton("Refresh")
        self.refresh.clicked.connect(lambda: self._request("STATUS"))
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.connection)
        header.addStretch()
        header.addWidget(self.domain)
        header.addWidget(self.refresh)
        self.add_instruments = QtWidgets.QPushButton("Add instruments…")
        self.add_instruments.clicked.connect(self._open_discovery)
        header.addWidget(self.add_instruments)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Name", "Driver", "Physical resource", "Connection", "Ownership", "Actions"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.terminate_all = QtWidgets.QPushButton("Terminate All")
        self.terminate_all.setStyleSheet("font-weight: bold; color: #ff8c95")
        self.terminate_all.clicked.connect(lambda: self._confirm("TERMINATE_ALL",
            "Kill and quarantine all registered workers? This does NOT switch off hardware."))
        self.disconnect_all = QtWidgets.QPushButton("Disconnect All")
        self.disconnect_all.clicked.connect(lambda: self._request("DISCONNECT_ALL"))
        self.clear_instruments = QtWidgets.QPushButton("Clear Instruments")
        self.clear_instruments.clicked.connect(lambda: self._confirm("CLEAR_INSTRUMENTS",
            "Deregister all instruments? All must be disconnected and unowned."))
        actions = QtWidgets.QHBoxLayout()
        for button in (self.terminate_all, self.disconnect_all, self.clear_instruments):
            actions.addWidget(button)
        self.warning = QtWidgets.QLabel("Terminate kills software workers only; disconnect does not switch off hardware.")
        self.warning.setStyleSheet("color: #ffd18a")
        self.message = QtWidgets.QLabel("Inventory only: no instrument settings or measurement data access.")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.table)
        layout.addLayout(actions)
        layout.addWidget(self.warning)
        layout.addWidget(self.message)

        self.thread = QtCore.QThread(self)
        self.worker = RequestWorker(socket_path)
        self.worker.moveToThread(self.thread)
        self.requested.connect(self.worker.perform)
        self.worker.finished.connect(self._completed)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(lambda: self._request("STATUS"))
        self.timer.start()
        self._update_actions()
        self._request("STATUS")

    def _open_discovery(self):
        """Reuse one non-modal discovery window without affecting inventory action feedback."""
        if self.discovery is None:
            self.discovery = DiscoveryWindow(self.socket_path, self)
            self.discovery.registered.connect(self._refresh_inventory)
        self.discovery.closed = False
        self.discovery.set_status(self.status)
        self.discovery.show()
        self.discovery.raise_()
        self.discovery.activateWindow()

    def _refresh_inventory(self):
        if self.pending:
            self.refresh_after_pending = True
        else:
            self._request("STATUS")

    def _confirm(self, op, question, **kwargs):
        """Require acknowledgement for destructive inventory or worker operations."""
        if QtWidgets.QMessageBox.question(self, "Confirm instrument action", question,
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.Yes:
            self._request(op, **kwargs)

    def _request(self, op, **kwargs):
        """Serialize inventory actions without opening live instrument sessions."""
        if self.closing or self.pending or op not in ("STATUS", "CONNECT", "DISCONNECT", "DISCONNECT_ALL",
                                       "CLEAR_INSTRUMENTS", "DEREGISTER", "TERMINATE", "RECOVER", "TERMINATE_ALL"):
            return
        self.pending = True
        self.pending_action = (op, kwargs)
        self._update_actions()
        self.requested.emit(op, kwargs)

    def _update_actions(self):
        """Gate inventory actions and propagate the latest ownership snapshot to discovery."""
        self.refresh.setEnabled(not self.pending)
        status = self.status
        instruments = status["instruments"] if status else {}
        free = status is not None and status["run"] is None and all(
            not spec["live_sessions"] for spec in instruments.values())
        self.terminate_all.setEnabled(not self.pending and bool(instruments))
        self.disconnect_all.setEnabled(not self.pending and free and any(
            spec["state"] == "CONNECTED" for spec in instruments.values()))
        self.clear_instruments.setEnabled(not self.pending and free and
            bool(instruments or status.get("registration_errors")) and all(
            spec["state"] == "DISCONNECTED" for spec in instruments.values()))
        if self.discovery is not None:
            self.discovery.set_status(status)
        for row in range(self.table.rowCount()):
            panel = self.table.cellWidget(row, 5)
            if panel is not None:
                for button in panel.findChildren(QtWidgets.QPushButton):
                    button.setEnabled(not self.pending and bool(button.property("allowed")))

    def _add_row_action(self, layout, name, label, op, *, allowed=True, confirm=None):
        """Add one state-dependent action without granting a live session."""
        button = QtWidgets.QPushButton(label)
        button.setProperty("allowed", allowed)
        if confirm:
            button.clicked.connect(lambda _checked=False: self._confirm(op, confirm, name=name))
        else:
            button.clicked.connect(lambda _checked=False: self._request(op, name=name))
        layout.addWidget(button)

    def _show_status(self, status):
        """Render a snapshot, never treating worker state as physical readback."""
        self.status = status
        errors = status.get("registration_errors", {})
        self.connection.setText("Connected to Instrument Server" +
                                (f" — {len(errors)} saved registrations rejected" if errors else ""))
        if errors != self.reported_registration_errors:
            self.reported_registration_errors = dict(errors)
            if errors:
                text = "\n\n".join(f"{name}: {error}" for name, error in sorted(errors.items()))
                QtCore.QTimer.singleShot(0, lambda: self._registration_error_popup(text))
        run = status["run"]
        self.domain.setText("Domain: " + (f"ORPHANED — {run['run_id']}" if run and run["orphaned"] else
                                           f"RUN LOCKED — {run['run_id']}" if run else "NO RUN LOCK"))
        instruments = sorted(status["instruments"].items())
        self.table.setRowCount(len(instruments))
        for row, (name, spec) in enumerate(instruments):
            ownership = ("Run locked" if run else "Live session" if spec["live_sessions"] else "Free")
            connection = spec["state"] + (" (worker running)" if spec.get("worker_running") else "")
            for col, text in enumerate((name, spec["driver"], spec["physical"], connection, ownership)):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
            panel = QtWidgets.QWidget()
            buttons = QtWidgets.QHBoxLayout(panel)
            buttons.setContentsMargins(0, 0, 0, 0)
            free = not run and not spec["live_sessions"]
            if spec["state"] == "DISCONNECTED":
                self._add_row_action(buttons, name, "Connect", "CONNECT", allowed=free)
            else:
                self._add_row_action(buttons, name, "Disconnect", "DISCONNECT",
                                     allowed=free and spec["state"] == "CONNECTED")
            if spec["state"] in ("TERMINATED", "UNKNOWN"):
                self._add_row_action(buttons, name, "Recover", "RECOVER", allowed=free)
            self._add_row_action(buttons, name, "Terminate", "TERMINATE")
            self._add_row_action(buttons, name, "Clear", "DEREGISTER",
                                 allowed=free and spec["state"] == "DISCONNECTED",
                                 confirm=f"Deregister {name}? This removes it from saved inventory.")
            self.table.setCellWidget(row, 5, panel)
        self._update_actions()

    def _registration_error_popup(self, text):
        if not self.closing:
            QtWidgets.QMessageBox.warning(self, "Saved instruments not restored", text)

    @QtCore.Slot(str, object)
    def _completed(self, op, outcome):
        """Show the result and refresh after actions without blocking Qt."""
        self.pending = False
        ok, value = outcome
        action = self.pending_action
        self.pending_action = None
        if op == "STATUS":
            if ok:
                self._show_status(value)
            else:
                self.status = None
                self.table.setRowCount(0)
                self.connection.setText(f"Disconnected — status failed: {value}")
                self.domain.setText("Domain: unknown")
        else:
            name = action[1].get("name") if action else None
            label = f"{op} {name}" if name else op
            failures = value.get("failures") if ok and isinstance(value, dict) else None
            detail = f"partial failure: {failures}" if failures else "request completed" if ok else f"failed: {value}"
            self.message.setText(f"Last action: {label} — {detail}")
            self._request("STATUS")
        self._update_actions()
        if self.refresh_after_pending:
            self.refresh_after_pending = False
            self._refresh_inventory()

    def closeEvent(self, event):
        """Finish the in-flight socket request before destroying its Qt thread."""
        self.closing = True
        self.timer.stop()
        if self.discovery is not None:
            self.discovery.shutdown()
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


def dark_theme(app):
    """Apply an accessible dark palette to the standalone operator GUI."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    for role, color in ((QtGui.QPalette.ColorRole.Window, "#202632"),
                        (QtGui.QPalette.ColorRole.WindowText, "#e7eef8"),
                        (QtGui.QPalette.ColorRole.Base, "#171b24"),
                        (QtGui.QPalette.ColorRole.AlternateBase, "#293241"),
                        (QtGui.QPalette.ColorRole.Text, "#e7eef8"),
                        (QtGui.QPalette.ColorRole.Button, "#344052"),
                        (QtGui.QPalette.ColorRole.ButtonText, "#e7eef8"),
                        (QtGui.QPalette.ColorRole.Highlight, "#337ba3"),
                        (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff"),
                        (QtGui.QPalette.ColorRole.ToolTipBase, "#293241"),
                        (QtGui.QPalette.ColorRole.ToolTipText, "#e7eef8")):
        palette.setColor(role, QtGui.QColor(color))
    for role in (QtGui.QPalette.ColorRole.Text, QtGui.QPalette.ColorRole.ButtonText,
                 QtGui.QPalette.ColorRole.WindowText):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor("#9eabbc"))
    app.setPalette(palette)


def main(argv=None):
    """Launch the optional GUI without starting or owning the server."""
    parser = argparse.ArgumentParser(description="Auspex V2 instrument inventory and discovery GUI")
    parser.add_argument("--socket", help="Path to an explicitly started instrument server")
    args = parser.parse_args(argv)
    if note := home_note():
        print(f"[Instrument GUI] {note}", flush=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dark_theme(app)
    window = InstrumentWindow(args.socket)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
