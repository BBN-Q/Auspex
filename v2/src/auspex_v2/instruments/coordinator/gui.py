"""Optional, status-only Qt client for the local instrument coordinator."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from pyqtgraph.Qt import QtCore, QtWidgets, QT_LIB

from .client import CoordinatorClient

if QT_LIB != "PySide6":
    raise RuntimeError("The instrument GUI requires PySide6 (install auspex-v2[gui])")


class RequestWorker(QtCore.QObject):
    """Keep synchronous coordinator requests off the GUI event loop."""

    finished = QtCore.Signal(str, object)

    def __init__(self, socket_path):
        super().__init__()
        self.socket_path = socket_path

    @QtCore.Slot(str, object)
    def perform(self, op, kwargs):
        """Use a short-lived, session-free client for one serialized request."""
        try:
            with CoordinatorClient(self.socket_path) as client:
                result = client.request(op, **kwargs)
            self.finished.emit(op, (True, result))
        except Exception as exc:
            self.finished.emit(op, (False, str(exc)))


class InstrumentWindow(QtWidgets.QWidget):
    """Show inventory, connection and ownership without instrument data access."""

    requested = QtCore.Signal(str, object)

    def __init__(self, socket_path=None):
        """Build a status-only window and start its serialized background client."""
        super().__init__()
        self.setWindowTitle("Auspex V2 — Instruments")
        self.resize(900, 420)
        self.pending = False
        self.status = None

        self.connection = QtWidgets.QLabel("Connecting…")
        self.domain = QtWidgets.QLabel("Domain: unknown")
        self.refresh = QtWidgets.QPushButton("Refresh")
        self.refresh.clicked.connect(lambda: self._request("STATUS"))
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.connection)
        header.addStretch()
        header.addWidget(self.domain)
        header.addWidget(self.refresh)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Name", "Driver", "Physical resource", "Connection", "Ownership", "Actions"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.terminate_all = QtWidgets.QPushButton("Terminate All")
        self.terminate_all.setStyleSheet("font-weight: bold; color: #b00020")
        self.terminate_all.clicked.connect(lambda: self._confirm("TERMINATE_ALL",
            "Kill and quarantine all registered workers? This does NOT switch off hardware."))
        self.disconnect_all = QtWidgets.QPushButton("Disconnect All")
        self.disconnect_all.clicked.connect(lambda: self._request("DISCONNECT_ALL"))
        self.clear_instruments = QtWidgets.QPushButton("Clear Instruments")
        self.clear_instruments.clicked.connect(lambda: self._confirm("CLEAR_INSTRUMENTS",
            "Deregister all disconnected instruments and clear saved inventory?"))
        actions = QtWidgets.QHBoxLayout()
        for button in (self.terminate_all, self.disconnect_all, self.clear_instruments):
            actions.addWidget(button)
        self.warning = QtWidgets.QLabel("Terminate kills software workers only; disconnect does not switch off hardware.")
        self.message = QtWidgets.QLabel("Status only: no instrument settings or data access.")
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

    def _confirm(self, op, question, **kwargs):
        """Require acknowledgement for destructive inventory or worker operations."""
        if QtWidgets.QMessageBox.question(self, "Confirm instrument action", question,
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.Yes:
            self._request(op, **kwargs)

    def _request(self, op, **kwargs):
        """Serialize inventory actions without opening live instrument sessions."""
        if self.pending or op not in ("STATUS", "CONNECT", "DISCONNECT", "DISCONNECT_ALL",
                                       "CLEAR_INSTRUMENTS", "TERMINATE", "RECOVER", "TERMINATE_ALL"):
            return
        self.pending = True
        self._update_actions()
        self.requested.emit(op, kwargs)

    def _update_actions(self):
        self.refresh.setEnabled(not self.pending)
        status = self.status
        instruments = status["instruments"] if status else {}
        free = status is not None and status["run"] is None and all(
            not spec["live_sessions"] for spec in instruments.values())
        self.terminate_all.setEnabled(not self.pending and bool(instruments))
        self.disconnect_all.setEnabled(not self.pending and free and any(
            spec["state"] == "CONNECTED" for spec in instruments.values()))
        self.clear_instruments.setEnabled(not self.pending and free and bool(instruments) and all(
            spec["state"] == "DISCONNECTED" for spec in instruments.values()))
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
        self.connection.setText("Connected to instrument coordinator")
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
            elif spec["state"] == "CONNECTED":
                self._add_row_action(buttons, name, "Disconnect", "DISCONNECT", allowed=free)
            elif spec["state"] in ("TERMINATED", "UNKNOWN"):
                self._add_row_action(buttons, name, "Recover", "RECOVER", allowed=free,
                                     confirm="Confirm operator inspection. Recovery marks this device disconnected, not safe or connected.")
            self._add_row_action(buttons, name, "Terminate", "TERMINATE",
                                 confirm="Kill the worker if running and quarantine this instrument? Hardware may remain active.")
            self.table.setCellWidget(row, 5, panel)
        self._update_actions()

    @QtCore.Slot(str, object)
    def _completed(self, op, outcome):
        """Show the result and refresh after actions without blocking Qt."""
        self.pending = False
        ok, value = outcome
        if op == "STATUS":
            if ok:
                self._show_status(value)
                self.message.setText("Status snapshot; no physical settings have been read.")
            else:
                self.status = None
                self.table.setRowCount(0)
                self.connection.setText("Disconnected — status unknown")
                self.domain.setText("Domain: unknown")
                self.message.setText(f"Status failed: {value}")
        else:
            self.message.setText(f"{op}: {value if ok else 'failed: ' + value}")
            self._request("STATUS")
        self._update_actions()

    def closeEvent(self, event):
        """Finish the in-flight socket request before destroying its Qt thread."""
        self.timer.stop()
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


def main(argv=None):
    """Launch the optional GUI without starting or owning the coordinator."""
    parser = argparse.ArgumentParser(description="Auspex V2 instrument status and termination GUI")
    parser.add_argument("--socket", help="Path to an explicitly started instrument coordinator")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication([])
    window = InstrumentWindow(args.socket)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
