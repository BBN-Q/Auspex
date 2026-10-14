# Instrument Server GUI — operator prototype

The optional **pyqtgraph + PySide6** inventory client displays registered devices, their connection state (`DISCONNECTED`, `CONNECTED`, `UNKNOWN`, `TERMINATED`), worker presence and separate ownership (`Free`, `Live session`, `Run locked`). It never opens sessions, reads instrument settings or data, or registers instruments; register from a notebook. A disconnected GUI never displays stale status as current.

Run `python v2/utils/instrument_server.py --gui` to start the service and window, or attach a standalone window with `python v2/utils/instrument_server_gui.py --socket PATH`. Closing the GUI does not stop the service. Headless service use never requires Qt.

Per-device actions follow the current state: **Connect** on disconnected devices, **Disconnect** on connected devices, **Recover** (with confirmation of operator inspection) on unknown/terminated devices, and **Terminate** on every registered device. Connect and Disconnect require no conflicting live/run ownership. Recover returns to `DISCONNECTED` without reconnecting or asserting physical safety. Terminate quarantines even a disconnected device with no worker, and kills its worker if present; it does not switch off hardware. Actions blocked by ownership are disabled; the domain banner displays active and orphaned runs.

Global actions:

- **Terminate All** quarantines all registered instruments and attempts to kill every worker, without releasing leases or run locks. Confirmation and a software-only warning are shown. This is **not** an emergency off.
- **Disconnect All** attempts graceful driver disconnection for every connected worker when no sessions or run exist. Registrations remain; failures are reported per instrument and latch `UNKNOWN`.
- **Clear Instruments** is enabled only when every registered device is `DISCONNECTED`, no workers remain, and there are no live sessions or run locks. It persists an empty inventory, so the next startup is blank. It does not disconnect devices implicitly.

The server, not the GUI, validates each request. Status polling and actions use a serialized background Qt thread; failures are shown in the window. None of these software operations provides a hardware interlock or proof of physical safety.
