# Instrument Server GUI — operator prototype

The optional **pyqtgraph + PySide6** dark-themed inventory client displays registered devices, their connection state (`DISCONNECTED`, `CONNECTED`, `UNKNOWN`, `TERMINATED`), worker presence and separate ownership (`Free`, `Live session`, `Run locked`). It can discover and register VISA candidates in a separate **Add instruments…** window. It never opens live/run sessions or reads instrument settings or measurement data; explicit identification reads only `*IDN?`. A disconnected GUI never displays stale status as current.

Run `python utils/instrument_server.py --gui` to start the service and window, or attach a standalone window with `python utils/instrument_server_gui.py --socket PATH`. Closing the GUI does not stop the service. Headless service use never requires Qt.

Per-device actions follow the current state: **Connect** on disconnected devices, **Disconnect** on connected devices (shown disabled in other non-disconnected states), **Recover** on unknown/terminated devices, and **Terminate** on every registered device. **Clear** deregisters a disconnected, unowned device with confirmation. Connect and Disconnect require no conflicting live/run ownership. Per-device Recover and Terminate have no confirmation dialog; recovery still acknowledges operator inspection without verifying physical safety. Recover returns to `DISCONNECTED` without reconnecting or asserting physical safety. Terminate quarantines even a disconnected device with no worker, and kills its worker if present; it does not switch off hardware. Actions blocked by ownership are disabled; the domain banner displays active and orphaned runs.

## Add instruments

**Add instruments…** opens a non-modal window with the same dark palette, alternating
rows and per-row buttons as the inventory. Opening it performs no discovery or I/O.

- **Enumerate VISA** replaces the candidate list without opening resources. Rows
  show the VISA resource (serial resources are labeled), identity/probe result,
  driver dropdown and separate **Identify** / **Register** buttons.
- **Identify** probes that row through the server with `*IDN?`. A sole suggested
  driver is selected automatically unless the operator has explicitly chosen a
  driver. No match or multiple matches leaves **— unknown —**; the dropdown tooltip
  lists all suggestions. The dropdown includes every admitted driver, and manual
  selection is authoritative. Identification is optional.
- **Identify all** requires Yes/No confirmation, probes sequentially and skips
  serial and registered resources. Progress/results appear in this window, not in
  the main window's last-action message. Individual failed probes remain visible;
  transport/ownership errors stop the remaining bulk requests.
- **Register** is disabled for **— unknown —**. It prompts for an editable inventory
  name defaulting to the selected driver name. Cancel or a blank name changes
  nothing. Successful registration leaves **DISCONNECTED**, disables the row's
  actions and refreshes the main inventory. **Connect** remains in the main window.

Registered resources are labeled and disabled. Discovery actions are disabled when
last-polled ownership is unavailable, run/live locked or busy; the server validates
requests against current state regardless. Re-enumerate to refresh candidates after
inventory removal. Probes use the same one-second VISA open/I/O timeout and LF
termination policy as the terminal commands; not every resource supports `*IDN?`.

All requests run off the Qt event loop. Closing this window hides it immediately
and stops subsequent bulk probes; an already dispatched probe or explicit
registration may finish. It never implicitly registers or connects anything. The
main inventory window owns its worker lifetime and joins in-flight requests on
application close. Closing either window does not stop the instrument server.

## Inventory actions

Global actions:

- **Terminate All** quarantines all registered instruments and attempts to kill every worker, without releasing leases or run locks. Confirmation and a software-only warning are shown. This is **not** an emergency off.
- **Disconnect All** attempts graceful driver disconnection for every connected worker when no sessions or run exist. Registrations remain; failures are reported per instrument and latch `UNKNOWN`.
- **Clear Instruments** is enabled only when every registered device is `DISCONNECTED`, no workers remain, and there are no live sessions or run locks. Confirmation is required. It persists an empty inventory, so the next startup is blank. It does not disconnect devices implicitly.

The server, not the GUI, validates each request. Status polling and actions use a serialized background Qt thread. The message bar retains the last action's outcome across status refreshes; connectivity/status failures appear in the connection banner instead. None of these software operations provides a hardware interlock or proof of physical safety.
