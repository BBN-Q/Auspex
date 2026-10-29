# Instrument Server GUI — operator prototype

The optional **pyqtgraph + PySide6** dark-themed inventory client displays registered devices, their connection state (`DISCONNECTED`, `CONNECTED`, `UNKNOWN`, `TERMINATED`), worker presence and separate ownership (`Free`, `Live session`, `Run locked`). It can discover and register VISA candidates in a separate **Add instruments…** window. It never opens live/run sessions or reads instrument settings or measurement data; explicit identification reads only `*IDN?`. A disconnected GUI never displays stale status as current.

Run `python utils/instrument_server.py --gui` to start the service and window, or attach a standalone window with `python utils/instrument_server_gui.py --socket PATH`. Closing the GUI does not stop the service. Headless service use never requires Qt.

Per-device actions follow the current state: **Connect** on disconnected devices, **Disconnect** on connected devices (shown disabled in other non-disconnected states), **Recover** on unknown/terminated devices, and **Terminate** on every registered device. **Clear** deregisters a disconnected, unowned device with confirmation. Connect and Disconnect require no conflicting live/run ownership. Per-device Recover and Terminate have no confirmation dialog; recovery still acknowledges operator inspection without verifying physical safety. Recover returns to `DISCONNECTED` without reconnecting or asserting physical safety. Terminate quarantines even a disconnected device with no worker, and kills its worker if present; it does not switch off hardware. Actions blocked by ownership are disabled; the domain banner displays active and orphaned runs.

## Add instruments

**Add instruments…** opens a non-modal window with the same dark palette, alternating
rows and per-row buttons as the inventory. Opening it performs no discovery or I/O.

Three checkboxes control display only:

- **Show Serial Resources** — unchecked by default; reveals `ASRL…` candidates.
- **Show Prologix GPIB Resources** — unchecked by default; reveals `PRLGX…`
  candidates, including `PRLGX-ASRL…`, independently of the Serial checkbox.
- **Hide permission denied resources** — checked by default; hides last-observed
  per-resource access denials, not timeouts or other failures.

The summary counts hidden resources and matching whitelist entries. Toggling filters
does not probe or discard candidates, identities or manual driver choices.
The server reloads `$AUSPEX_HOME/instrument_whitelist.txt` (HOME fallback, directly
in the base directory rather than `.auspex/`) on each enumeration. One address per
line, with blank lines and `#` comment lines ignored. Matching enumerated resources
stay visible regardless of all three checkboxes. The summary tooltip shows the
server-side path; unreadable-file warnings appear in the summary. Missing files
are normal. See the [operator guide](instrument_server.md#discovery-whitelist-and-permission-errors)
for matching and retry details. Whitelisting never adds probes or changes ownership.

- **Enumerate VISA** replaces the candidate list without opening resources. Rows
  show the VISA resource (serial/Prologix and whitelisted resources are labeled), identity/probe result,
  driver dropdown and separate **Identify** / **Register** buttons.
- **Identify** probes that row through the server with `*IDN?`. A sole suggested
  driver is selected automatically unless the operator has explicitly chosen a
  driver. No match or multiple matches leaves **— unknown —**; the dropdown tooltip
  lists all suggestions. The dropdown includes every admitted driver, and manual
  selection is authoritative. Identification is optional.
- **Identify all** requires Yes/No confirmation and probes visible candidates
  sequentially, always skipping serial, Prologix and registered resources—even when
  whitelisted. Progress/results appear in this window, not in the main window's
  last-action message. Permission-denied rows hide according to the checkbox;
  unchecking it reveals them for an explicit per-row retry. Other failed probes
  remain visible; transport/ownership errors stop remaining bulk requests. Changing
  filters during a bulk operation also excludes newly hidden rows not yet probed.
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

## Saved registration errors

The server normalizes saved definitions before restoring inventory. If multiple
entries resolve to one connection endpoint, all entries in that conflicting group
are skipped, without connecting hardware. The main window shows a **Saved
instruments not restored** popup with names/drivers/addresses and a rejected count
in the connection banner. Unchanged errors do not reopen the popup on each poll.
Unrelated instruments remain available. Original rejected definitions are retained
in the server state for inspection; explicitly register a corrected definition to
resolve its record, or confirm Clear Instruments to discard the records along with
the disconnected inventory. See the [operator guide](instrument_server.md#saved-inventory-normalization-and-conflicts).

Whitelist entries also require complete VISA addresses; ambiguous bare IP entries
warn in the discovery summary rather than guessing an endpoint.

## Inventory actions

Live ownership shows the optional notebook label and short session ID. Per-device
**Release live ownership…** requires confirmation identifying that owner, even if
the notebook cannot be found. It leaves hardware connected and outputs unchanged,
and cannot release run ownership. A changed owner rejects the stale action. Old
handles lose access; an in-flight command may finish while ownership stays locked.
A pending release is labelled `releasing`; a stuck command still requires the
termination/quarantine procedure. This is separate from run reconciliation.

Global actions:

- **Terminate All** quarantines all registered instruments and attempts to kill every worker, without releasing leases or run locks. Confirmation and a software-only warning are shown. This is **not** an emergency off.
- **Disconnect All** attempts graceful driver disconnection for every connected worker when no sessions or run exist. Registrations remain; failures are reported per instrument and latch `UNKNOWN`.
- **Clear Instruments** is enabled only when every registered device is `DISCONNECTED`, no workers remain, and there are no live sessions or run locks. Confirmation is required. It persists an empty inventory, so the next startup is blank. It does not disconnect devices implicitly.

The server, not the GUI, validates each request. Status polling and actions use a serialized background Qt thread. The message bar retains the last action's outcome across status refreshes; connectivity/status failures appear in the connection banner instead. None of these software operations provides a hardware interlock or proof of physical safety.
