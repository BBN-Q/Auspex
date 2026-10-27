# Instrument Server

The persistent local service owns registered instrument connections in spawned
workers. Admission comes from the bundled driver modules' explicit `__all__`
exports, shared with top-level Python imports; there is no second allowlist to
edit. See the [export contract](../drivers/instrument_drivers.md#exporting-a-bundled-driver).
The current catalog contains:

- `SimulatedSource`, `SimulatedVNA`
- `AgilentN5183A`, `AgilentE8363C`, `AgilentN5230A`
- `YokogawaGS200`

Real drivers use PyVISA-py (`@py`) consistently in direct and server use. Install
with `python -m pip install -e '.[visa]'`. Registration performs no VISA import,
connection, probing, or hardware I/O; a missing backend fails on `connect()` or explicit discovery.

## Start and register

```sh
python utils/instrument_server.py
python utils/instrument_server.py --gui
python utils/instrument_server.py -q
python utils/instrument_server.py --clean
```

The default socket is `$AUSPEX_HOME/.auspex/instruments.sock` (mode `0600`),
using `$HOME` as the base when `AUSPEX_HOME` is unset; startup prints a note on
this fallback. `--socket PATH` selects another socket, but does not change the
log directory. The server truncates `$AUSPEX_HOME/.auspex/instruments.log` at
each successful launch, after claiming the socket lock. This best-effort debug
log contains command/state transitions and may include command arguments; each
entry carries a local `[YYYY-MM-DD HH:MM:SS]` timestamp. The instrument CLI/TUI
output omits these timestamps. The log is not an audit record or physical readback. This is trusted local IPC. Only one
service owns a socket.
Closing the optional GUI leaves the service running. With an interactive stdin,
the terminal commands remain available even with `--gui`: `list`, `help`,
`connect <id|all>`, `disconnect <id|all>`, `recover <id>`,
`terminate <id|all>`, `clear <id|all>`, and `shutdown`.
`list` numbers instruments by name; IDs refer to that snapshot until the next
`list`. `recover all` is intentionally unavailable. `clear <id>` deregisters a
single disconnected, unowned instrument; `clear all` requires the entire inventory
to be disconnected and unowned, then wipes it. Clear, Terminate All and shutdown
require `y` confirmation; shutdown disconnects and clears only on success. Bulk
connect attempts every disconnected instrument and reports individual failures.
The combined `utils/launch_auspex.py` starts its instrument child without stdin,
but its own interactive prompt (or optional `--tui`) provides these inventory
commands in Instruments mode, excluding standalone `shutdown`.

```python
from auspex_v2.instruments.server import InstrumentServerClient

with InstrumentServerClient() as server:
    server.register("source", "AgilentN5183A", "TCPIP0::192.0.2.10::5025::SOCKET",
                    command_timeout=None, io_timeout=10)
    server.register("vna", "AgilentE8363C", "TCPIP0::192.0.2.11::INSTR",
                    command_timeout=None, io_timeout=30)
    server.register("bias", "YokogawaGS200", "GPIB0::1::INSTR")
```

Driver IDs are canonical Python class names; old lowercase simulator IDs are not
aliases. Registration is declarative. Explicit `connect(name)` starts a worker and
opens hardware. Already-connected workers can be used through a live or run lease.

## Discover and register from the terminal

Both launcher consoles and the standalone instrument terminal support:

```text
enumerate visa
enumerate visa --serial
enumerate visa --prologix
enumerate visa --serial --prologix --show-permission-denied
identify b
identify all
register b AgilentN5230A
```

Enumeration collects potential VISA resources without opening them or constructing
drivers. Display defaults hide serial (`ASRL…`) and Prologix GPIB (`PRLGX…`)
resources, plus resources with a known permission-denied outcome. `--serial` and
`--prologix` independently reveal their categories; `PRLGX-ASRL…` is controlled
solely by `--prologix`. `--show-permission-denied` disables access-denial hiding,
but serial/Prologix filters still apply unless enabled or whitelisted. Options
apply to that snapshot; a new enumeration starts from defaults plus its options.
Hidden counts and matching whitelist counts are printed after each result list.

Each candidate, including hidden ones, gets a letter ID (`a`…`z`, `aa`…), separate
from numeric registered inventory IDs. Visible letters may therefore have gaps.
The output shows resource, suggested driver(s), identity/error, category and
registered/whitelisted status. Before identification the driver is `unknown`.
Re-enumeration replaces the snapshot; identification never renumbers candidates.
`identify <letter>` can explicitly retry a hidden candidate from that snapshot.

`identify <letter>` temporarily opens that address, sends only `*IDN?`, and closes
it. Open and I/O timeouts are one second each; backend discovery/connection work
can take longer, so this is not a hard whole-command deadline. Probing uses LF
terminations and default serial settings; some devices need different settings
and cannot be identified this way. `identify all` requires explicit `y`
confirmation and runs sequentially over **visible** candidates only, always skipping
serial, Prologix and registered resources, even if whitelisted or shown by options.
Selecting a serial/Prologix candidate explicitly allows a probe. Registered
resources are never probed.
Timeouts/errors mean **not identified**, not necessarily absent or unsupported.

Real VISA drivers declare `IDN_STRING`, a case-sensitive literal substring matched
against the identity response. Every match is shown; no match or several matches
never causes an automatic selection. Non-VISA drivers inherit `None`.
`register <letter> <driver>` always requires a canonical admitted driver name,
then prompts for an inventory name (blank cancels). Identification is optional;
the explicit driver need not match the suggestion. Registration records
**DISCONNECTED** without hardware I/O. Run `list`, then `connect <numeric-id>`
when ready. The GUI's **Add instruments…** window offers the same enumeration,
identification and explicit-driver registration workflow; see the
[GUI guide](instrument_server_gui.md).

Discovery runs in the instrument server and is rejected during a run, live session,
or busy worker operation. Temporary probing excludes new leases/registration;
status remains available. One shared VISA normalizer supplies both the actual
connection endpoint and the registration identity. Driver-specific bare IPv4
shorthand expands to `5025::SOCKET` for N5183A and `hpib7,16::INSTR` for the current
VNA drivers; discovery recognizes those endpoints as already registered. Common
VISA spellings/default fields are standardized without changing case-sensitive
serial paths or USB serial numbers. DNS aliases and all physical-device aliases
cannot be detected; different endpoints on the same IP are not automatically equated. Never probe an alternate address for a device already in use.
Enumeration is backend-dependent and may miss actual instruments; manual-address
registration through the Python client remains available. No subnet scan, reset,
configuration, automatic registration, or automatic connection is performed.

### Discovery whitelist and permission errors

The **instrument server** reloads `$AUSPEX_HOME/instrument_whitelist.txt` on every
enumeration, using `$HOME/instrument_whitelist.txt` when `AUSPEX_HOME` is unset.
This file lives directly in that base directory, **not** in `.auspex/`. It contains
one VISA address per line; blank lines and lines starting with `#` are ignored:

```text
# Always show these candidates
ASRL/dev/ttyUSB0::INSTR
PRLGX-ASRL/dev/ttyUSB1::1::INSTR
TCPIP0::192.0.2.11::inst0::INSTR
```

Matching uses the same address normalization as registration (including common
TCPIP INSTR aliases). **Whitelist entries must be complete VISA addresses**: bare
IPs/aliases are skipped with a line-numbered warning, since the whitelist has no
driver from which to infer an endpoint. Valid entries are still used. A whitelist
match overrides all display filters,
not ownership checks, registration validation or bulk-probing restrictions. It does
not add undiscovered addresses, open connections or trigger probes. Missing files
are normal; unreadable files produce a warning in terminal output/GUI discovery.
Edit the file directly and enumerate again; no whitelist-editing command is needed.

Permission-denied status comes only from an error attributable to an explicit
identification attempt (OS permission errors, VISA access-denied status or a wrapped
backend permission-denied error). Backend diagnostic logs are **not** scraped, and
no extra opens or filesystem access tests are performed just to filter candidates.
The server retains last-observed permission denials across enumerations until an
explicit retry returns another outcome or the server restarts. Use
`--show-permission-denied` or a whitelist entry to reveal them; an explicit letter
retry also works while hidden. Other failures such as timeouts remain visible.

Python clients can use `server.enumerate("visa")` for the **unfiltered** candidate
list, with `serial`, `prologix`, `permission_denied`, `whitelisted` and `registered`
fields. `server.enumerate("visa", include_metadata=True)` instead returns an envelope
with `candidates`, `whitelist_path` and `whitelist_warning`; operator interfaces use
this form to show warnings even when no resources were found.
`server.identify(resource)` returns identity, suggestions, an error field and
`permission_denied`. Letter IDs belong only to the operator's local snapshot.

### Saved inventory normalization and conflicts

At startup the server recomputes each saved definition's canonical endpoint without
connecting hardware, preserving its connection/recovery state under the normal
restart rules. If several saved entries resolve to the same endpoint, **none of
that conflicting group is restored**; unrelated entries remain available. Invalid
or unavailable saved drivers are also skipped with a clear error.

Errors appear in service logs, terminal `list`/startup status and `status()` under
`registration_errors`. The GUI shows a popup once per changed set of errors and
labels the connection banner with the rejected-registration count. Polling does
not repeatedly reopen the same popup.

The original skipped definitions and error messages are retained under
`rejected_instruments` in the server's state JSON, not silently merged or deleted.
They remain excluded on later restarts. After inspecting hardware, explicitly
re-register a corrected definition (the original inventory name is available);
that removes its rejection record. Normal registration still rejects an endpoint
already registered under another name. Confirmed Clear Instruments/`clear all`
discards the rejection records along with the disconnected inventory. Forced
`--clean` also discards them. Neither action verifies physical safety, and skipped
entries are not connected or included in the registered domain.

## Timeouts

Two independent settings are persisted per registration:

- `io_timeout`: optional positive seconds for one VISA I/O operation. `None` uses
  the driver's default (N5183A 3 s, VNA 10 s, GS200/PyVISA default). Change it only
  while disconnected with `set_io_timeout(name, seconds_or_none)`.
- `command_timeout`: optional positive seconds for the whole worker request,
  including connect, GET/SET/CALL, acquisition, and disconnect. `None` means no
  InstrumentServer deadline. Change it while unowned with
  `set_command_timeout(name, seconds_or_none)`.

Clients impose no hidden shorter socket deadline. During connection, GET/SET/CALL,
and disconnection, `status()` remains responsive and reports `busy: true` with
`operation` equal to `connecting`, `executing`, or `disconnecting` (otherwise null).
Conflicting operations are rejected; `terminate()` may kill the worker in any
phase. Shutdown kills already-busy workers rather than sending another request;
idle workers are disconnected gracefully. A terminated worker's late reply cannot
undo termination or overwrite a subsequently recovered connection. VNA
`get_trace()` and `reaverage()` have no implicit total sweep deadline; callers may
pass an explicit positive `timeout=`. Cancellation cannot interrupt an in-flight
instrument call and waits for its return or configured command timeout.

## Live use

```python
with InstrumentServerClient() as server:
    server.connect("source")
    with server.live("source") as source:
        source.frequency = 5e9
        source.power = -40
        source.output = True
        print(source.frequency)
```

A live proxy exposes only declared settings and operations. GET/SET enforce declared
permissions and setting keyword names; unsupported commands fail before worker
dispatch. Operation Python signatures are checked by the worker. Closing a live
lease leaves hardware connected; disconnect explicitly when desired.

## Ownership and recovery

| Action | Result |
| --- | --- |
| `register(...)` | Persist `DISCONNECTED`; no hardware I/O. |
| `connect(name)` / `run.connect(name)` | Spawn and connect a worker; failure → `UNKNOWN`. |
| `live(name)` | Lease one connected instrument. |
| `disconnect(name)` / `disconnect_all()` | Gracefully close workers; failure → `UNKNOWN`. |
| `recover(name, reason=None)` | Operator acknowledgement: `UNKNOWN`/`TERMINATED` → `DISCONNECTED`; no readback or reconnect. |
| `terminate(name)` / `terminate_all()` | Kill workers and latch `TERMINATED`; not a hardware emergency-off. |
| `clear_instruments()` | Remove only fully disconnected, unowned registrations. |

`begin_run(run_id)` locks the whole registered domain against live access. A run
connects only its required instruments. Loss of a run client leaves an orphaned
lockout; after operator inspection, `reconcile_run(reason)` releases it, and affected
instruments can be recovered separately. Commands are never automatically retried.

`--clean` is an intentional forced reset: it discards saved registrations and run
lockout without connecting, disconnecting, reading, or changing hardware. It may
forget uncertain real-device state. The operator is responsible for inspection and
for handling any subsequent registration/connection error.

Saved registrations and run lockout survive restart. A normal stop attempts worker
disconnect. Previously connected state becomes `UNKNOWN` after an uncertain restart;
interrupted runs become orphaned. Nothing resumes automatically.

Protocol version 4 uses a 64 KiB JSON control envelope and up to 16 inline numeric
NumPy arrays totaling 8 MiB per result. Typed result dictionaries preserve scalar
keys such as integer VNA port numbers. This suits VNA traces, not digitizer streaming. Console output and ExperimentServer's execution log are diagnostics,
not a durable audit system. Direct drivers remain available and intentionally
bypass server ownership. This trusted laboratory software does not implement
hardware interlocks or prove physical safety.
