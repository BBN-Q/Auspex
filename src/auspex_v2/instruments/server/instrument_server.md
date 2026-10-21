# Instrument Server

The persistent local service owns registered instrument connections in spawned
workers. It admits the explicit class-name registry:

- `SimulatedSource`, `SimulatedVNA`
- `AgilentN5183A`, `AgilentE8363C`, `AgilentN5230A`
- `YokogawaGS200`

Real drivers use PyVISA-py (`@py`) consistently in direct and server use. Install
with `python -m pip install -e '.[visa]'`. Registration performs no VISA import,
connection, probing, or hardware I/O; a missing backend fails only on `connect()`.

## Start and register

```sh
python utils/instrument_server.py
python utils/instrument_server.py --gui
python utils/instrument_server.py -q
python utils/instrument_server.py --clean
```

The default socket is `~/.auspex/instruments.sock` (mode `0600`); `--socket PATH`
selects another. This is trusted local IPC. Only one service owns a socket.
Closing the optional GUI leaves the service running. The interactive headless CLI
accepts `shutdown`, which disconnects and clears only on success.

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
