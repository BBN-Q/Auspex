# Instrument Server (simulated prototype)

The server runs on the acquisition host. Notebooks, scripts, and the optional GUI are clients. It accepts **only** `simulated_source` and `simulated_vna`; it cannot control real hardware. The experiment coordinator does not exist yet.

## Start the service

From the repository root, with Python 3.13 and `auspex-v2` installed:

```sh
python v2/utils/instrument_server.py
python v2/utils/instrument_server.py --gui  # optional inventory GUI
python v2/utils/instrument_server.py -q     # no console activity
python v2/utils/instrument_server.py -c     # discard saved simulator state
```

The service stays in that terminal. Closing its GUI does **not** stop it. Ctrl+C stops the service and workers but keeps registrations. In an interactive headless terminal, type `shutdown` to disconnect workers, clear registrations, and exit; it refuses if any step is unsafe or fails. The module entry point, `python -m auspex_v2.instruments.server`, also accepts `--socket`, `-q`, and `-c` but has no GUI or interactive `shutdown` command.

The default socket is `~/.auspex/instruments.sock` (mode `0600`). Use `--socket PATH` to choose another. This is local POSIX IPC; do not expose the socket to other users. Only one service can own a socket at a time.

The CLI prints command and state activity by default; successful `STATUS` polling is silent. Its output is transient, can contain settings, and is **not** a durable audit log. `-q` suppresses routine output, not client errors.

## Use from Python

```python
from auspex_v2.instruments.server import InstrumentServerClient

with InstrumentServerClient() as server:
    server.register("source", "simulated_source", "SIM::SOURCE")
    server.connect("source")  # registration alone does not connect
    with server.live("source") as source:
        source.set_frequency(5e9)
        print(source.frequency)      # reads the worker, not a cache
        print(source.identify())     # declared operation

    server.register("vna", "simulated_vna", "SIM::VNA")
    with server.begin_run("demo") as run:
        run.connect("vna")          # explicit connection under the run lock
        vna = run.instrument("vna")
        frequencies, s21 = vna.get_trace(measurement="S21")

    server.disconnect_all()    # keeps registrations
    server.clear_instruments()  # persists an empty inventory
```

Live access requires a connected instrument. A live context yields a session-bound proxy: properties and `get_`/`set_` helpers use GET/SET; registered methods use CALL. `source.session` exposes the explicit `get`/`set`/`call` API. Parameterized commands accept keyword arguments, for example `source.set_channel_frequency(5e9, channel=2)`. Unknown proxy members and wrong argument names fail before worker dispatch. A proxy cannot be used after its session closes.

## Connection and ownership

| Action | Result |
| --- | --- |
| `register(name, driver, resource)` | Save identity as `DISCONNECTED`; no worker or hardware I/O. |
| `connect(name)` or `run.connect(name)` | Spawn a worker and call driver `connect()`; success → `CONNECTED`, failure → `UNKNOWN`. |
| `live(name)` | Lease one **connected** instrument; closing the lease leaves it connected. |
| `disconnect(name)` / `disconnect_all()` | Gracefully stop connected workers when unowned. A failed disconnect → `UNKNOWN`; `disconnect_all()` reports each failure. Registrations remain. |
| `recover(name, reason=None)` | After operator inspection, change `UNKNOWN`/`TERMINATED` → `DISCONNECTED`. **Does not reconnect or verify hardware safety.** |
| `terminate(name)` / `terminate_all()` | Latch `TERMINATED` and kill any worker, including for a disconnected registration. Does not release ownership or switch off hardware. |
| `clear_instruments()` | Remove and persist all registrations **only** when every device is disconnected, no workers remain, and no live sessions or run exist. |

`begin_run(run_id)` locks out live access to **every registered instrument**, connected or not. It fails if another run, any live session, or a device needing recovery exists. The run connects devices it needs explicitly. A disconnected live client loses its lease; a disconnected run client leaves an **orphaned lockout**. After inspection, call `reconcile_run(reason)` to release it, then recover affected instruments individually.

Deregistration requires a disconnected instrument and no live sessions or run. `status()` reports connection state, worker presence, ownership, and run lockout. Command failures or timeouts quarantine the worker; the server does not retry commands with uncertain outcomes. A result-encoding failure after a command completes reports an unavailable result without killing its worker.

## Saved state and limits

Registrations and run lockout survive service restarts. A normal stop disconnects workers; successfully disconnected devices remain `DISCONNECTED`. A crash while connected makes their saved state `UNKNOWN` on restart. An interrupted run returns as orphaned. Nothing resumes automatically.

`-c` / `--clean` **discards saved registrations and run lockout at startup** after acquiring the single-instance lock. It bypasses reconciliation: use it only for this simulated prototype. Prefer `disconnect_all()` and `clear_instruments()` for an orderly reset.

Protocol version 2 uses a 64 KiB JSON control envelope and up to 16 inline numeric NumPy arrays totaling 8 MiB per result. Arrays are copied through the worker pipe and Unix socket. This supports VNA-sized traces, **not digitizer streaming**; object arrays and socket-side pickle are unsupported.

This prototype has no real-driver server adapter, hardware readback or interlocks, operator authentication beyond socket permissions, durable lab-notebook audit trail, typed argument schemas, shared-memory streaming, or experiment runner. Direct use of **unregistered** drivers bypasses server lockout. Worker termination is not an emergency off. See the [GUI guide](instrument_server_gui.md) for operator actions.

Run offline tests from the repository root with `python -B -m unittest discover -s v2/tests -q`.
