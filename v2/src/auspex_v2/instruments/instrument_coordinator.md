# Local instrument coordinator — first prototype

This is an **instrument coordinator only**, not the experiment coordinator. It runs in a separate terminal on the acquisition host; notebooks and scripts are clients. The current implementation accepts **only** `simulated_source` and `simulated_vna` registrations and never opens real hardware. Scalar control uses JSON; NumPy results use framed binary data. Do not expose its Unix socket to other users or assume it provides interlocks or an audit trail.

From the repository root (Python 3.13 with `auspex-v2` installed), start the operator CLI in its own terminal:

```sh
python v2/utils/instrument_coordinator.py
# or: python v2/utils/instrument_coordinator.py --socket /path/to/instruments.sock
# or: python v2/utils/instrument_coordinator.py -q  # no splash or console activity
```

By default it displays an ASCII splash and timestamps commands (`GET`, `SET`, `CALL`), their small results, session activity, rejected requests, and instrument/domain state transitions. `-q`/`--quiet` silences the splash, startup message, and activity output; errors still reach clients. The same options work with `python -m auspex_v2.instruments.coordinator` (without the splash). **Console output is transient and may include instrument settings; it is not the planned durable lab notebook or a secret-safe audit log.**

This first implementation targets POSIX hosts (Unix-domain sockets and `fcntl`). The socket defaults to `~/.auspex/instruments.sock` (mode `0600`). Use `--socket /path/to/instruments.sock` to override it. On the **same host**, use the synchronous client:

```python
from auspex_v2.instruments.client import CoordinatorClient

with CoordinatorClient() as coordinator:
    coordinator.register("source", "simulated_source", "SIM::SOURCE")
    with coordinator.live("source") as source:
        source.set("frequency", 5e9)
        print(source.get("frequency"))
        print(source.call("identify"))

    coordinator.register("vna", "simulated_vna", "SIM::VNA")
    with coordinator.live("vna") as vna:
        frequencies, s21 = vna.call("get_trace", measurement="S21")
        print(frequencies.shape, s21.dtype)  # (2000,) complex64

    with coordinator.begin_run("demo-run") as run:
        run.set("frequency", 6e9, name="source")
        print(run.get("frequency", name="source"))
```

A run locks **all registered instruments** against live access. `begin_run` fails without changing anything if another run exists, **any live session is open**, or any registered instrument needs recovery. Only one live session per instrument is allowed; closing a live client's connection releases its live session. A disconnected run client instead leaves an **orphaned lockout**. After operator inspection, `reconcile_run(reason)` releases that lockout; affected instruments remain `UNKNOWN` until individually checked with `recover(name)` (or `recover(name, reason)` to include an optional note in the console log). Killing a worker with `terminate(name, reason)` latches `TERMINATED`. Neither termination nor recovery guarantees that physical hardware is safe. `status()` shows registered resources, state, lockout and live-session counts.

The service starts a spawned worker on first live access or run command; it constructs and connects the driver there. Workers serialize requests. A driver command failure or timeout conservatively terminates that worker rather than retrying a potentially completed operation. A result-encoding failure **after** a command completed instead reports that its result is unavailable without declaring the worker dead. Server restart preserves registrations but marks previously registered instruments `UNKNOWN` (or retains `TERMINATED`); operator recovery is required. De-registration requires no run, no live sessions, and an `IDLE` instrument. A single-instance lock prevents a second coordinator from claiming the same socket. Protocol version 2 uses local Unix-domain IPC with a JSON-line envelope (64 KiB header/request limit) followed, when needed, by inline binary array payloads (8 MiB aggregate limit and 16 arrays per result); `GET`/`SET`/`CALL` and session/registration commands retain the same control model. Results describe logical values with array dtype, shape, byte count, and `inline` transport, so later shared-buffer references need not change the logical result type. Numeric and complex NumPy arrays are supported, not object arrays or socket-side pickle. Array results are currently copied through the trusted private worker pipe and returned by value over the client socket; this is for VNA-scale traces, not digitizer throughput.

**Limitations to review before real hardware:** recovery currently only re-connects the *simulated* driver, with an optional console-log note; it performs no device-specific readback, emergency shutoff, or operator authentication beyond local socket permissions. Reasons are not yet written to the planned lab notebook. No **real** VNA adapter in the coordinator, shared-memory buffers, parameterized GET/SET commands, driver argument schemas, or experiment-runner process are supported. Device identity currently uses the simulator's resource name; real multi-channel drivers will require reviewed physical-device canonicalization. Direct use of unregistered drivers bypasses this coordinator. The first real-driver adapter and buffer protocol require separate review.

Run the offline integration tests with `python -m unittest discover -s v2/tests -v`.
