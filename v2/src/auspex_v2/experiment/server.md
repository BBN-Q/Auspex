# ExperimentServer: writer-only laboratory slice

ExperimentServer is an explicitly started local service. It admits **one active
writer-only run, no queue**, and executes a saved version-4 plan in a separate
runner process. A client can disconnect or exit without cancelling its run. The
runner accepts every driver in the instrument server's explicit registry when its
fixed KIND matches the experiment role. Real-hardware use is trusted, attended
laboratory operation; offline tests do not constitute model-specific hardware validation.

## Start and submit

Start both services on the acquisition host (separate terminals):

```sh
python v2/utils/instrument_server.py
python v2/utils/experiment_server.py
```

Both accept `--socket`; ExperimentServer also accepts `--instrument-socket`.
Unix socket permissions restrict local access; this is a trusted-local prototype,
not a sandbox or an authenticated remote service. A lock prevents two services
from owning the same experiment socket.

Register instruments explicitly, once, through the instrument server:

```python
from auspex_v2.instruments.server import InstrumentServerClient
with InstrumentServerClient() as instruments:
    instruments.register("lab_source", "AgilentN5183A", "TCPIP0::host::5025::SOCKET")
    instruments.register("lab_vna", "AgilentE8363C", "TCPIP0::host::INSTR",
                         command_timeout=None, io_timeout=30)
```

See the [instrument server guide](../instruments/server/instrument_server.md) for
the canonical registry, PyVISA-py backend and timeout semantics.

The runner connects required disconnected instruments under its lease. It never
implicitly registers or recovers them. Already-connected instruments are reused.

```sh
# This client submits and exits; the independent services continue executing.
python v2/examples/server_vna_sweep.py /absolute/existing/data
```

Or submit a saved plan directly:

```python
from auspex_v2.experiment import CompiledPlan, ExperimentServerClient
client = ExperimentServerClient()
run = client.submit(CompiledPlan.load("measurement.json"),
                    name="measurement", dir="/absolute/existing/data", date=True)
print(run.run_id, run.path)
```

`dir` must be an absolute, existing server-host directory. With `date=True`, the
server inserts a `YYYY-MM-DD` directory using its local date at allocation time.
The default is `date=False`. Numbering starts at `000` and advances beyond existing
numbers within the chosen directory: `measurement_000.auspex`, etc. There is no
`plan.run()` API: plans describe execution; clients submit it.

Reconnect from another process:

```python
client = ExperimentServerClient()
print(client.list_runs())
run = client.run("the-run-id-printed-at-submission")
print(run.status())
print(run.log())                 # Last 64 KiB; complete log remains in the folder.
print(run.plan().explain())
print(run.wait(timeout=60))
# run.cancel()                  # Cooperative, not an emergency stop.
# run.cleanup()                 # Terminal catalog entry only; never deletes files.
```

Read published data with `AuspexStore(run.path)` or DataBrowser. Published datasets
are visible before terminal status if a scan has already completed. The default
scan encloses the entire fixed sweep: this first demonstration is **not incremental
publication**. Temporary-buffer transfer, auto-cleanup, and incremental acquisition
are deferred; plans containing buffer sinks are rejected before any folder write.

## Ownership and failure rules

- The service validates plans, topology, budgets and the InstrumentServer domain
  snapshot before allocating a folder. Another/orphaned run, live sessions,
  in-flight commands, recovery-required instruments, invalid plans and unsupported
  buffers do not create run folders. A date subdirectory is created only after preflight.
  After folder claim, operational failures leave a record rather than reuse a number.
- The runner acquires the instrument lease, rechecks mappings/KINDs, starts the
  pipeline, connects required devices, then asks the service to persist `RUNNING`.
  Setup cannot begin until that status write is acknowledged.
- Status is authoritative: `ACCEPTED` → `RUNNING` → `SUCCEEDED`, `FAILED`, or
  `CANCELLED`. Pre-setup failures may go directly from `ACCEPTED` to terminal.
  A failed status write disables new admission and produces a prominent diagnostic.
  If terminal persistence fails, the diagnostic includes whether the lease was
  released; the old durable status is not presented as a successful completion.
- `execution.log` is best-effort diagnostic text, not an audit trail. Log messages
  have no acknowledgments; disk-log failures do not stop the run. Returned command
  data is not logged. Arguments may be logged and long messages are truncated.
  No separate events file is implemented.
- Once setup begins, normal completion, failure and cancellation attempt teardown.
  Teardown stops at its first error, marks the run `FAILED`, and reports that
  remaining cleanup was skipped and safe state was not established. The runner
  disconnects **without END_RUN**, orphaning the instrument lease for reconciliation.
- Successful teardown releases the instrument lease **before** waiting for final
  filter/storage completion. A later storage failure still makes the run `FAILED`.
  The service remains busy through finalization and terminal-status persistence.
  Pre-setup startup failures release the lease; failed instruments remain quarantined.
  The instrument server independently quarantines driver failures, so a subsequent
  teardown command to that instrument may itself fail.
- Cancellation is cooperative: an in-flight command returns or times out before
  cancellation takes effect. No later measurement command is dispatched. Teardown
  ignores ordinary cancellation. Partial scans are cancelled, not sealed into
  apparently complete datasets. Already published datasets survive failures.
- The runner checks service control and latched filter health between instructions
  and before measurement commands; waits check control in short intervals. Loss of
  service supervision stops further commands **including teardown**, closing the
  instrument connection without END_RUN. An in-flight instrument call cannot be
  undone and loss may not be noticed until it returns/times out. There is no watchdog.
- Runner death similarly orphans its instrument connection. The service reaps its
  process group (filter/store children, not instrument-server workers). Forced-kill
  tests can produce Python resource-tracker warnings while its per-run tracker
  reclaims shared memory/semaphores; no automatic command replay or crash resume exists.

## Restart and scope

The catalog is volatile. Startup does **not** scan folders, restore old entries,
repair stale `RUNNING` statuses, or reconcile orphaned instrument runs. Inspect
known folders explicitly; use the instrument server's explicit operator recovery
and reconciliation. An existing handle's `wait()` can read durable terminal status
from its known folder after catalog cleanup or loss of service. Unknown/stale
nonterminal status times out rather than implying success.

Integration tests exercise a real service, runner, simulator commands, writer
workers, notebook-process exit/reconnect, cancellation, startup failure, runner
death, supervision loss, teardown failure, and status persistence failures. Real
drivers and the three lab examples compile against fake transports without VISA
I/O. Model-specific hardware behavior and throughput must be checked in the lab.
Reviewed examples are `real_vna_sweep.py`, `real_source_vna_sweep.py`, and
`real_gs200_vna_sweep.py` under `v2/examples/`.
