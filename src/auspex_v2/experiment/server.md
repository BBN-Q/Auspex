# ExperimentServer: writer-only laboratory slice

ExperimentServer is an explicitly started local service. It admits **one active
writer-only run, no queue**, and executes a saved version-6 plan in a separate
runner process. A client can disconnect or exit without cancelling its run. The
runner accepts every driver in the instrument server's explicit registry when its
fixed KIND matches the experiment role. Real-hardware use is trusted, attended
laboratory operation; offline tests do not constitute model-specific hardware validation.

## Start and submit

Start both services together on the acquisition host:

```sh
python utils/launch_auspex.py
# Optional InstrumentServer operator GUI:
python utils/launch_auspex.py --gui
# Optional full-screen terminal console (install auspex-v2[tui]):
python utils/launch_auspex.py --tui
```

The foreground launcher waits for InstrumentServer readiness before starting
ExperimentServer, and prints both endpoints. By default both sockets and two
best-effort diagnostic files (`instruments.log`, `experiment.log`) live under
`$AUSPEX_HOME/.auspex/`, or `$HOME/.auspex/` if `AUSPEX_HOME` is unset (announced
at startup). Logs are truncated when their respective services successfully
start, after the service lock is claimed; custom socket paths do not relocate
logs. Entries carry local `[YYYY-MM-DD HH:MM:SS]` timestamps; interactive
instrument output remains untimestamped. These files are for debugging, not
durable run history or safety auditing.
Experiment execution logs remain in each run folder. `--tui` requires an interactive
terminal and the optional Textual dependency. It captures service output in one
chronological scrolling pane, with differently colored service labels and
Instruments/Experiment prompts (plain labels when `NO_COLOR` is set). Type
`experiment` or `instruments` to change modes. Instrument mode supports `list`,
`connect <id|all>`, `disconnect <id|all>`, `recover <id>`, `terminate <id|all>`,
`clear <id|all>`, and `help`; clear and terminate-all require `y` confirmation.
Experiment mode supports `status`, `runs`, `cancel <run-id>`, and `help`. Ctrl+C
exits the console and requests the launcher's usual cooperative cancellation
and ordered service shutdown. The plain launcher remains the default; when its
stdin is interactive it also offers the same modal commands (service output may
interrupt a prompt). Headless/redirected plain launches do not read commands.
The combined launcher does not expose the standalone instrument `shutdown`
command: use Ctrl+C for ordered shutdown. Use `--instrument-socket PATH` and
`--experiment-socket PATH` for custom endpoints. It refuses existing services,
never uses `--clean`, and preserves saved registrations. Individual service
scripts remain available.

First Ctrl+C requests cooperative cancellation of the active run and waits for
teardown and durable completion, then stops ExperimentServer before InstrumentServer.
A second Ctrl+C stops supervision without waiting for cooperative completion; an
active lease may need reconciliation. SIGTERM uses non-interactive ordered shutdown
and warns about the active run. If status/cancellation is unavailable, the launcher
reports that it cannot confirm completion before stopping supervision.
**Neither cancellation nor service shutdown guarantees hardware outputs are off.**
Recorded Python calculations execute synchronously in the runner. Cancellation is
checked before/after calls, not inside them: a hung function blocks cooperative
cancellation and teardown. These calls have no separate worker isolation. Preflight
imports recorded dependencies before allocating a run folder/acquiring instruments;
imports and functions are trusted Python, not sandboxed. See [recorded calculations](experiment.md#recorded-python-calculations).
If either service exits unexpectedly, the launcher stops the other and returns a
failure status. Unresponsive service shutdown is force-killed after 25 seconds
with an operator warning. Closing the optional GUI does not stop the services.

Startup includes a read-only inventory/ownership summary: no hardware queries,
automatic connections, or recovery. Inspect the same information at any time:

```sh
python utils/auspex_status.py
python utils/auspex_status.py --json
```

The utility accepts the same socket options, checks each service independently
with bounded waits, and reports unavailable services, recovery blockers, active
run labels/progress, and degraded status. Exit status is nonzero if either service
is unavailable; ownership and recovery blockers remain visible in the snapshot.

ExperimentServer prints its banner and concise `[ExperimentServer]` run lifecycle messages:
accepted (ID/folder), running, cancellation requested, and terminal state/reason.
Persistence failures and unexpected runner exits are prominent. Detailed command
traces remain in `execution.log`; terminal output is diagnostic, not authoritative.
Both services prefix their activity messages for a shared terminal. Human-facing
run identities use the allocated folder stem and seven ID characters, for example
`measurement_003 [1abc4de]`, without automatic full-path/browser-command lines.
Full IDs and paths remain in saved status and programmatic responses; use `run.path`
when opening DataBrowser. A failed run may have no published datasets.

### Point progress

A completed point is a completed execution of `sequence()`, not an instrument
command, VNA internal sweep, publication, or saved dataset. Nested sweep totals
are the product of axis lengths; joint axes count their shared points once.
No sweeps means one sequence execution. Internal conditions/loops/acquisitions do
not inflate the point count. Early termination, failure, and cancellation retain
the actual completed count; the bar is never forced to 100%. Unknown totals use
count-only display. Version 5 introduced explicit point boundaries; current plans
use version 6, which also records Python computation dependencies. Recompile older plans.

`run.status()['progress']` contains `phase`, `completed`, `total` (or null),
`coordinates`, `coordinate_units`, `windows_submitted`, and `updated_at`.
Phases distinguish starting, connecting, setup, acquisition, teardown, and processing
drain. Progress is a live in-memory snapshot; the latest snapshot is also included
in durable terminal status. An old timestamp means no progress update, not proof of
a hang. Point completion does **not** mean teardown or data publication has finished.

Terminal updates are throttled to about two seconds, plus phase changes. A standalone
interactive server updates a bar in place; redirected output and the launcher's
shared terminal use complete bar lines so instrument messages do not collide with
an unfinished line. Detailed command logs remain unchanged.

Alternatively, start services separately (separate terminals):

```sh
python utils/instrument_server.py
python utils/experiment_server.py
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
python examples/server_vna_sweep.py /absolute/existing/data
```

Or submit a saved plan directly:

```python
from auspex_v2.experiment import CompiledPlan, ExperimentServerClient
client = ExperimentServerClient()
run = client.submit(CompiledPlan.load("measurement.json"),
                    name="measurement", dir="/absolute/existing/data", date=True)
print(run)  # measurement_000 [1abc4de]
```

`dir` must be an absolute, existing server-host directory. With `date=True`, the
server inserts a `YYYY-MM-DD` directory using its local date at allocation time.
The default is `date=False`. Numbering starts at `000` and advances beyond existing
numbers within the chosen directory: `measurement_000.auspex`, etc. There is no
`plan.run()` API: plans describe execution; clients submit it.

Reconnect from another process:

```python
client = ExperimentServerClient()
client.summary()  # Readable service/active/known run state and progress, no path fields
run = client.run("the-full-or-unambiguous-short-run-id")
status = run.status()  # Unchanged programmatic metadata
print(run.log())                 # Last 64 KiB; complete log remains in the folder.
print(run.plan().explain())
print(run.wait(timeout=60))
# run.cancel()                  # Cooperative, not an emergency stop.
# run.cleanup()                 # Terminal catalog entry only; never deletes files.
```

`summary()` prints plain text and returns None; failed service queries print an
unavailable diagnostic. `status()` and `list_runs()` retain raw dictionaries and
full IDs/paths. Short prefixes resolve against the current catalog for status,
plan, log, cancel and cleanup; missing/ambiguous prefixes reject before action.
`client.run(prefix)` returns a handle with its canonical full ID. Display shortening
is not identity truncation or old-folder discovery.

Read published data with `AuspexStore(run.path)` or DataBrowser. Published datasets
are visible before terminal status if a scan has already completed. The default
scan encloses the entire fixed sweep: this first demonstration is **not incremental
publication**. Temporary-buffer transfer, auto-cleanup, and incremental acquisition
are deferred; plans containing buffer sinks are rejected before any folder write.

Server-backed runs allow 60 seconds per scan-publication wait and for final graph
drainage. This is independent of the client's `run.wait(timeout=...)`, which only
limits how long that client waits. Standalone `ManualPipeline` retains its default
10-second completion timeout.

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
`real_gs200_vna_sweep.py` under `examples/`.
