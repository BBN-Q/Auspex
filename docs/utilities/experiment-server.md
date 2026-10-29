# Experiment server

`experiment_server.py` starts only ExperimentServer. InstrumentServer must already
be available. Use separate terminals when not using the combined launcher.

```sh
python utils/instrument_server.py
# In another terminal:
python utils/experiment_server.py
```

Options are `--socket PATH`, `--instrument-socket PATH`, and `--no-splash`.
The service accepts one active writer-backed run, with no queue. It does not
register instruments, repair inventory, or open hardware merely by starting.

## Submit and reconnect

```python
from auspex_v2.experiment import CompiledPlan, ExperimentServerClient

client = ExperimentServerClient()
run = client.submit(CompiledPlan.load("measurement.plan.json"),
                    name="measurement", dir="/absolute/existing/data", date=False)
print(run)                  # measurement_000 [1abc4de]
```

The directory belongs to the server host and must already exist. The server
allocates a new numbered `.auspex` run, never overwriting an existing folder.
`date=True` inserts a server-local date directory. The returned handle is not a
completion acknowledgement.

```python
client.summary()            # Readable server state, known runs, progress/diagnostics
run = client.run("1abc4de")  # Use your actual unambiguous short ID (or the full ID)
status = run.status()        # Unchanged programmatic metadata
print(run.log())             # Last 64 KiB; complete log remains in the folder.
print(run.plan().explain())
# run.cancel()               # Cooperative request, not an emergency stop.
# run.wait(timeout=60)       # Timeout does not cancel execution.
# run.cleanup()             # Forget a terminal catalog entry, not its files.
```

The client can exit while the runner continues. The catalog is volatile: restart
does not scan run folders or restore handles automatically. Known saved folders
remain inspectable directly.

## Readable identities and summaries

Notebook handles, terminal/TUI output, and GUI run labels use the allocated folder
stem plus seven ID characters, such as `measurement_003 [1abc4de]`. These are display
abbreviations, not new identifiers. `run.run_id`, `status()`, `list_runs()`, saved
status, and JSON inspection retain full IDs; `run.path` retains the full folder path.
Terminal announcements omit automatic full-path/browser-command lines.

`client.summary()` prints service availability, the active run, known runs, state,
point progress, and failure/cancellation diagnostics without path fields. It returns
None and needs no GUI or pandas. CLI/TUI `status` and `runs` use the same readable
presentation. Diagnostic messages can still contain paths reported by failing I/O.

`client.run(prefix)` and run actions accept any unambiguous prefix in the current
server catalog. For example, terminal `cancel 1abc4de` requests cooperative
cancellation. Unknown or ambiguous prefixes fail without taking action; use a
longer prefix or a full ID if needed. Returned handles always retain the canonical
full ID. Lookup does not search old folders after catalog cleanup or service restart.

## Interpret completion

States progress from ACCEPTED to RUNNING, then SUCCEEDED, FAILED, or CANCELLED.
A startup error can fail before setup begins. Successful completion includes
required processing/storage finalization. Status-write failures can leave durable
status unresolved and disable new admission; do not infer success from old status.

Once setup begins, ordinary outcomes attempt teardown. Failure there can orphan
ownership. Supervision loss stops further instrument commands, including teardown.
Recorded Python calls execute synchronously and may block cooperative cancellation.
Restart never silently resumes or retries commands.

See [experiments](../components/experiments.md) and the retained
[service contract](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/experiment/server.md).
