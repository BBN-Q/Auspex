# Run resources and explicit service handoffs

`experiment.resources` contains **building blocks**, not the daemon itself.
The first writer-only [ExperimentServer](server.md) uses these helpers through
`client.submit(plan, name=..., dir=..., date=False)` and terminal `run.cleanup()`.
There is deliberately no `plan.run()` API. Buffer transfer and `run.release_buffers()`
remain deferred.

## Admission preflight

`preflight_run(plan, memory_budget_bytes=..., store_scan_budget_bytes=...)`
checks the single nonempty pipeline, its graph topology and declared transport and
writer staging budgets without starting workers. It also resolves recorded Python
callables before folder/instrument acquisition. Those imports are trusted code and
may themselves write files or have other side effects; preflight is not a sandbox. Server-run
buffers must fit the existing 8 MiB inline array transport; larger digitizer
transports are deferred. The service validates pipeline budgets and claims its
single-run slot **before** allocating a folder. Buffers are currently rejected;
future buffer support must reserve the retained-buffer budget before allocation. Preflight does not guarantee successful worker startup or
instrument connection. The manual/offline workflows are not subject to the
server's inline buffer-transfer limit.

## Folder handoff

`RunFolder.allocate(plan, name="experiment", directory=existing_directory)`
requires a validated version-6 compiled plan. It claims an unused
`experiment_NNN.auspex` folder starting at `000`, advancing beyond the highest
existing number. `AuspexStore(mode="w")` initializes an empty manifest and an
atomic `UNINITIALIZED` status stub. The helper atomically writes `plan.json`,
writes immutable `metadata.json` under the store lock, then replaces the status
with `ACCEPTED` before returning. Initialization failures after the stub attempt
to mark `FAILED`; a crash between these steps leaves `UNINITIALIZED`. Neither
state is a reason to reuse a run number or execute an uncertain run.

For a run with writers, `ManualPipeline(compiled_pipeline, store_path=run.path,
existing_store=True, buffer_full="fail")` demonstrates the handoff: the single
store-owner process opens the existing store with `mode="a"`, owns dataset and
manifest writes, and never rewrites server metadata. An uninitialized/missing
store fails startup. Buffer-only runs still receive an initialized, independently
readable folder; they do not start a store-owner process. Status currently remains
`ACCEPTED` after the offline example because it does not invoke the service.
The service owns status and best-effort execution.log; a separate events file is
deferred. `RunFolder.status()` reads a folder explicitly by path; no
folder scanning or automatic catalog restoration is added.

## Volatile result retention

`RetainedRunBuffers(max_bytes=...)` budgets completed results in **server-owned
memory**. `reserve(run_id, plan)` charges each declared buffer's capacity times
its schema byte bound against the server-wide total before startup; it never
evicts another run. After runner completion, `finish(run_id, batches)` takes
independent snapshots. `view(run_id, name)` returns copies; `release_buffers(run_id)`
frees the reservation and results while keeping the run registered; `cleanup(run_id)`
removes a terminal run and its results. Both release and cleanup reject active
runs. A missing run raises `KeyError`. Released buffers are not recoverable from
the run folder and none are saved to disk. Server restart loses these results.

This is not cross-process transfer: a future ExperimentServer must copy bounded
batches from the runner to its own process before marking a run terminal, using an
explicit IPC transport and failure policy. Writer-only run handles and cancellation
now exist; buffer methods and an auto-cleanup submission option remain deferred.
The public submission boundary is `client.submit(...)`, not a plan method.
Dataset publication, metadata and status files remain independent of volatile
buffers. `run.wait()` must read durable terminal status by folder path if an
auto-cleaned handle is no longer in the live catalog.

Compatibility modules keep prior `auspex_v2.browser`,
`auspex_v2.experiment_fake`, and `auspex_v2.experiment_pipeline` imports usable.
Prefer `auspex_v2.data.browser` and `auspex_v2.experiment` in new code.
