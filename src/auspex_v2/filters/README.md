# Filter pipeline

For reusable **V1-style connector-pair definitions**, start with the [pipeline construction API](definitions.md). `filters.definitions` contains inert inputs, passthrough/complex-average definitions, writers and buffers. Compile/save a detached recipe or pass it to `ManualPipeline`; creating a definition starts nothing. Existing processor imports and low-level recipes below remain supported.

For acquisition controlled by ordinary Python, see [standalone storage and manual pipelines](manual.md). `ManualPipeline` submits complete windows to writers and/or explicit bounded `BufferSpec` sinks. Buffers expose completed outputs with `receive()`, `drain()`, and non-consuming `view()`, retain data after shutdown, and warn when full. The low-level `FilterServer` accepts these via `buffer_sinks=`; existing `ResultSpec` behavior is unchanged.

`core.py` defines the contracts and synchronous **reference executor** for a validated, one-input-per-filter DAG. Filters are ordinary Python objects, not `Process` subclasses. `Passthrough` lives in `core.py` as an example, `ComplexAverager` in `averager.py`, and `Collector` (required result or optional preview) and `DatasetWriter` (one existing `AuspexStore` writer, one completed array per window or group) in `data.py`. `PortSchema` fixes dtype, rank, axis names/units, and maximum total batch bytes **including coordinate arrays**, not the points or length of every window. `WindowState` validates experiment-owned windows in both executors; filters such as the averager derive their own groups from completed windows. Input batches have `group=None`. The averager assigns its own IDs (`group_name` plus sequence), emits `GroupComplete` at its configured repeat count, and rejects an incomplete required group on run end. An error fails the required pipeline. An averager's repeat-count configuration can change only after its active group has completed. A retained writer batch owns copies of its data and coordinates; in-process previews retain only the latest snapshot.

`transport.py` provides reusable per-edge shared-memory slots and ordered, size-limited JSON messages over pipes. Array and coordinate bytes travel in slots, not JSON. `worker.py` is a **spawned, long-lived single-filter worker** using that transport; only built-in filter identifiers, JSON-compatible config, schemas, and startup pipe/shared-memory handles are passed at spawn. The parent owns slot cleanup. Filter objects are constructed in the worker. An output batch is copied before its input slot is released. The convenience `FilterWorker.receive()` copies a result before releasing its slot. Use `FilterWorker.push()`, `end_window(window_id)` and `finish()` to submit checked input; the experiment never sends `GroupEnd`. `configure()` refuses changes until the completed group's required output has delivered both its group-end and window-end markers. The `partial` output is best-effort: it drops new snapshots when full and sends no completion events; each received snapshot includes its count and IDs. The required `final` output never silently drops.

The test suite demonstrates five repeated simulated-VNA sweeps, labeled partial complex averages, an exact-grid final result, persistence in one V2 `.auspex` folder, a changed-grid failure, worker death, and parameter-boundary rules. These are **offline tests**, not a hardware or high-rate throughput benchmark.

## Initial multi-worker Filter Server

`server.py` adds `FilterServer`, a per-run **supervisor thread** that spawns one process per filter, monitors failures, drains terminal outputs, and owns cleanup. It is not a persistent daemon. `NodeSpec` describes a built-in filter and plain JSON-compatible configuration; `ResultSpec` names a terminal port. `WriterSpec` names a required output routed to the single store-owner process. Initially supported workers are `passthrough` and `complex_average`. Graphs have one acquisition ingress, one input per worker, and arbitrary acyclic fan-out; partial outputs can connect only to preview sinks.

`graph_transport.py` reuses the shared-memory array layout with **bounded queues containing only framed JSON metadata**, not NumPy arrays. Both data slots and boundary notifications have finite capacity. Cancellable queue/slot waits replace potentially uninterruptible pipe writes in this runtime. Workers communicate directly along edges; the supervisor copies only terminal results. This is a correctness-oriented implementation, not a high-rate notification benchmark.

```python
from auspex_v2.filters import FilterServer, NodeSpec, ResultSpec

nodes = {
    "pass": NodeSpec("passthrough", schema),
    "average": NodeSpec("complex_average", schema, {"repeats": 5}),
}
edges = [("$input", "out", "pass"), ("pass", "out", "average")]
results = {
    "mean": ResultSpec("average", "final"),
    "plot": ResultSpec("average", "partial", preview=True),
}
with FilterServer(schema, nodes, edges, results, backpressure=True) as graph:
    for batch in five_sweep_batches:  # group=None; windows 0 through 4
        graph.push(batch)
        graph.end_window(batch.window)
    graph.finish()
    mean = graph.receive("mean")
    graph.wait_finished()
```

### Admission, completion, and capacity

- `push()` copies the input before returning. Submission is serialized, but does **not** wait for downstream window completion. Later windows can overlap processing of earlier windows. There is no hidden unbounded submission queue.
- `wait_window(n)` waits for all required terminal paths to process the prefix through window `n`; that does not imply every averager has finished its group or a scan has been published. `wait_scan(scan)` waits for publication acknowledgements and required terminal scan boundaries.
- `receive(name)` consumes an owned result batch. Group IDs are scoped by producing node/result name. Required result buffers are bounded; clients must consume concurrently with acquisition when the result count exceeds capacity. Waiting for run completion without draining a full required buffer can intentionally stall progress.
- `finish()` submits run end. `wait_finished()` confirms required terminal drainage, filter run-end validation, and successful store-owner completion when configured. Retained results remain available after successful cleanup.
- `slots` bounds each edge's arrays; its notification capacity is `slots + 2`. `max_windows` bounds the admitted-but-not-completed window count. `result_capacity` bounds batches retained per terminal. `memory_budget_bytes` checks declared shared-memory and retained-array capacity before startup; it is **not** an OS RSS limit and excludes worker internals, metadata, and temporary copies.
- With `backpressure=True`, required capacity waits propagate upstream while supervision stays active. With `False`, required capacity exhaustion fails the graph, including internal edges and retained results. This is fail-on-overflow, not a guarantee that acquisition can run at an arbitrary rate. Preview edges drop new snapshots when full, and terminal preview retention evicts the oldest snapshot.

### Configuration and failures

`configure(node, changes)` holds admission, waits for the admitted window prefix to complete, then requests and acknowledges a filter-local revision. Input batches retain the acquisition revision (initially zero); each worker labels its emitted batches with its own revision. This is not a full configuration-provenance history. A target averager rejects updates during its active group, while an unrelated branch can retain its partial group unchanged. Schema changes and topology changes during a run are unsupported. A configuration-ack timeout fails the graph because the update outcome is unknown; an explicit rejection leaves it usable.

`failed` is a thread event for the future runner/coordinator. `check()`, submission, and result/completion waits raise the latched error. Required errors stop admission, wake blocked producers, and trigger cooperative shutdown followed by termination if necessary. `cancel()` requests this failure path; `close()` joins cleanup and cancels an unfinished run. Neither cancellation nor a timeout counts as successful completion. The coordinator remains responsible for stopping the experiment and hardware; killing a filter does not stop an instrument.

### Scan boundaries and graph-backed persistence

`Batch(scan="sweep_000", sweep=(SweepPosition("bias", 0.2, "V"), ...))` carries the runner-supplied actual sweep position. `PortSchema` declares the value name/unit and per-batch measurement-axis meaning. `FilterServer(input_name="vna_s21", ...)` names the acquisition ingress; saved datasets identify both that input and their producing output port. After its final `end_window`, the runner calls `end_scan("sweep_000")`, then may `wait_scan("sweep_000")`. The averager rejects unfinished groups at scan end and may not average across sweep positions or scans. `finish()` requires an explicit scan end for any non-default scan. The runner cannot switch scans before closing the previous one. `scan` must be a safe file-name component.

For example, use `FilterServer(schema, {"pass": NodeSpec("passthrough", schema)}, [("$input", "out", "pass")], {}, input_name="vna_s21", writers={"raw": WriterSpec("pass", "out", "data")}, store_path="experiment_001.auspex")`; submit `Batch(..., scan="sweep_000", sweep=(SweepPosition("bias", 0.2, "V"),))`, call `end_window(window)` for each trace, then `end_scan("sweep_000")`, `wait_scan("sweep_000")`, `finish()` and `wait_finished()`. `FilterServer` creates one spawned store owner that opens a **new** `.auspex` folder and multiplexes required writer edges. Each writer stages at most one batch per window; a group-producing output can skip windows. Completed batches are spooled to hidden temporary directories inside the run folder, subject to each writer's `max_scan_bytes` and `max_scan_windows` and an aggregate `store_scan_budget_bytes`, and assembled at `scan_end`. A killed process may leave unlisted staging files; do not interpret them as published data. `WriterSpec.prefix + scan` is the dataset name within its group; collisions fail rather than overwrite. Optional `run_metadata={...}` supplies a JSON-compatible settings snapshot written to `metadata.json` before graph startup; no settings or changes are automatically captured. The owner publishes and acknowledges each dataset before `wait_scan` returns; errors or owner death fail the graph. `graph.published` holds transient publication acknowledgements, not a durable run log. The store owner handles identical, dependent, and variable-length rank-one frequency coordinates (see [data format](../data.md)). Complete Cartesian sweeps acquired in product order become named sweep dimensions; arbitrary/repeated sweep paths retain an outer `point` dimension with numeric per-point sweep coordinates instead of falsely claiming a grid. These output datasets can be multidimensional, but filters still process one rectangular batch at a time; a graph worker cannot yet consume a dependent/ragged *assembled scan*. These budgets cover declared array bytes/window counts, not staging-file overhead, temporary copies, or process RSS. The owner assembles the scan in memory at publication, so this is **not** a digitizer throughput or out-of-core benchmark. A scan with no output batches for a required writer fails.

### Read-only graph viewer

Set `inspect_socket` on `FilterServer` to enable a local snapshot endpoint and launch `utils/filter_graph_viewer.py SOCKET`. The separate optional PySide6 window renders the DAG and lets you inspect last-acknowledged configurations, connections and schemas. Refresh is manual; no worker commands, measurement values, automatic status polling, or topology/parameter editing. See the [inspection guide](inspection.md) for socket permissions, launch instructions and limitations.

**Still deferred:** multiple acquisition inputs, multi-input joins, custom filter admission, plot subscriptions/reconnect, automatic run metadata/events and settings changes, full scan-as-filter payloads, and digitizer throughput/overflow measurements. The reference executor and older single-worker harness remain separate test tools.
