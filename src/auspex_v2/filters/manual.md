# Use storage and processing without an Auspex-controlled experiment

These are first-class, independent workflows. Neither requires instrument
registration, ExperimentServer, a compiled experiment, or Qt. Your own Python code
controls acquisition. The existing DataBrowser can inspect the resulting stores
(its current plotting support remains limited to 1D datasets).

## Standalone AuspexStore

```python
from auspex_v2.data import Axis, AuspexStore

# frequency and response are numeric NumPy arrays from any acquisition library.
with AuspexStore("manual_run.auspex", mode="w") as store:
    store.write_run_metadata({"operator": "Alice", "sample": "device_12"})
    store.write_dataset("data", "trace", response,
                        axes=[Axis("frequency", frequency, "Hz")])

with AuspexStore("manual_run.auspex") as store:
    dataset = store.read_dataset("data", "trace")
```

`mode="w"` creates a new folder and refuses to overwrite an existing one.
`mode="a"` adds datasets to an existing store under its exclusive writer lock.
Dataset names cannot be overwritten. This API writes **complete arrays**, not
appendable datasets. It also supports the coordinate/dimension/ragged forms in
[the data guide](../data/data.md). There is no new storage format or mandatory wrapper.

## ManualPipeline: one complete window per submission

The recommended authoring interface is now the reusable [pipeline definition API](definitions.md):
`ManualPipeline(definition, input_schema=schema, store_path=...)` or
`ManualPipeline(compiled_pipeline, store_path=...)`. Nodes, connections and sinks
come from the definition. Startup snapshots it; subsequent definition edits do not
change a running pipeline. Runtime buffers remain at `pipeline.buffers["name"]`.

The explicit low-level recipe form remains available:

```python
from auspex_v2.filters import ManualPipeline, NodeSpec, WriterSpec, BufferSpec

# schema is a PortSchema; arrays and coordinates must match its dtype/rank/units.
with ManualPipeline(
    schema,
    {"pass": NodeSpec("passthrough", schema)},
    [("$input", "out", "pass")],
    writers={"disk": WriterSpec("pass", "out")},
    buffers={"memory": BufferSpec("pass", "out", capacity=16)},
    store_path="processed_run.auspex",
) as pipeline:
    for bias in biases:
        my_source.set_bias(bias)       # Your own code; no ownership implied.
        frequency, response = my_measurement()
        pipeline.submit(response, axes=[Axis("frequency", frequency, "Hz")],
                        sweep={"bias": bias}, sweep_units={"bias": "V"})
    pipeline.end_scan()                # Optional here: normal exit also does this.

results = pipeline.buffers["memory"].drain()
```

With multiple named inputs, declare their schemas on the definition (or compile
with an input-schema mapping) and call `submit(data, input="name", ...)`. Each
input has independent window IDs; `end_scan()` closes all branches. An input may
remain idle and creates no dataset. See [independent inputs](definitions.md#independent-named-inputs).
For a low-level recipe, supply a `{name: PortSchema}` mapping as the first argument
and use producer IDs `$input:name` in edges. `FilterServer.push`, `end_window`, and
`wait_window` likewise accept `input="name"`. A single schema retains `$input` and
the existing unnamed calls.

`submit()` copies input, assigns consecutive window IDs per input, and automatically emits
window end. It returns the window ID, not completion. There is no partial-window
submission, filter-state readback, reconfiguration, or preview result API in this
wrapper. `sweep` also accepts a tuple of `SweepPosition` objects. No implicit dtype
conversion, interpolation, or coordinate fabrication occurs.

The first submitted scan is `sweep_000`; after `end_scan()` the next submission
starts `sweep_001`, etc. `end_scan()` waits for all required endpoints and writer
publication. Normal context exit ends an active scan, finishes required processing,
and closes workers/shared memory. Exceptional exit cancels unpublished work;
already published datasets remain. Missing averaging contributions cause failure,
not a successful partial average. An empty manual context finishes without a scan.

Only writers, only buffers, or both are supported. A store path is required exactly
when writer sinks are configured. Multiple writers still share one store owner.
Filter graph options such as `backpressure`, `slots`, and memory budgets pass
through to FilterServer. `timeout=10` bounds completion waits, **not** capacity
waits during submission. By default, a required buffer blocks when full, so a
manual caller can consume concurrently. `buffer_full="fail"` instead fails clearly
at capacity, without requiring a consumer; this is the offline unattended-run
policy. It is independent of graph-edge `backpressure=`. With server-managed runs,
use `existing_store=True` and an already initialized `.auspex` folder to let the
store owner append datasets; `run_metadata` must then be written by the server,
not by the filter owner. Ordinary manual runs still create a new folder. An exception/timeout during finalization fails and cleans
up; it does not imply publication success.

## Explicit bounded buffer sinks

`BufferSpec(node, port, capacity=16)` identifies a required endpoint. In the
low-level API use `FilterServer(..., buffer_sinks={...})` and access
`graph.buffer_sinks[name]`. The manual wrapper uses `buffers={...}` and
`pipeline.buffers[name]`. Existing raw ResultSpec behavior is unchanged.

A buffer retains FIFO **completed output batches**, including axes, window/scan IDs,
sweep positions, counts, and configuration revision. It does not stack different
shapes into a fabricated rectangular array. A batch becomes visible only after its
ordered window-end marker. One output batch per output window is supported; an
averager's final output may skip windows. Partial-average ports are rejected.

- `receive(timeout=10)`: consume the oldest completed batch, waiting if necessary.
  An empty completed buffer raises `EOFError`; a live empty buffer times out with
  `TimeoutError`. `timeout=None` waits indefinitely.
- `drain()`: consume all batches currently retained; does not wait for future data.
- `view()`: copy the currently retained batches without consuming.
- `clear()`: explicitly discard retained batches.

These four methods are the public data interface. Queue insertion/nonblocking
removal methods are private to FilterServer; user submissions go through
`ManualPipeline.submit()`, not directly into a buffer. `full()`/`empty()` are
read-only occupancy checks, not access to filter state.

Returned data is independent of input/shared memory. Mutating a view does not
change retained data. Consuming transfers ownership of the buffer's independent
copies. Data remains available after successful graph closure; completed batches
also remain inspectable after failure. Once those are consumed, `receive()` reports
the failed graph instead of waiting. None of this exposes filter internal state.

### Capacity and warnings

Capacity is in completed batches. The graph budget includes retained array capacity
plus one staging batch per buffer for window-completion validation. As elsewhere,
this is an array-byte budget, not a process-RSS guarantee. Calling `view()` creates
caller-owned copies outside the graph budget.

A `BufferFullWarning` is emitted when a buffer reaches capacity, once per full
episode (normal Python warning filters apply). `view()` does **not** free capacity.
With `backpressure=True`, further required output waits; with `False`, exhaustion
fails the graph. No required data is silently dropped.

**Do not submit an arbitrarily long scan and only drain afterward.** Provision
capacity for all retained outputs or consume concurrently. A full sink can block
`submit()`, `end_scan()`, or context exit. Warnings identify the buffer and advise
`receive()`/`drain()`; they are not automatic draining or a deadlock cure. Use
`cancel()` from another thread if an acquisition script is blocked on capacity.

Completed batches can be fed to a subsequent manual pipeline with
`other.submit(batch.data, axes=batch.axes, sweep=batch.sweep)`, analyzed directly in
a notebook, or saved using `store.write_dataset(..., batch.data, axes=batch.axes)`.
Separate downstream pipelines assign their own window/scan identifiers.

## Executable example

From the repository root, with V2 installed:

```sh
python examples/manual_data.py /path/to/new-example-output
```

The guarded `main()` is important for spawned processing workers. Ordinary scripts
must use `if __name__ == "__main__":`; notebook users should only use the installed,
importable built-in worker definitions, not notebook-defined worker classes.
The example creates standalone and averaged synthetic stores, with no hardware.
