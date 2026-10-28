# Define a pipeline without starting it

`filters.definitions` provides inert, V1-style authoring objects. It is separate
from the executable processors already exported by `filters` (`Passthrough`,
`ComplexAverager`), so existing imports and low-level pipelines keep their meaning.

```python
from auspex_v2.filters import ManualPipeline, PortSchema
from auspex_v2.filters.definitions import (
    PipelineInput, PipelineDefinition, Passthrough, ComplexAverage,
    WriteToFile, DataBuffer,
)

trace = PipelineInput("trace")
forward = Passthrough(name="forward")
disk = WriteToFile(name="disk", group="measurement", dataset="response")
memory = DataBuffer(name="memory", capacity=16)

pipeline = PipelineDefinition(
    inputs=[trace],
    edges=[
        (trace, forward.sink),
        (forward.source, disk.sink),
        (forward.source, memory.sink),
    ],
)

schema = PortSchema("complex128", 1, ("frequency",), ("Hz",), max_bytes=65536)
compiled = pipeline.compile(inputs={"trace": schema})
compiled.save("trace_pipeline.json")

with ManualPipeline(compiled, store_path="measurement.auspex") as running:
    # response and frequency are actual arrays from your own acquisition code.
    running.submit(response, axes=[Axis("frequency", frequency, "Hz")],
                   sweep={"bias": 0.1}, sweep_units={"bias": "V"})

completed_windows = running.buffers["memory"].drain()
```

Import `Axis` from `auspex_v2.data`. A runnable synthetic example is in
[`examples/pipeline_definition.py`](../../../examples/pipeline_definition.py).

## Definitions versus running pipelines

Constructors and `compile()` open no files, start no workers, and perform no
instrument I/O. Filter and sink nodes are discovered from the edge list. Public
inputs must be listed explicitly; an input omitted from the wiring is an error.
All input/node names are explicit, unique within a pipeline, and suitable for
saved identifiers (letters, digits, underscores, dots and hyphens; first character
alphanumeric). Python variable names are not inspected.

Filter configuration uses ordinary attributes:

```python
average = ComplexAverage(name="average", repeats=5)
average.repeats = 10
```

**Compilation freezes a detached snapshot, not the Python definition objects.**
You may continue editing definitions without warnings, but edits only affect a
future compilation/startup. Starting `ManualPipeline(pipeline, input_schema=schema)`
compiles a snapshot automatically. A compiled pipeline already has its schemas;
do not also supply `input_schema`. Running the same definition again creates
independent worker/filter/buffer state.

Sink definitions have no `receive()` or live data. Use the running pipeline's
`buffers["name"].receive()`, `drain()`, `view()`, and `clear()`. Provision enough
capacity or consume concurrently: required buffers are not silently lossy and
can block a run even if another branch is writing to disk.

## Ports and schemas

Simple filters use `.sink` / `.source`. `ComplexAverage` uses `.sink`,
`.final_average`, and `.partial_average`, with no ambiguous `.source` alias.
Writers and buffers have `.sink` only. One output may feed multiple destinations;
each sink has exactly one producer. Cycles and missing required outputs are errors.

Schemas are inferred from the supplied input schemas, and propagated through the
current schema-preserving processors. To declare a fixed *contract*, use
`PipelineInput("trace", schema=...)`; an externally supplied schema must agree.
A schema declares dtype, rank, dimension names/units and a byte budget, **not a
fixed shape or grid**. Actual coordinate arrays accompany each publication.

Validation is deliberately modest. Construction checks wiring. Compilation checks
configuration, schemas already known to be incompatible, required endpoints and
writer collisions, reusing local processor checks. Filters validate actual data
at execution; there is no exhaustive simulation of possible future traces.

`ComplexAverage` averages a positive number of complete **1D complex windows**
with matching grids and sweep positions. It is not V1's arbitrary axis-reduction
averager. `.partial_average` is exposed as a named port, but the current required
writer/buffer sinks cannot consume it. A `Plot(name="current_trace")` sink can
consume it as an optional current-trace display. `Plot(name="bias_map", mode="map")`
consumes finalized rows alongside a writer on the same output. Plots are inert
compiled declarations, never GUI objects or required results. See the
[live plotting guide](../../../docs/utilities/live-plots.md) for budgets, client
behavior and a real simulated-pipeline visual demo.

The input contract permits changing trace lengths and nonuniform grids. The
current runner handles rectangular NumPy windows with one coordinate vector per
dimension; it does **not** implement arbitrary multidimensional ragged arrays.
Changing-length 1D scans can use the existing ragged storage support. A filter
may impose stricter requirements, such as matching grids for averaging.

## Named storage

The running pipeline supplies the `.auspex` directory. Writers choose explicit
`group` and `dataset` names within the shared store; defaults are `data/response`.
All writers share one manifest owner. `max_scan_bytes` and `max_scan_windows`
bound each writer's scan (defaults: 512 MiB and 16,384 completed output rows).
One row is one emitted measurement window, such as a complete trace, not an
individual trace sample. Byte accounting includes each row's coordinates. The
default aggregate `store_scan_budget_bytes` is 2 GiB, allowing four default-sized
writers. These are capacity ceilings, not upfront scan allocations or RSS bounds;
publication inspects staged windows then copies bounded payload chunks, without
loading or stacking the full scan. Numerical working memory scales with window
size plus per-row layout metadata, not the Cartesian product of possible settings.
Transport defaults, public save/read APIs and the dataset format are unchanged.
Existing saved definitions/plans retain their explicit limits; recompile to adopt
new defaults.

For this first authoring slice, named writer definitions support **one scan per
execution**. `end_scan()` publishes the exact named dataset; a subsequent
`submit()` is rejected rather than silently suffixing or overwriting it. An empty
run creates no dataset. Buffer-only pipelines can still use multiple scans.
The older explicit `WriterSpec(prefix=...)` API retains its per-scan naming and
multiple-scan behavior. A broader explicit multiple-scan naming policy is deferred.

## Independent named inputs

Multiple acquisition inputs may feed separate branches in one definition. Each
filter still has exactly one input: this is routing, not a numerical join.
Inputs have independent schemas, consecutive window IDs and averaging groups.
For example, a scalar temperature needs no artificial measurement axis:

```python
trace = PipelineInput("trace", PortSchema(
    "complex128", 1, ("frequency",), ("Hz",), max_bytes=65536,
    value_name="s21"))
temperature = PipelineInput("temperature", PortSchema(
    "float64", 0, (), (), max_bytes=8, value_name="temperature", value_unit="K"))
pipeline = PipelineDefinition(inputs=[trace, temperature], edges=[
    (trace, WriteToFile(name="vna", group="vna", dataset="s21").sink),
    (temperature, WriteToFile(name="thermometer", group="environment",
                              dataset="temperature").sink),
])

with ManualPipeline(pipeline, store_path="measurement.auspex") as running:
    running.submit(response, input="trace", axes=[Axis("frequency", frequency, "Hz")],
                   sweep={"bias": 0.1})
    running.submit(np.array(295., dtype="float64"), input="temperature", sweep={"bias": 0.1})
```

Here `np` is NumPy and `Axis` comes from `auspex_v2.data`; response/frequency are
actual acquisition arrays. Declare each input schema or compile with a mapping
via `pipeline.compile(inputs={"trace": trace_schema, "temperature": temperature_schema})`.
The `input_schema=` convenience argument remains single-input only.

`submit(..., input="name")` selects one input; omitting the name is allowed only
for a single-input pipeline. Publications need not alternate or have equal counts.
Sweep coordinates associate datasets, but do not guarantee unique pairing at
repeated positions. Scan/run boundaries close all branches, including idle ones;
no output rows means no dataset for that writer. Required branch failures fail
the graph. Success waits for all required branches and writer acknowledgments.
One physical owner writes distinct datasets to the shared store; their publication
is not an atomic transaction. Already published datasets survive later failures.

## Serialization and empty sources

A failed definition construction never claims any input: all inputs are checked
before ownership is assigned. Each `PipelineInput` still belongs to only one
`PipelineDefinition`; create fresh inputs to build a different topology.

`PipelineDefinition.save/load` persists a reusable definition, including optional
input schemas. `CompiledPipeline.save/load` persists resolved schemas and empty
input bindings. Loading a compiled recipe revalidates it. Documents contain only
JSON, built-in kind identifiers, node configurations and named port connections;
no pickles, Python source, module lookup, or custom filter registration is used.
The `document` properties return independent copies.

A declared input may receive no publications during execution. This differs from
binding it to an explicit `EmptySource`, whose execution remains deferred:

```python
from auspex_v2.filters.definitions import EmptySource

compiled = pipeline.compile(inputs={"trace": schema, "reference": EmptySource()})
```

Here `reference` must be a declared, wired input with an explicit schema, since an
empty source cannot infer one. An empty source means no data windows—not zeros or
an empty array. Its future execution contract includes normal scan/run completion.

**Execution accepts multiple inputs with nonempty bindings.** Explicit empty
bindings can be serialized/compiled, but runtime rejects them before starting
workers or creating storage. Joins and explicit empty-source execution are deferred,
not emulated by dropping branches.

Experiment `OutputConnector`, explicit `seq.publish`, external `connect`, and
internal `setup_pipeline()` now consume these same definitions; see the
[experiment guide](../experiment/experiment.md). Inputs belong to one definition so an
experiment connection can identify the whole attached graph. Reuse that definition
across executions, or create fresh input objects for a different definition.
Saved compiled pipeline recipes reject unknown fields at every schema, input,
node, configuration, and definition level; incompatible fields are not silently
dropped on load. The pipeline-definition format's version 1 is independent of
the experiment-plan format's version 6.

The experiment's legacy opaque `set_graph()` is not executable pipeline wiring.
ExperimentServer executes writer-backed plans, including optional live plot
endpoints; see the [server guide](../experiment/server.md).
