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
[`v2/examples/pipeline_definition.py`](../../../examples/pipeline_definition.py).

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
writer/buffer sinks cannot consume it; compilation reports that it needs a preview
sink. Preview definitions are not part of this slice.

The input contract permits changing trace lengths and nonuniform grids. The
current runner handles rectangular NumPy windows with one coordinate vector per
dimension; it does **not** implement arbitrary multidimensional ragged arrays.
Changing-length 1D scans can use the existing ragged storage support. A filter
may impose stricter requirements, such as matching grids for averaging.

## Named storage

The running pipeline supplies the `.auspex` directory. Writers choose explicit
`group` and `dataset` names within the shared store; defaults are `data/response`.
All writers share one manifest owner. `max_scan_bytes` and `max_scan_windows`
bound each writer's assembled scan (defaults: 64 MiB and 8192 windows).

For this first authoring slice, named writer definitions support **one scan per
execution**. `end_scan()` publishes the exact named dataset; a subsequent
`submit()` is rejected rather than silently suffixing or overwriting it. An empty
run creates no dataset. Buffer-only pipelines can still use multiple scans.
The older explicit `WriterSpec(prefix=...)` API retains its per-scan naming and
multiple-scan behavior. A broader explicit multiple-scan naming policy is deferred.

## Serialization and future inputs

A failed definition construction never claims any input: all inputs are checked
before ownership is assigned. Each `PipelineInput` still belongs to only one
`PipelineDefinition`; create fresh inputs to build a different topology.

`PipelineDefinition.save/load` persists a reusable definition, including optional
input schemas. `CompiledPipeline.save/load` persists resolved schemas and empty
input bindings. Loading a compiled recipe revalidates it. Documents contain only
JSON, built-in kind identifiers, node configurations and named port connections;
no pickles, Python source, module lookup, or custom filter registration is used.
The `document` properties return independent copies.

Multiple named inputs are expressible now. To deliberately leave one empty:

```python
from auspex_v2.filters.definitions import EmptySource

compiled = pipeline.compile(inputs={"trace": schema, "reference": EmptySource()})
```

Here `reference` must be a declared, wired input with an explicit schema, since an
empty source cannot infer one. An empty source means no data windows—not zeros or
an empty array. Its future execution contract includes normal scan/run completion.

**Current execution accepts exactly one nonempty input.** Multi-input recipes and
empty bindings can be serialized/compiled, but the manual adapter rejects them
before starting workers or creating storage. Joins and empty-source event delivery
are deferred, not emulated by dropping branches.

Experiment `OutputConnector`, explicit `seq.publish`, external `connect`, and
internal `setup_pipeline()` now consume these same definitions; see the
[experiment guide](../experiment/experiment.md). Inputs belong to one definition so an
experiment connection can identify the whole attached graph. Reuse that definition
across executions, or create fresh input objects for a different definition.
Saved compiled pipeline recipes reject unknown fields at every schema, input,
node, configuration, and definition level; incompatible fields are not silently
dropped on load. The pipeline-definition format's version 1 is independent of
the experiment-plan format's version 4.

The experiment's legacy opaque `set_graph()` is not executable pipeline wiring.
Persistent ExperimentServer and live execution remain unimplemented.
