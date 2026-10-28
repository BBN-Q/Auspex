# Filters and pipelines

A pipeline transforms complete acquisition windows and delivers required results
to writers or bounded buffers. Filters do not control instruments. A pipeline
specifies data flow; the experiment specifies when acquisitions occur.

## Definitions are inert

Import authoring objects from `auspex_v2.filters.definitions`. Define explicit
inputs, named nodes, and `(producer_port, consumer_port)` edges. Compilation infers
schemas where possible and creates a detached recipe; defining a graph starts no
workers and opens no data files.

Simple transforms use `.sink` and `.source`. Writers and buffers have `.sink` only.
ComplexAverage has distinct `.final_average` and `.partial_average` ports. One
producer can fan out to several sinks; cycles and ambiguous inputs are rejected.

The [annotated experiment](../annotated-experiment.md#3-attach-a-required-writer)
uses a direct writer. Existing examples also show passthrough and averaging graphs.
The same definition API serves experiments and manual acquisition.

## Windows, groups, and scans

- A **window** is one complete submitted acquisition array with coordinates.
- A **group** belongs to a filter, such as five windows forming one average.
- A **scan** contains an experiment's sweep traversal and controls dataset assembly.

The runner owns windows and scan boundaries. An averager owns its groups. Missing
required contributions fail finalization; scan end does not manufacture a partial
average. Complex averaging requires matching actual grids and sweep positions.
There is no silent interpolation across changed coordinates.

A submission is not a completion acknowledgement. A point can finish while data
is still processing. Required writer publication must complete before a run can
be reported successful.

## Capacity and failure

The per-run FilterServer supervises spawned workers and bounded shared-memory
edges. It is not a third persistent service to start manually. Required paths
never silently drop data: capacity either blocks upstream acquisition or fails the
graph according to policy. Preview paths may be lossy and are not substitutes for
required data.

Writer defaults are **512 MiB and 16,384 completed output rows per scan**, with
**2 GiB aggregate declared writer capacity** per pipeline. A row is one emitted
measurement window (for example, one complete VNA trace), not a sample within a
trace or an input window suppressed by averaging. Byte accounting includes each
row's coordinate arrays. These scan limits are ceilings, not upfront allocations;
shared-memory transport slots and their defaults are unchanged. Existing saved
plans/recipes retain their explicit limits; recompile to adopt new defaults.

Writers spool completed windows to temporary disk files during acquisition. At
scan end they inspect one window at a time to determine the actual layout, then
copy bounded numerical chunks into the final file. Publication does not reload
or stack the entire scan in RAM, or enumerate a hypothetical Cartesian grid.
Numerical working memory scales with a few windows and a reference grid; sweep
coordinates, shapes and ragged offsets are per-row metadata. This entails an extra
disk pass. The final file and manifest are published only after writing completes;
staging files are not datasets or an interrupted-run recovery mechanism.

Budgets describe declared array capacity, not total process memory, OS file-cache
usage or guaranteed digitizer throughput. Ordinary save/read APIs still use whole
arrays; this is bounded writer assembly, not continuous acquisition streaming.
A stopped filter does not stop hardware; the runner remains responsible for cleanup.

## Live display

Optional `Plot` sinks declare a current 1D trace or a writer-backed growing 2D
map. Partial averages can feed trace plots; finalized rows feed maps. A separate
[plot client](../utilities/live-plots.md) follows new runs and can reconnect to
complete acquired snapshots. Viewer loss never becomes required backpressure or
a publication acknowledgment. Display data is bounded and explicitly non-authoritative.

## Manual processing

Use `ManualPipeline` when your own code controls acquisition. It accepts one whole
window per `submit()`, then sends ordered boundary events and manages worker/store
cleanup. Normal exit finalizes an active scan; exceptional exit cancels unpublished
work. Already published datasets remain.

Required buffers expose completed batches through `receive()`, `drain()`, `view()`,
and `clear()`. Consume concurrently or provision enough capacity. Waiting for an
entire long scan before draining a full required buffer can block acquisition.
`view()` copies without freeing capacity.

Run `examples/manual_data.py` or `examples/pipeline_definition.py` for independent
processing examples. Ordinary scripts starting spawned workers need a guarded
`if __name__ == "__main__":` entry point.

## Scope

Current execution has one nonempty ingress and built-in passthrough/complex-average
processing. Named writer definitions support one scan per execution. Definitions
can describe multiple inputs, but that does not imply runtime join support.
ExperimentServer currently accepts writer-backed plans, not buffer sinks.

Inspect an explicitly enabled running graph with the
[filter graph viewer](../utilities/filter-graph-viewer.md). It shows topology and
last-acknowledged configuration, not measurement arrays or editable worker state.

## Repository notes

- [Pipeline definitions](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/filters/definitions.md)
- [Manual storage and processing](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/filters/manual.md)
- [Runtime contracts](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/filters/README.md)
