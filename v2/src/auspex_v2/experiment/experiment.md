# Experiment compiler and explicit pipeline publication

`compiler.py` implements trusted Python authoring, a structured instruction
compiler, detached version-5 JSON execution units, `save`/`load`, and `explain()`.
Only version 5 is accepted; older plans must be recompiled from their experiment
authoring code. New compilation requires an attached pipeline. No saved source, pickle, or `eval` is used.

`FakeExecutor` interprets commands and publications without workers or storage.
`PipelineExecutor` in `pipeline.py` is a separate **offline integration
harness**: injected fake command results feed real filter workers/writers, while
WAIT remains virtual. Neither executor connects hardware, owns instrument leases,
implements cancellation/supervision, or starts ExperimentServer. Do not use these
harnesses as live runners. A separate [ExperimentServer](server.md) runs saved
writer-only plans against the explicit simulated/real driver registry under an
instrument lease, with real waits, cooperative cancellation, and reconnectable handles.

## Example

```python
from auspex_v2.experiment import Experiment, InstrumentRef, FloatParameter, OutputConnector
from auspex_v2.experiment.fake import FakeExecutor
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, DataBuffer

class Measurement(Experiment):
    source = InstrumentRef(kind="DC_SOURCE")
    voltage = FloatParameter(default=0.0, unit="V", bounds=(-1, 1))
    settling = FloatParameter(default=0.1, unit="s")
    signal = OutputConnector(dtype="float64")

    def setup_pipeline(self):
        signal = PipelineInput("signal")
        memory = DataBuffer(name="memory", capacity=16)
        PipelineDefinition(inputs=[signal], edges=[(signal, memory.sink)])
        self.connect(self.signal, signal)

    def setup(self, seq):
        seq.call(self.source, "prepare")

    def sweep_begin(self, seq, axis):
        seq.call(self.source, "begin", axis.name)

    def sequence(self, seq):
        value = seq.get(self.source, "temperature")
        with seq.if_(value > 100):
            seq.terminate("FAILED", reason="Temperature limit exceeded")
        result = seq.call(self.source, "measure")
        seq.publish(self.signal, data=result)

    def sweep_end(self, seq, axis):
        seq.call(self.source, "end", axis.name)

    def teardown(self, seq):
        seq.call(self.source, "finish")

exp = Measurement()
exp.bind_instrument("source", "lab_source")
exp.voltage.bind(exp.source, "voltage", wait=exp.settling)
exp.add_sweep(exp.voltage, [-0.5, 0.0, 0.5])
plan = exp.compile()
plan.save("measurement.plan.json")
print(plan.explain())
run = FakeExecutor(lambda op, device, member, args, kwargs: 20.0).run(plan)
assert run.outcome == "SUCCEEDED"
```

Instrument roles use the fixed, case-sensitive `InstrumentKind` vocabulary. An
abstract role supplies `kind=`. Supplying `driver=` infers KIND and enables its
shorthand API; `InstrumentRef().bind(Driver)` is the equivalent class-body form.
If both are supplied they must agree. `kind` and `driver` are read-only after
construction; use `.bind()` before compilation to select another same-KIND
authoring driver. ExperimentServer compares the role KIND to
the registered driver's KIND; unsupported driver members fail at dispatch. The
fake callback is deliberately supplied separately, never saved in the plan.

## Pipeline attachment and publication

An `OutputConnector` declares `dtype`, `dimensions`, `axis_units`, optional value
`unit`, and `max_bytes` (default 64 KiB including coordinates). It does not fix a
shape or coordinate grid. Actual arrays are validated locally against this schema
and downstream filter requirements, without dtype coercion or invented coordinates.

```python
trace = OutputConnector(dtype="complex128", dimensions=("frequency",),
                        axis_units={"frequency": "Hz"})
# Inside sequence(), for a driver returning (frequency, response):
result = self.vna.get_trace()
seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})
```

Record fields (`result["response"]`) and integer selections (`result[1]`) are
supported. Symbolic values cannot be unpacked or iterated by authoring Python.

A publication creates **one complete window**, with current runner-owned sweep
coordinates. Several windows may share a sweep point. It may wait for capacity;
submission is not processing or disk-publication completion. Publication is allowed
only in the main scan (including sweep hooks), not in setup or teardown. Both sides
of symbolic branches are checked, even when a literal condition is false.

For an internal pipeline, define `setup_pipeline(self)` as above. It runs during
compilation on the detached experiment copy, before recording measurement hooks;
it may use configured parameter values. It starts no workers or instrument I/O.
Default `setup_pipeline = None` does nothing. Connections use the same independent
[definition API](filters/definitions.md) as manual pipelines.

External wiring instead uses `exp.connect(exp.trace, pipeline_input)` before
compilation. The target must belong to a `PipelineDefinition`. **Any external
connections replace internal setup entirely**: compilation emits
`PipelineOverrideWarning` and never calls `setup_pipeline()`. Later external
connections add to that external configuration; they do not replace each other.

Every input of each attached pipeline must have exactly one source. An explicit
`EmptySource()` can stand in for an intentionally unused input with a declared
schema. Published outputs must be connected; unused declarations need not publish.
Checks are syntactic, not exhaustive path/reachability analysis. Output fan-out to
several pipelines can compile, but the current offline harness admits only **one
pipeline with one nonempty input**. Multiple inputs/empty execution remain deferred.

Compiled plans contain resolved schemas, named input/output connections, filter
configuration and sink destinations. Later changes to definitions or experiment
configuration cannot alter a saved plan. Schema/connection validation also runs on
load. `Experiment.outputs` and `Experiment.graph` are not authoring APIs:
output schemas derive from declared `OutputConnector`s, and the processing graph
from attached `PipelineDefinition`s. Assigning either legacy attribute or calling
`set_graph()` fails with guidance rather than saving inert configuration.

## Offline pipeline execution

```python
from auspex_v2.experiment.pipeline import PipelineExecutor

run = PipelineExecutor(fake_command).run(plan, store_path="run.auspex")
# Omit store_path for buffer-only pipelines.
print(run.outcome, run.reason)
```

The harness starts the pipeline before executing setup. It attempts teardown once
setup begins, stopping on the first teardown error and preserving any earlier
failure. Only after teardown does it wait for final processing/publication. A
completed scan can still publish if teardown failed; a failed partial scan is
cancelled. Successful early termination closes the admitted prefix, but incomplete
required filter groups fail finalization. Buffers remain accessible on `run.buffers`.

The offline pipeline harness fails on a full required buffer instead of waiting
for an absent consumer; an ordinary `ManualPipeline` still defaults to blocking
so scripts can drain buffers concurrently. The harness stores only window metadata,
not all published payload arrays in its trace; `FakeExecutor` retains payloads by
default for tests. The harness does not promise bounded hardware command time or safe hardware state,
and does not implement leases, crash recovery or durable execution logs. Its
`PROGRAM_FINISHED` trace precedes `PIPELINE_FINISHED`; only the latter and the final
`outcome` include required processing completion. The saved plan is independent of
the injected command callback. Existing import paths `auspex_v2.experiment_fake`
and `auspex_v2.experiment_pipeline` remain compatibility re-exports; new code can
import `FakeExecutor` and `PipelineExecutor` from `auspex_v2.experiment`. For a runnable example that also saves the plan with
data, see `v2/examples/compiled_publication.py`.

## Driver-shaped shorthand

An optional driver class supplies the **offline API description**, without being
constructed or connected:

```python
from auspex_v2.instruments.drivers.simulated import SimulatedVNA

class TraceExperiment(Experiment):
    vna = InstrumentRef(kind="VNA", driver=SimulatedVNA)

    def sequence(self, seq):
        self.vna.frequency_center = 6.1e9           # SET
        center = self.vna.get_frequency_center()   # GET (symbolic result)
        with seq.if_(center > 1e9):
            trace = self.vna.get_trace(measurement="S21")  # declared CALL
        seq.wait(0.5)
```

Properties, generated get/set helpers, managed properties, and declared operations
use the same `InstrumentProxy` as live sessions, backed by a recording session.
`get_trace` remains a CALL because it is declared as an operation, not because of
its name. Property reads also emit GET. Parameterized settings require helpers,
e.g. `self.source.set_channel_frequency(value, channel=2)`.

All proxy call keywords belong to the driver, including arguments named `name`,
`member`, `value`, `args`, `kwargs`, or `wait`. Routing metadata travels separately
from this argument mapping in both live and recording backends. Explicit proxy
`get`/`set`/`call` take their member (and set value) positionally. Use `seq.wait(...)` after shorthand, or retain
`seq.set(self.vna, "frequency_center", center, wait=0.5)`. Proxy CALLs do **not**
emit output data or create acquisition windows: use `seq.publish` explicitly.
`seq.acquire` has been removed; use CALL followed by explicit values/coordinate
selection and `seq.publish`.

Instrument access outside compilation raises an error. Retained proxy helpers are
invalidated when compilation ends, including on authoring errors. `kind` and
`driver` are reserved role metadata names. A role without `driver=` retains the
explicit `seq.*` API but cannot use shorthand. The compiler checks declared member
names, read/write permissions and setting keyword names for roles with driver
metadata, including explicit `seq.*` calls. It does not validate arbitrary operation
signatures or physical limits yet.

Only plain capability descriptions and instructions are saved—not driver classes,
module references, or live proxy objects. Supplying an authoring driver does not
require that exact class at runtime. Admission checks the registered driver's
KIND; InstrumentServer reports an error if a dispatched member is unsupported. Shared description/proxy code lives under `instruments/drivers/`;
the server's previous proxy import remains compatible.

## Configuration, runtime parameters, and instrument settings

These are intentionally different operations:

| Operation | Meaning |
| --- | --- |
| `exp.bias.value = 0.2` before compiling | Configure an initial experiment value |
| `seq.set_parameter(self.bias, 0.2)` inside a hook | Change requested runtime parameter state and apply its binding, if any |
| `self.source.voltage = 0.2` inside a hook | Record an instrument SET, without inferring a parameter update |
| `self.bias.value = 0.2` inside a hook | Error: configuration is frozen while recording |

Replacing a declared parameter (`exp.bias = 0.2`) or instrument role is rejected,
with guidance to `.value` or `bind_instrument()`. Parameter declarations/bindings,
instrument metadata, sweeps, output declarations, pipeline configuration and run metadata are
frozen during recording. Attribute writes are rejected immediately; in-place edits
to nested configuration are checked when a hook returns. Ordinary Python locals
and symbolic temporary attributes remain usable for building the program. These
are authoring-error checks, not a sandbox.

For example:

```python
class BiasMeasurement(Experiment):
    source = InstrumentRef(kind="DC_SOURCE")
    bias = FloatParameter(default=0.2, unit="V", bounds=(-1., 1.))

    def sequence(self, seq):
        seq.set_parameter(self.bias, 0.4)
        seq.call(self.source, "measure")

    def teardown(self, seq):
        seq.set_parameter(self.bias, 0.0)

exp = BiasMeasurement()
exp.bind_instrument("source", "lab_source")
exp.bias.bind(exp.source, "voltage", wait=0.5)
# Add output declarations and attach a pipeline as above before exp.compile().
```

`set_parameter` records `SET_PARAMETER`, then explicit binding SET/WAIT instructions.
It checks the value's type and declared bounds before changing state or issuing
the bound command. Literal errors fail compilation; dynamic values are checked by
the executor. Subsequent symbolic references see the new requested value. An
unbound parameter changes runtime state only. Requested values are **not physical
readbacks**; a failed bound command does not imply the hardware adopted the value.

A parameter without an initial value must be initialized before use; teardown can
initialize an unswept parameter explicitly even when setup failed. Automatic static
binding application skips parameters without initial values. For parameters with
initial values, automatic application still follows setup and reapplies their
**current** runtime values, including the binding wait; it does not restore the
original default if setup used `set_parameter`.

Runtime changes to **swept parameters are rejected**, including in teardown. Use
an explicit instrument SET to turn off a swept bias. Adaptive sweep updates and
separate scheduled/requested/readback coordinate tracking remain future work.
Direct device commands are not inferred into the sweep metadata, so users must
avoid changing measurement conditions behind those coordinates.

A runtime-updated parameter is not a frozen loop count: `seq.repeat(self.count,
max_iterations=10)` evaluates its current value on entry. Maximum iteration bounds
must remain fixed literals or configuration parameters never changed at runtime.
WAIT reads the current value of an initialized unswept parameter, checking finite,
nonnegative duration when executed. Binding delays can use that same mechanism.

## Semantics

- Compilation records from a deep copy of the configured experiment. Changes to
  original values, sweep arrays, bindings, or returned `plan.document` dictionaries
  do not mutate the plan. Its SHA-256 identity covers the canonical executable
  document, including frozen configuration, and does not include a creation time.
- Class declarations are inherited and instantiated separately per experiment,
  including independent parameter declaration metadata (bounds and defaults).
  Set configuration through `exp.voltage.value`; bind settings using `.bind()`.
  NumPy scalars/arrays, including nested configuration values, are normalized to
  plain JSON values before validation. Integer parameters still reject floats.
- `setup` precedes automatic unswept parameter application. The first sweep added
  changes fastest. An axis begin hook surrounds each traversal of that axis,
  followed by its end hook on normal completion. Outer values apply before inner
  values. Every point reapplies its bindings, even for repeated values.
- A swept parameter with no initial value cannot be read in setup or before its
  first point (including its own begin hook). An inner begin hook may read an
  already assigned outer parameter. A reachable end hook may read the last point
  of its nonempty sweep. Teardown must work even if setup failed before sweeping,
  so a swept parameter read in teardown requires a valid initial value.
- Joint paths use `add_sweep((exp.x, exp.y), [[x1, y1], ...], name="path")`.
  Empty sweeps, duplicate parameter ownership, and mismatched widths are rejected.
- No sweeps means one measurement and no axis hooks. The main program carries one
  default `SCAN_BEGIN`/`SCAN_END` pair around the complete traversal. These are
  trace markers in the fake interpreter, not persistence acknowledgements.
- `get`, `set`, and `call` accept runner `wait=`. It lowers to an explicit
  `WAIT` following successful command completion. `seq.wait(seconds)` is also
  available. Durations are finite nonnegative literals or unswept parameters.
  Driver keyword arguments go in `kwargs={...}` so driver `wait` cannot collide.
- Every `publish` submits one complete window with current sweep coordinates.
  No instrument command publishes implicitly.
- Python `if`, `and`, `or`, `not` cannot consume symbolic values. Use `seq.if_`,
  `seq.else_`, `seq.all`, `seq.any`, and `seq.not_`. Both branch bodies are built.
  Comparisons and basic arithmetic produce expression trees; field selection is
  explicit (`result["field"]`). GET/CALL result types are unknown until execution
  in this slice; scalar Boolean conditions are enforced at runtime. Comparisons
  between incompatible known types fail at compile time; numeric int/float
  comparisons and string/string ordering are supported. Arithmetic supports both
  operand orders (`x * 2`, `2 * x`), reflected subtraction/division, and unary `-x`.
- State uses `seq.variable("counter", type=int, initial=0)` and
  `seq.assign(counter, counter + 1)`. Ordinary Python assignment does not record
  runtime assignment. Names must not collide with generated `result_N` names.
- `seq.repeat(count)` freezes its count on entry. Dynamic counts require an explicit
  positive `max_iterations`. `seq.while_(condition, max_iterations=N)` checks before
  each iteration and fails if still true at the limit. Frozen integer parameters
  can specify counts and bounds. Loops remain structured, never unrolled.
- Values produced only in a possibly unexecuted branch/loop cannot be used afterward.
  `TERMINATE` supports `SUCCEEDED` and `FAILED`. Finalization attempts teardown,
  which cannot depend on potentially failed setup/main reads. Teardown loops,
  acquisitions, and termination are disallowed in this initial slice.

## Fake executor and trace

`FakeExecutor(command, max_steps=100000).run(plan)` returns the executor with
`outcome`, `reason`, `trace`, `windows`, and virtual `time`. It logs command
arguments before dispatch, acknowledgment/failure afterward, and control-flow
outcomes. Returned payloads are not logged. WAIT advances virtual time without
sleeping. An instruction/iteration budget also bounds empty loops. Failed commands
skip the normal post-command wait and attempt teardown. This is a test harness,
not a safety mechanism, cancellation implementation, or processing/publication
completion guarantee. Its successful outcome means only that fake execution
completed. Trace timestamps are virtual elapsed seconds, not wall-clock times.
Command arguments and returned values are copied at the fake-command boundary,
modeling serialized IPC: callback mutation cannot change plan literals or earlier
readbacks used by later iterations.

## Deferred deliberately

- Persistent ExperimentServer, admission, run IDs, leases, typed live driver command
  schemas/KIND matching, cancellation, orphan reconciliation, and real execution.
- Live acquisition integration, filter-result policy waits, filter updates, and
  durable run orchestration. Explicit publication into the existing pipeline is
  available only through the offline harness. Arbitrary multidimensional ragged
  windows and multi-input pipeline execution are not implemented.
- Adaptive policies and scan segmentation. The structured condition/loop IR is
  the foundation, not an implementation of adaptive analysis.
- Run folders, durable status, and wall-clock `execution.log` writing. The agreed
  future policy is one `.auspex` folder per run, immutable submissions, no run
  queue, completed scans retained on cancellation, and connected instruments
  retained after success unless explicit cleanup says otherwise.

Validation targets authoring mistakes, not hostile inputs. Authoring methods are
trusted ordinary Python; compilation is not a sandbox. The builder itself performs
no instrument I/O. `CompiledPlan.load()` uses the same basic validator as compilation.
The deferred-contract list above describes the offline compiler/harness scope;
see [server.md](server.md) for the implemented writer-only service lifecycle.

## Compiler organization

`compiler.py` contains declarations, the recording builder, and explicit authoring
phases: snapshot/normalize configuration, validate sweep definitions, record setup,
append static bindings, record nested sweeps, and record teardown. Bindings have
named fields rather than positional tuples.

`_experiment_plan.py` handles plain plan data. Normalization resolves frozen loop
bounds; `PlanValidator` separately checks types and definite value availability
without rewriting the document; a final pass assigns stable instruction IDs.
`FlowState` records available references, mutable variables, and initialized
parameters at each point. Branches intersect availability and general loops do not
export possibly uninitialized declarations. Compile/load share these passes.
Version 5 adds explicit `POINT_BEGIN`/`POINT_END` boundaries around each execution
of `sequence()`, enabling point progress without counting commands or publications.
Only version-5 plans are accepted. Versions 1–4 fail immediately with guidance
to recompile; there is no silent plan upgrade. Plans contain validated `pipelines`
attachments and explicit `PUBLISH` instructions. Unknown top-level fields and
obsolete `ACQUIRE` instructions are rejected rather than ignored.

The fake executor has separate handlers for commands, conditionals, REPEAT, WHILE,
and SWEEP. Publication validation/attachment lives in `_experiment_publication.py`;
`pipeline.py` adds the offline single-pipeline execution harness.
`plan.explain()` renders symbolic expressions with `$parameter` and
`%runtime_reference` names instead of raw expression-tree JSON. Its summary includes
registered instrument targets/KINDs, initial parameter values/units/bounds and
bindings, outer-to-inner sweep order, and resolved pipeline connections/nodes/sinks.
Actual window data is validated at execution, not by simulating authoring hooks.

Run the offline tests with:

```sh
python -B -m unittest discover -s v2/tests -p 'test_experiment*.py' -v
```
