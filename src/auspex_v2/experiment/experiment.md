# Experiment compiler and explicit pipeline publication

`compiler.py` implements trusted Python authoring, a structured instruction
compiler, detached version-6 JSON execution units, `save`/`load`, and `explain()`.
Only version 6 is accepted; older plans must be recompiled from their experiment
authoring code. New compilation requires an attached pipeline. The loader embeds no
function source and uses neither pickle nor `eval`; recorded Python functions are
explicit external dependencies, not sandboxed code.

[ExperimentServer](server.md) runs saved writer-only plans against the explicit
simulated/real driver registry under an instrument lease, with real waits,
cooperative cancellation and reconnectable handles. Its `LiveInterpreter` uses
the production instruction semantics in `interpreter.py`; test executors are
repository-only fixtures, not runtime dependencies or public execution APIs.

## Example

```python
from auspex_v2.experiment import (
    Experiment, ExperimentServerClient, InstrumentRef, FloatParameter, OutputConnector,
)
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, WriteToFile

class Measurement(Experiment):
    source = InstrumentRef(kind="DC_SOURCE")
    settling = FloatParameter(default=0.1, unit="s")
    voltage = FloatParameter(default=0.0, unit="V", bounds=(-1, 1)).bind(
        source, "voltage", wait=settling)
    signal = OutputConnector(dtype="float64")

    def setup_pipeline(self):
        signal = PipelineInput("signal")
        writer = WriteToFile(name="disk", dataset="signal")
        PipelineDefinition(inputs=[signal], edges=[(signal, writer.sink)])
        self.connect(self.signal, signal)

    def setup(self, seq):
        seq.call(self.source, "prepare")
        seq.set_parameter(self.voltage, self.voltage)  # Explicit initial application.

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
        seq.set_parameter(self.voltage, 0.0)
        seq.call(self.source, "finish")

exp = Measurement()
exp.bind_instrument("source", "lab_source")
exp.add_sweep(exp.voltage, [-0.5, 0.0, 0.5])
plan = exp.compile()
plan.save("measurement.plan.json")
print(plan.explain())
# Start the services and register a matching source first. These abstract
# source commands illustrate the API; adapt them to your admitted driver.
run = ExperimentServerClient().submit(plan, name="measurement", dir="/path/to/data")
print(run)  # measurement_000 [1abc4de]
```

Instrument roles use the fixed, case-sensitive `InstrumentKind` vocabulary. An
abstract role supplies `kind=`. Supplying `driver=` infers KIND and enables its
shorthand API; `InstrumentRef().bind(Driver)` is the equivalent class-body form.
If both are supplied they must agree. `kind` and `driver` are read-only after
construction; use `.bind()` before compilation to select another same-KIND
authoring driver. ExperimentServer compares the role KIND to
the registered driver's KIND; unsupported driver members fail at dispatch. The
fake callback is deliberately supplied separately, never saved in the plan.

## Recorded Python calculations

Ordinary imports execute **now**, while `seq.module("numpy")` or another importable
module records calls for **runtime**, even when all arguments are concrete:

```python
# Inside sequence(self, seq):
samples = np.cos(2 * np.pi * 3 * np.arange(32) / 32)  # Ordinary compile-time NumPy.
recorded_numpy = seq.module("numpy")
spectrum = recorded_numpy.fft.fft(samples)           # One ordered COMPUTE instruction.
seq.set_parameter(self.component, spectrum[self.bin_index].real / 32)
```

See [`examples/recorded_fft.py`](../../../examples/recorded_fft.py) for a complete
submission example requiring no instruments. Start the services first; the plan
publishes the independently known coefficient 0.5 through ExperimentServer.
Arrays live in runtime results; parameters remain scalar.

The proxy traverses modules and records function/ufunc calls generically, without
individual NumPy/SciPy adapters. Supported arguments/results are numerical arrays
and scalars, strings/booleans/None, and nested lists, tuples and string-keyed
dictionaries. Concrete arrays retain dtype and shape in JSON; complex values have
explicit numeric components. Container subclasses (including NumPy named tuple
results) become plain containers; their extra attributes/methods are not retained.
Dictionary items and keyword arguments use ordered pairs in JSON so canonical key
sorting cannot change Python insertion order. Inputs and results are copied at the
call boundary, preventing input mutation from
altering earlier runtime results or plan literals. This is not zero-copy processing.

Symbolic results support integer/string selection, symbolic indices, slicing,
multi-axis tuple indices, and restricted numerical `.real`/`.imag` access. They do
not expose arbitrary attributes/methods. Use recorded functions for numerical
operations beyond the existing scalar-oriented expression operators. Rich objects,
constructors, function-valued arguments/callbacks and nonnumerical arrays are not
supported. Wrap callback-heavy fitting or legacy analysis in an importable ordinary
function that returns supported data; missing libraries or unsuitable functions
simply raise errors. SciPy is not installed automatically by this feature.

### Trust and dependencies

Every successful `Experiment.compile()` containing a COMPUTE instruction emits
`RuntimePythonWarning`: **"This compiled experiment executes runtime Python code.
Auspex cannot guarantee repeatability"**. This whole-experiment notice also applies
to reviewed functions and nested instructions in setup/main/teardown. It is saved
in the plan's top-level `warnings` list and displayed prominently by `explain()`;
plans without COMPUTE have an empty list. Notices are derived from instructions
when constructing/loading a plan, without changing execution semantics. Ordinary
Python warning filters remain under user control; the saved notice is independent
of those filters. Loading a plan does not re-emit the compile-time warning.

Unmarked callables emit `RecordedPythonWarning` during compilation, once per
callable (not once per point). There is no implicit trust for NumPy, SciPy or an
Auspex namespace. A reviewed wrapper can carry an **author assertion**:

```python
# In an importable laboratory analysis module:
from auspex_v2.experiment import safe_for_recording

@safe_for_recording
def fit_response(frequency, response):
    # Ordinary NumPy/SciPy/legacy analysis; return only numerical/container data.
    ...

# While recording:
fit = seq.module("lab_analysis").fit_response(frequency, response)
seq.set_parameter(self.center, fit["center"])
```

The decorator suppresses only the callable-level warning, never the whole-plan
repeatability notice; it does not prove purity. The author asserts
no external side effects, input mutation, or dependence on persistent hidden state,
including in transitive dependencies. The runner never injects an Experiment,
sequence builder or instrument handle into the function, but **this is trusted
Python execution, not a sandbox**. Imports and functions can execute arbitrary code,
write files, access hardware, hang or crash. Review saved plans and their dependencies.

The JSON `operations` manifest stores each module/attribute path, observed root
module version when available, and the author's safety assertion at compilation.
`explain()` displays these alongside ordered COMPUTE instructions. Function source,
closures and implementations are not embedded or frozen. Versions are informative,
not enforced or a complete dependency/environment inventory. Loading validates
structure without resolving these imports; service preflight resolves them before
folder allocation/instrument acquisition. Imports themselves may have side effects.

### Ordering, failure and cancellation

Calls run synchronously in instruction order, once per executed COMPUTE, in both
the live interpreter and test backends. Reusing a result reference does not repeat the call.
There is no helper thread/process or automatic parallelism. Exceptions and
unsupported results fail the run and attempt the usual teardown.

Live cancellation is checked immediately before and after a successful call, never
inside arbitrary Python. A hung call blocks cooperative cancellation and teardown;
forced runner termination cannot undo side effects or guarantee hardware safety.
Teardown calculations follow the existing cleanup policy (a cancellation request
does not skip cleanup). Analysis memory/time are not bounded by pipeline budgets.

## Pipeline attachment and publication

An `OutputConnector` declares `dtype`, `dimensions`, `axis_units`, optional value
`unit`, and `max_bytes` (default 64 KiB including coordinates).
`temperature = OutputConnector(dtype="float64", unit="K")` declares a scalar
measurement with no measurement axis; value units are independent of axis units. It does not fix a
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
Checks are syntactic, not exhaustive path/reachability analysis. Execution supports
multiple named inputs and multiple attached definitions, including output fan-out.
The runner combines attachments into one graph and one physical store owner;
writers choose distinct group/dataset destinations in the same run folder.
Definition-local names are qualified as `pipeline0.name`, `pipeline1.name`, etc.
when there are multiple attachments; explicit group/dataset names do not change.

Each `seq.publish(output, ...)` routes only to that output's connected inputs.
Window IDs and averaging state are independent per input. Counts need not match;
there is no enforced point pairing or numerical join. All branches share scan/run
boundaries, including inputs that never publish. An idle writer produces no dataset.
Completion waits for all required paths, and a required branch failure fails the
run without removing already published datasets. Explicit `EmptySource()` execution
remains deferred: connect a real output and simply do not publish it when idle.

Compiled plans contain resolved schemas, named input/output connections, filter
configuration and sink destinations. Later changes to definitions or experiment
configuration cannot alter a saved plan. Schema/connection validation also runs on
load. `Experiment.outputs` and `Experiment.graph` are not authoring APIs:
output schemas derive from declared `OutputConnector`s, and the processing graph
from attached `PipelineDefinition`s. Assigning either legacy attribute or calling
`set_graph()` fails with guidance rather than saving inert configuration.

## Saved-plan execution

```python
from auspex_v2.experiment import CompiledPlan, ExperimentServerClient

plan = CompiledPlan.load("measurement.plan.json")
run = ExperimentServerClient().submit(plan, name="measurement", dir="/path/to/data")
print(run)  # measurement_000 [1abc4de]
# Optional: result = run.wait(timeout=60)
```

The server owns the run folder, saves the submitted plan and manages instrument
leases, processing and cleanup. Disconnecting the submitting client does not
cancel the run. Only writer-backed plans are admitted in the current service.
See [the lifecycle contract](server.md) and `examples/compiled_publication.py`.
For direct processing without a saved experiment plan, use
[ManualPipeline](../filters/manual.md), not a test executor.

The former FakeExecutor/PipelineExecutor APIs and compatibility shims have been
removed from the installed package. Their virtual-time/callback-based counterparts
live only under `tests/experiment/` to verify shared production semantics.

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

    @bias.setter
    def set_bias(self, seq, value):
        seq.set(self.source, "voltage", value)
        seq.wait(0.5)

    def sequence(self, seq):
        seq.set_parameter(self.bias, 0.4)
        seq.call(self.source, "measure")

    def teardown(self, seq):
        seq.set_parameter(self.bias, 0.0)

exp = BiasMeasurement()
exp.bind_instrument("source", "lab_source")
# Add output declarations and attach a pipeline as above before exp.compile().
```

### Declared and external parameter setters

A simple class-body `.bind(source, "voltage", wait=0.5)` associates a parameter
with one setting. Binding waits and keyword arguments can also reference other
parameter declarations, as the first example demonstrates. A decorated setter
instead records a multi-step operation using the same sequence API as a hook.
Its signature is `(self, seq, value)`; `value` is the symbolic **current requested
parameter value**, not a physical readback or a Python value to branch on.
Runtime conditions use `seq.if_()`, not Python `if`. Existing instruction limits
still apply, including restrictions on runtime WAIT expressions.

The decorator registers and returns the function, not a property. Give it a
name distinct from the parameter (`set_bias`, not `bias`). Subclasses inherit the
registered function; overriding that method name alone does not re-register a
setter. To change the association, redeclare the parameter or configure the instance.

An ordinary external function can replace an association before compilation:

```python
def apply_bias(exp, seq, value):
    seq.set(exp.source, "voltage", value)
    seq.wait(0.1)

exp.bias.setter(apply_bias)                 # Replace the entire existing association.
exp.bias.bind(exp.source, "voltage")        # Or replace it with a simple binding.
```

There is one active association per parameter. Instance overrides do not change
other instances or class declarations. Recorders must be ordinary synchronous
functions, not coroutines, generators or bound methods. They run on the detached
recording experiment during compilation, not once per hardware point. Use the
supplied experiment to reference its instruments/configuration rather than
capturing another instance. Trusted Python globals/closures are not sandboxed or
frozen; the **resulting instructions** are detached and saved. Neither callable
objects, source code nor function names are needed to execute the saved plan.
Direct and indirect parameter-setting recursion is rejected during recording,
even inside a recorded conditional. Frozen-configuration checks apply to setters
just as they do to hooks.

`set_parameter` records `SET_PARAMETER`, followed by the expanded association's
instructions. It checks the value's type and declared bounds before changing
requested state or executing **any** of those instructions. Literal errors fail compilation; dynamic values are checked by
the executor. Subsequent symbolic references see the new requested value. An
unbound parameter changes runtime state only. Requested values are **not physical
readbacks**; a failed bound command does not imply the hardware adopted the value.

Defaults and configured `.value` initialize requested state only: they never
implicitly invoke an association. To apply an initial value, explicitly record
`seq.set_parameter(self.bias, self.bias)` in setup, or supply a literal if the
parameter has no initial value. Every explicit update invokes its association,
even when the value is unchanged. Version-5 plans must now be recompiled for the
version-6 recorded-computation format; no old plan is silently rewritten.

A parameter without an initial value must be initialized before use. Setup and
teardown can explicitly set **swept or unswept parameters**, including initialization
in teardown after setup failed. During the main phase (including axis hooks),
explicit changes to swept parameters remain rejected; automatic sweep assignments
still apply each parameter's association in declared order at every point.
Setup/teardown updates change requested state, not acquired sweep coordinates.
Adaptive sweep updates and separate scheduled/requested/readback coordinate
tracking remain future work. Direct device commands are not inferred into the
parameter state or sweep metadata.

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
  Set configuration through `exp.voltage.value`; associate settings using `.bind()`
  or a `.setter` recorder, in the class body or before compiling the instance.
  NumPy scalars/arrays, including nested configuration values, are normalized to
  plain JSON values before validation. Integer parameters still reject floats.
- `setup` records explicit initialization; no unswept parameters apply automatically. The first sweep added
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

## Runtime and test boundaries

The production Interpreter owns expression evaluation, typed state updates,
control flow, computations and instruction budgets. LiveInterpreter supplies
instrument dispatch, real waits, pipeline publication, logging and supervision.
Required backend operations raise if unimplemented; there is no production
no-op command callback or virtual clock. The runner, not the shared interpreter,
owns phase ordering, teardown, leases and publication finalization.

Test-only FakeExecutor and PipelineExecutor inherit those same production
semantics, adding injected responses, virtual time, window/trace capture and a
convenient test lifecycle. They are not distributed, re-exported or imported by
ExperimentServer. Tests with these fixtures do not establish hardware behavior;
real-service/simulated tests exercise the actual supervised lifecycle.

## Deferred deliberately

Arbitrary multidimensional ragged windows, multi-input numerical joins,
adaptive sweep updates and multi-scan segmentation remain unsupported. The current
writer-only server, its ownership/recovery limits and durable run artifacts are
documented in [server.md](server.md).

Validation targets authoring mistakes, not hostile inputs. Authoring methods are
trusted ordinary Python; compilation is not a sandbox. The builder itself performs
no instrument I/O. `CompiledPlan.load()` uses the same basic validator as compilation.
See [server.md](server.md) for the implemented writer-only service lifecycle.

## Compiler organization

`compiler.py` contains declarations, the recording builder, and explicit authoring
phases: snapshot/normalize configuration, validate sweep definitions, record setup,
record nested sweeps, and record teardown. Simple bindings have named fields
rather than positional tuples. Setter recorders expand at explicit updates and
sweep assignments; the saved plan contains only ordinary instructions. The builder
and plan validator independently enforce the setup/main/teardown update policy.

`_experiment_plan.py` handles plain plan data. Normalization resolves frozen loop
bounds; `PlanValidator` separately checks types and definite value availability
without rewriting the document; a final pass assigns stable instruction IDs.
`FlowState` records available references, mutable variables, and initialized
parameters at each point. Branches intersect availability and general loops do not
export possibly uninitialized declarations. Compile/load share these passes.
Version 5 adds explicit `POINT_BEGIN`/`POINT_END` boundaries around each execution
of `sequence()`, enabling point progress without counting commands or publications.
Version 6 adds the operation manifest, COMPUTE instructions and numerical/container
argument expressions, indexing and components. Only version-6 plans are accepted.
Versions 1–5 fail immediately with recompile guidance; there is no silent upgrade. Plans contain validated `pipelines`
attachments and explicit `PUBLISH` instructions. Unknown top-level fields and
obsolete `ACQUIRE` instructions are rejected rather than ignored.

`interpreter.py` supplies shared expression and instruction handlers;
`runner.py` supplies the live backend and supervised lifecycle. Publication
validation/attachment lives in `_experiment_publication.py`. Virtual-time test
backends live in `tests/experiment/fake_executor.py` and `pipeline_executor.py`,
with no reverse dependency from production code.
`plan.explain()` renders symbolic expressions with `$parameter`, `%named_variable`,
and `x1`, `x2`, … temporary-result aliases instead of raw expression-tree JSON.
COMPUTE instructions and the dependency list use qualified function names such as
`numpy.fft.fft`. Empty instruction fields (None, empty strings/collections) and
empty binding kwargs are omitted; zero and false remain visible. Empty arguments
inside a nonempty argument list remain visible as well. These are display-only
changes: saved operation/result IDs, plan identity, and execution are unchanged. Its summary includes
registered instrument targets/KINDs, recorded Python dependencies and author safety
assertions, initial parameter values/units/bounds and bindings, outer-to-inner sweep
order, and resolved pipeline connections/nodes/sinks.
Actual window data is validated at execution, not by simulating authoring hooks.

Run the offline tests with:

```sh
python -B -m unittest discover -s tests -p 'test_experiment*.py' -v
```
