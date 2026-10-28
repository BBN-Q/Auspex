# An annotated source/VNA experiment

The [quickstart](quickstart.md) runs
[`examples/documented_vna_sweep.py`](https://github.com/BBN-Q/Auspex/blob/auspex-v2/examples/documented_vna_sweep.py).
The code fragments below are included directly from that file, not maintained as
separate copies. All methods shown belong to the same experiment class.

The experiment sets three source frequencies, records one VNA trace at each point,
computes a sampled magnitude minimum, and saves the complex traces. It demonstrates
control flow and data handling, not a model of coupled source/VNA hardware: the two
simulators are independent.

## 1. Import authoring objects

```python
--8<-- "examples/documented_vna_sweep.py:imports"
```

`Experiment` describes a measurement. `InstrumentRef` describes a role; it does not
construct a connected driver. Pipeline definitions are inert until execution.
Importing these objects does not discover or open laboratory hardware.

## 2. Declare roles, parameters, and an output

```python
--8<-- "examples/documented_vna_sweep.py:declarations"
```

The driver classes provide authoring capabilities and infer each role's `KIND`.
An instance later maps `source` and `vna` to registered inventory names. KIND
compatibility does not prove that every same-KIND model implements every command.

`source_frequency` is requested experimental state. Its binding records a source
frequency SET followed by a 10 ms WAIT whenever that parameter is applied. Every
sweep point applies it, including repeated values. The declared bounds are checked
before its setting operation. A default alone issues no hardware instructions.

`minimum_magnitude` has no instrument association or initial value. It will hold a
runtime numerical result. Its bounds reflect this simulator's response, not a
universal calibration constraint on real S-parameters.

`trace` names the complex value array. Its schema declares dtype, dimension name,
coordinate unit, and a byte limit including coordinates. It does not declare a
fixed point count. `source_frequency` is an outer sweep coordinate; `frequency`
is the coordinate inside a VNA trace. They are distinct quantities despite sharing
units.

## 3. Attach a required writer

```python
--8<-- "examples/documented_vna_sweep.py:pipeline"
```

A pipeline has an explicit named input and connector-pair edges. The writer
selects `measurement/s21` within the eventual run store; it does not choose the
run directory. `connect()` associates the experiment's output with that input.

This is a single-input pipeline with a required file sink. ExperimentServer also
supports independent named inputs and multiple attached definitions sharing one
store; buffer-backed runs remain unsupported. Filter workers and the store owner
are created during execution, not by defining this graph.

## 4. Record initialization explicitly

```python
--8<-- "examples/documented_vna_sweep.py:setup"
```

These statements run during **recording**, against a detached copy of the configured
experiment. Driver-shaped assignments emit SET instructions. They do not contact
an instrument during compilation.

The explicit `set_parameter()` applies the configured initial source frequency
before enabling output. Setup may set a parameter even when it will later be
swept. It then configures a 20 MHz VNA span around 6.1 GHz with 201 samples.

Do not treat this as a real-hardware startup procedure. Real RF/bias measurements
require reviewed power, protection, routing, and output policies.

## 5. Record one acquisition and calculation

```python
--8<-- "examples/documented_vna_sweep.py:sequence"
```

`get_trace()` records an instrument CALL. Its actual arrays do not exist while
recording. The explicit tuple selections `result[0]` and `result[1]` become references
to future frequency and response arrays.

`seq.module("numpy")` represents a computational instrument. `abs` and `min` are
recorded as ordered runtime calls receiving concrete arrays. Ordinary imported
NumPy outside this proxy would instead execute during recording.

The minimum is a **sampled** minimum of `|S21|`, not a resonance fit, linewidth, or
uncertainty estimate. The simulator's center sample has magnitude approximately
0.3. Both the callable-level trust warnings and the whole-plan repeatability
warning apply; no purity or reproducibility is inferred from the NumPy name.

`set_parameter()` validates and stores that scalar result. It does not issue an
instrument command because this parameter has no association. The requested value
is visible in diagnostic execution logging; it is **not** automatically a separate
saved measurement dataset or a physical readback.

`publish()` submits one complete raw complex trace with its actual coordinate
array. Submission may wait for capacity, but it does not acknowledge final storage.
The default scan covers all three sweep points; the writer publishes the assembled
dataset at scan completion, not after each individual point.

## 6. Record bounded cleanup

```python
--8<-- "examples/documented_vna_sweep.py:teardown"
```

Normal completion, ordinary failure, and cooperative cancellation attempt teardown.
This example switches the simulated source output off. Cleanup stops on its first
error; a failed teardown makes the run fail and leaves ownership for reconciliation.
Loss of service supervision can prevent teardown entirely.

An instrument call or Python calculation that never returns can block cooperative
cancellation. Process termination does not establish physical safety.

## 7. Configure the instance and submit

The command-line entry point supplies `args` from its directory/socket/name options:

```python
--8<-- "examples/documented_vna_sweep.py:submission"
```

Role binding selects registered names; registration itself is a separate operator
action. Compilation freezes the experiment, processing definition, and sweep points
into an inspectable version-6 JSON plan. Later edits to the authoring instance do
not change that plan. Runtime Python implementations remain external dependencies.

`submit()` returns a run handle promptly. ExperimentServer allocates a new numbered
`.auspex` directory under the absolute existing server-host directory, and saves
the plan there. The notebook or script can disconnect without cancelling the run.

## 8. Interpret the result

A successful run publishes one `measurement/s21` dataset with shape `(3, 201)`.
Rows correspond to source-frequency points; columns correspond to VNA frequencies.
The simulators yield identical rows because their states are independent.

Read status before interpreting absence of data. An interrupted partial scan is
not published as a complete measurement. Conversely, data can already be published
when a later teardown or finalization error causes the run to fail.

The [quickstart](quickstart.md#6-read-the-data) shows named-coordinate access. The
[experiment guide](components/experiments.md) explains the general recording model,
and the [instrument guide](components/instruments.md) explains ownership.
