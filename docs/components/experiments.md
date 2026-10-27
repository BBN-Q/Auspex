# Experiments

An experiment combines instrument roles, requested parameters, a measurement
sequence, and an explicit processing graph. It is authored in Python, compiled
into a detached plan, and submitted to ExperimentServer.

Start with the [annotated example](../annotated-experiment.md). For acquisition
controlled by your own Python loop, [ManualPipeline](filters.md#manual-processing)
is an independent alternative; no experiment service is required.

## Recording is not execution

`compile()` records from a detached copy. Setup, sequence, per-axis hooks, and
teardown emit instructions through `seq` or driver-shaped proxies. Ordinary Python
locals help construct the plan; they are not persistent runtime state.

- `seq.if_()` records a runtime condition; Python `if` evaluates during recording.
- `seq.repeat()` and bounded `seq.while_()` record structured loops.
- `seq.variable()` / `seq.assign()` express mutable runtime scalar locals.
- Instrument results are symbolic: select fields/indices explicitly rather than
  iterating or implicitly unpacking them.

The saved plan is plain version-6 JSON with `save()`, `load()`, and `explain()`.
Older plan versions require recompilation. Validation catches known schema/type
errors and unavailable values, but does not simulate all possible runtime data.

## Parameters and application

A configured `.value` is initial requested state, not a command or readback.
A class-level `.bind()` associates a parameter with a setting and optional wait.
For several operations, register a named recorder using `@parameter.setter` with
signature `(self, seq, value)`. The method records instructions; it is not a
per-point Python callback. Give the method a name distinct from its parameter.

Instance `.bind()` or `.setter(function)` replaces the whole association before
compilation, without changing other instances. Direct/indirect setting recursion
fails compilation. Inherited setters are registered functions: overriding the
method name alone does not re-register them.

| Change | Meaning |
| --- | --- |
| Set `exp.parameter.value` before compilation | Configure requested initial state; no hardware operation |
| `seq.set_parameter(parameter, value)` | Validate/store requested state, then apply its association |
| Assign `self.source.frequency` in a hook | Record an instrument SET without inferring a parameter update |

Explicit updates apply even for equal values. Setup and teardown may update swept
parameters; explicit updates to them in the main phase, including axis hooks,
remain forbidden. Sweeps apply each point automatically. The first-added sweep
changes fastest; a joint sweep advances its listed parameters together.

## Recorded calculations

`seq.module("numpy")` or another importable module records numerical calls as
computational instruments. They execute synchronously in plan order and return
arrays/scalars or supported nested containers. Parameters remain scalar. Use an
importable wrapper for callback-heavy fitting or analysis returning rich objects.

A callable without `@safe_for_recording` warns during compilation. That decorator
is an author assertion, not verification. Every compiled plan containing COMPUTE
also warns, independently of markers:

> This compiled experiment executes runtime Python code. Auspex cannot guarantee repeatability

The notice is saved in `warnings` and shown by `explain()`. Paths, observed versions,
and author assertions appear in the operation manifest, but implementations and
transitive dependencies are not frozen. Imports and functions are trusted Python,
not sandboxed code. Cancellation cannot interrupt an arbitrary function.

## Execution and failure

ExperimentServer accepts one active writer-backed run, with no queue. The runner
owns the instrument lease, connects required devices, executes setup and the scan,
attempts teardown, then waits for processing/storage completion. A point completed
is not yet a successful run.

Successful teardown releases the instrument lease before final processing drains;
a later storage failure still fails the run. A failed teardown or lost runner can
leave an orphaned domain. Notebook disconnection alone does not cancel a run.
See [ownership and recovery](instruments.md#ownership-and-recovery).

Current limits include a single nonempty pipeline ingress, no server-side buffer
transfer, no automatic interrupted-run resumption, and no general adaptive sweep
mutation. Recorded calculations do not remove these limits.

## Repository notes

- [Experiment authoring and compilation](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/experiment/experiment.md)
- [ExperimentServer lifecycle](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/experiment/server.md)
- [Run resources and handoffs](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/experiment/resources.md)
