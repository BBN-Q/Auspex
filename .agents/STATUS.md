# Auspex V2 — current handoff

V2 is a working **trusted-laboratory, writer-backed fixed-sweep** system, not a
safety-certified or high-rate acquisition platform. Python **3.13+**; distribution
`auspex-v2`, imports `auspex_v2`; V1 is preserved under `v1/`.

Read [PLAN.md](PLAN.md) for lifecycle invariants, collaboration rules and deferred
contracts, and [principles.md](principles.md) before designing a slice. This handoff
replaces the historical discussion; it is not authorization to implement the backlog.

## Resume here

- No implementation slice is currently authorized. Handoff-note cleanup is complete;
  choose the next proof point with the user. Runner lifecycle refactoring is deferred.
- Inspect `git status` and preserve accumulated uncommitted work; these notes do not
  imply that implementation changes have been committed.
- Start with the [manual](../docs/index.md),
  [annotated example](../examples/documented_vna_sweep.py), and
  [utility map](../utils/README.md). Use subsystem guides/code below for detail,
  rather than treating old conversation snippets as API contracts.

## Implemented scope

### Experiments and runtime

- Python experiment classes compile to detached JSON plans with typed expressions,
  fixed/joint sweeps, hooks, bounded control flow, explicit waits/publications and
  attached pipeline definitions. `InstrumentRef` supports driver-shaped recording
  without hardware I/O. Bindings/recording setters expand at compile time; defaults
  alone do not issue commands. Explicit scalar runtime updates retain phase/bounds rules.
- `seq.module(...)` records trusted synchronous calculations with dependency manifests,
  numerical/container snapshots and symbolic indexing/slicing. Safety annotations
  do not establish purity; every COMPUTE plan saves/displays the repeatability notice.
  Calls/imports may have side effects; a running call cannot be cooperatively interrupted.
- `Interpreter` owns production semantics, `LiveInterpreter` real I/O, and the runner
  lifecycle/leases. FakeExecutor/PipelineExecutor live only in `tests/experiment/`;
  obsolete production test modules/exports are removed and dependency-tested.
- Persistent ExperimentServer accepts `client.submit(plan, name=..., dir=..., date=False)`:
  **one active writer-only run, no queue**, unique numbered run folders, durable
  status, best-effort `execution.log`, progress, reconnecting clients and cooperative
  cancellation. No service buffer sinks, auto-cleanup, incremental acquisition or
  startup catalog scanning. Terminal `cleanup()` removes catalog entries, never files.

### Instruments and operator tools

- Independent InstrumentServer: persistent inventory/run state, local socket/lock,
  process-backed connected devices, separate connection/ownership states, declared
  GET/SET/CALL, domain-wide run lockout, live/run proxies, quarantine and explicit
  reconciliation. Admission includes SimulatedSource/SimulatedVNA and real
  AgilentN5183A, YokogawaGS200, AgilentE8363C and AgilentN5230A. VISA is lazy/optional;
  real VISA connections use PyVISA-py `@py`. Whole-command deadline defaults to none.
- Explicit VISA enumeration/identification/registration works in GUI and terminals,
  with visibility filters, whitelist reload, driver-specific endpoint normalization
  and retained conflicting saved registrations. Suggestions remain advisory (GUI
  may preselect a sole suggestion); registration needs operator approval and leaves
  DISCONNECTED. Visibility controls never bypass ownership.
- Plain launcher (default), optional Textual console and standalone instrument terminal
  share `utils/src/operator_commands.py`: syntax/help, contextual availability,
  confirmations, name prompts, snapshots/ID revalidation and IPC result text.
  `CommandResult(text, exit_console)` separates policy from frontend lifetime.
  Standalone shutdown disconnects/clears/stops only on success; launcher shutdown
  preserves registrations and retains cooperative first/second Ctrl+C behavior.
- Launcher capture starts before readiness, retains 5,000 pending lines with overflow
  reporting and has a one-second total EOF budget. Optional inventory/discovery GUI
  cannot read settings/acquire; closing it leaves services running. Cached proxy
  name completion makes no server requests, including IPython/Jedi scalar discovery.
- Default sockets/state/logs use `$AUSPEX_HOME/.auspex` (announced HOME fallback).
  Whitelist is `$AUSPEX_HOME/instrument_whitelist.txt` (HOME fallback), not inside
  `.auspex/`. Service debug logs are
  per-launch, best effort and truncated; they are not durable operator history.

### Processing, data and manual use

- Shared inert pipeline definitions and ManualPipeline support independent whole-window
  processing/storage without experiment ownership. FilterServer supervises spawned
  passthrough/complex-average workers, single-ingress fan-out, bounded shared-memory
  edges/results, overlapping windows, local configuration revisions and cleanup.
  Required results backpressure or fail; only previews may drop. Filters own groups;
  runner owns windows/scans. One store process multiplexes required writer sinks.
- Complete Cartesian scans become named dimensions; irregular paths use point axes,
  variable rank-one traces ragged offsets. Dataset files embed values/coordinates,
  names/units and support manifest-independent reads. Axes/coordinates allow positional
  and case-sensitive name lookup with tuple behavior; numerical payloads stay mutable.
  Standalone AuspexStore writes and bounded completed-output buffers remain supported.
- Optional DataBrowser lists regular/ragged datasets and plots non-ragged 1D data.
  Separate read-only graph inspector shows topology/schemas/acknowledged configuration,
  not measurement plots or editing. Scan assembly is bounded-memory, not out-of-core.

## Compatibility and recent regression guardrails

- **Plan format 6; dataset format 2; InstrumentServer protocol 4.** They version
  independently. Old plans require recompilation; no V1/old-V2 dataset import or
  automatic repair. Instrument replies use 64 KiB JSON control and bounded inline
  numeric arrays (up to 16 arrays / 8 MiB), not digitizer transport.
- Failed orphan-run reconciliation restores ownership and reports that the domain
  remains locked until a persisted retry. See lifecycle tests in `tests/instruments/`.
- Ragged offsets use unsigned-safe adjacent comparisons, preserving empty rows.
  `_validated_dataset()` separates logical checks from serialization; both read paths
  work with serializer/TemporaryFile disabled. Writers retain snapshots/canonical
  encoding; readers retain envelope checks and owned mutable arrays.
- Filter supervision and both service registries now use cohesive private helpers
  with ownership/locking/ordering contracts. No common framework or lifecycle-policy
  change; `runner.run()` was intentionally not refactored. Preserve these boundaries.
- Real IPython completion regression runs in a subprocess: InteractiveShell mutates
  `__main__` and otherwise interferes with spawned-worker tests. IPython is not a
  runtime dependency; completion guarantees names, not signatures or arbitrary expressions.

## Evidence and limits

Latest full implementation validation (after terminal consolidation):

```sh
python -B -m unittest discover -s tests -q  # 339 passed
```

The 33 targeted command/CLI/TUI/discovery/address tests also passed; strict MkDocs
build and `git diff --check` passed. No new test methods were added for command
consolidation; existing context/shutdown checks were extended. This handoff-only
edit does not rerun runtime tests. Crash-path resource-tracker, offscreen Qt and
asyncio slow-task diagnostics are known test output, not failed assertions.

Automated evidence uses fake transports, simulators and real local service/process
integration. The user reports attended basic end-to-end hardware operation and
successful serial/GPIB/USB/Ethernet discovery/filter/whitelist use. This is **not**
exhaustive per-model, recovery/readback, safety, adaptive-fit or throughput validation.
The endpoint-normalization patch had offline/fake-VISA coverage, not real-backend
parser validation. Modified bench examples and TUI shutdown/capture still need
attended review. VNA drivers retain the linear-frequency-axis assumption rather
than actual stimulus-vector readback.

Recovery, cancellation, process termination and teardown do not prove outputs are
off. No hardware interlocks, durable lab timeline/settings history, stronger operator
authentication, plot subscription service, multi-input joins/custom worker admission
or shared-buffer digitizer path. Direct non-server instrument access bypasses lockout.

## Documentation and code navigation

The local-first MkDocs/Material manual has 16 narrative pages; snippets include the
canonical example as text, never execute it. Workflow is `workflow_dispatch` only;
Pages has not been enabled/published remotely. API navigation and attended prose/visual
review remain deferred. Material 9.8.0 announces EOL **May 5, 2027**; notice is documented,
not suppressed. Local commands and deployment prerequisites:
[docs/maintaining.md](../docs/maintaining.md).

| Area | Entry points / guides |
| --- | --- |
| Authoring, plans, calculations | [experiment guide](../src/auspex_v2/experiment/experiment.md); `experiment/{compiler,_experiment_plan,computations}.py` |
| Service, runner, leases | [server guide](../src/auspex_v2/experiment/server.md); `experiment/{service,runner,interpreter,resources}.py` |
| Instrument service/GUI | [operator guide](../src/auspex_v2/instruments/server/instrument_server.md); [GUI guide](../src/auspex_v2/instruments/server/instrument_server_gui.md); `instruments/server/` |
| Drivers/proxies | [driver guide](../src/auspex_v2/instruments/drivers/instrument_drivers.md); `instruments/drivers/` |
| Pipeline runtime/definitions | [filter guide](../src/auspex_v2/filters/README.md); [manual use](../src/auspex_v2/filters/manual.md); `filters/` |
| Dataset store/browser | [data guide](../src/auspex_v2/data/data.md); `data/` |
| Console commands/output/TUI | [utility map](../utils/README.md); `utils/src/{operator_commands,service_output,auspex_tui}.py` |

Module paths above are relative to `src/auspex_v2/`, except `utils/`. Tests mirror
subsystems under `tests/{experiment,data,filters,instruments}/`. Examples are
trusted authoring templates, not safe-to-run bench configurations without review.
