# Auspex V2 — current handoff

V2 is a working **trusted-laboratory, writer-backed fixed-sweep** system, not a
safety-certified or high-rate acquisition platform. Python **3.13+**; distribution
`auspex-v2`, imports `auspex_v2`; V1 is preserved under `v1/`.

Read [PLAN.md](PLAN.md) for lifecycle invariants, collaboration rules and deferred
contracts, and [principles.md](principles.md) before designing a slice. This handoff
replaces the historical discussion; it is not authorization to implement the backlog.

## Resume here

- **V1 VISA port slice complete under revised scope:** 11 implemented ports plus
  16 unavailable stubs, deferred by the user to manual porting/hardware validation.
  Stub classes live in their vendor modules, raise NotImplementedError on
  construction and are excluded from module __all__, top-level exports and
  server admission. The main instrument guide lists all models.
  Current ports: SMB100B, AMI430, BNC845, BOP2020M, Picosecond10070A,
  DSInstrumentsSG12000/SG12000Pro, DAT64H, RelayDriver, DigitalAttenuator and
  Keithley2400. All use ordinary V2 declarations and automatic admission.
  `@experimental` preserves class identity, sets inherited `EXPERIMENTAL` metadata
  and emits a standard UserWarning during normal connection; no GUI/admission
  gates. Existing four lab-validated models remain unchanged.
  User approved transport/protocol-only connect and confirmed BNC active-low
  pulse/mod/ALC mappings. BNC reference setup and Kepco current-mode/VOLT MAX
  configuration are explicit operations, without enabling output. AMI Tesla
  units/Picosecond GPIB triggering are explicit settings. See the main instrument
  guide for initialization differences and retained blocking/output behavior.
  Twelve stub/decorator/export tests passed, including all sixteen constructor
  failures and export/catalog exclusions, a real experimental simulator server
  worker, all eleven ports' import/catalog metadata, lazy backend imports and
  completion. `git diff --check` passed. No ported hardware commands tested.
  Strict MkDocs build was blocked because `.venv-docs/bin/python` resolves outside
  the repository, forbidden by this session's filesystem restriction; no bypass
  attempted. The full test-suite evidence below predates this port slice.
  PLAN.md retains known V1 review concerns for manual work, not an automatic
  backlog. All 16 remaining models are stubbed, including ones without specific
  identified defects. No further implementation slice is authorized.

- The 2D offline DataBrowser was visually reviewed by the user; CET/PAL colormaps
  are excluded. Run `python tests/data/visual_data_browser.py` for the complex map.
  Standalone `save_data` and `open_data` are implemented and documented in the
  [main data guide](../docs/components/data.md), including examples, selection and
  overwrite contracts. Strict MkDocs build and example round-trip checks passed.
  Live plotting is implemented; attended review starts with
  `python tests/visual_plot_client.py` (two real simulated pipelines, no hardware).
  Core offline fitting and hanger S21 extraction are implemented; see the
  [analysis guide](../docs/components/analysis.md) and
  [hanger guide](../docs/components/resonator-fits.md). Real-trace resonator
  validation remains an attended lab step. Qubit fitting remains deferred. No
  further implementation slice is authorized; runner lifecycle refactoring remains deferred.
  Bundled driver-module `__all__` exports now control top-level imports/completion
  and automatic service admission; no separate registry list. Actual V1 driver
  ports beyond the existing four real models follow the completed revised slice above.
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
  real VISA connections use PyVISA-py `@py`. Eleven additional experimental VISA
  ports are now admitted; see Resume here and the main instrument guide for models
  and initialization differences. Whole-command deadline defaults to none.
  Each bundled vendor/simulator module's explicit `__all__` supplies intended
  concrete driver classes to the shared catalog and both Python export levels.
  Duplicate/invalid/abstract/nonlocal exports and public-name collisions fail
  visibly. Catalog construction never constructs a driver or opens hardware;
  optional backend imports stay lazy. No external plugin scan or hot reload.
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
  `from auspex_v2.data import save_data` writes one regular array to `data/data`
  in a new `.auspex` folder and returns its Path. Keyword-ordered axes accept
  arrays or matching-name Axis objects with units; optional comments are metadata.
  Explicit `force_write=True` replaces an entire valid offline store, respecting
  its writer lock; replacement is not atomic.
  `open_data` returns one loaded Dataset, selected via optional group/dataset
  filters. It accepts an exact store path, an optional-Qt folder picker (None on
  cancel), or numbered immediate-folder lookup with optional YYYY-MM-DD/`today`
  date subfolder. Numeric suffixes ignore zero-padding; missing/ambiguous matches
  fail explicitly. Explicit paths require no GUI dependencies.
- Optional DataBrowser lists regular/ragged datasets and plots non-ragged 1D and
  rectilinear 2D data (including nonuniform monotonic axes). Complex heatmaps offer
  seven component layouts, a shared plasma-default colormap selector, independent
  color scales and optional draggable, coordinate-snapped X/Y cuts in single-panel
  views. Phase is wrapped radians, never unwrapped. One additional synthetic 2D
  map opens by default in the visual demo. Ragged/higher-rank plotting is deferred.
  Separate read-only graph inspector shows topology/schemas/acknowledged configuration,
  not measurement plots or editing. Scan assembly is bounded-memory, not out-of-core.
- Optional compiled `Plot` sinks declare a current trace or growing writer-backed
  map. The store owner retains complete acquired map state, serves private Unix
  snapshots and writes best-effort final `.preview/` caches / `preview.json` for
  late viewing. Trace delivery is lossy; map-row ingestion precedes coalescing.
  Display budgets default to 8 MiB/plot, sum to at most 64 MiB/run; they are payload
  limits, not RSS guarantees. Preview errors disable displays, not required files.
  `utils/plot_client.py` discovers known ExperimentServer runs or explicit folders;
  one three-pane window shares renderers with DataBrowser and preserves view/cuts.
  New tabs do not steal focus; closed tabs remain listed; Close plot suppresses for
  the session. The 64-plot cap evicts oldest terminal entries first and visibly
  declines new entries when all retained plots are active. Run/publication status
  remains separate from preview completeness. Initial maps require one monotonic
  fixed sweep and identical actual trace grids; repeated finalized positions or
  changed grids disable only the map. No TCP broker or automatic GUI launch.

### Offline analysis

- `auspex_v2.analysis` exports V1-style AuspexFit/Auspex2DFit, LorentzFit,
  GaussianFit, MultiGaussianFit and QuadraticFit. Optional `analysis` extra adds
  SciPy/Matplotlib only; core acquisition dependencies remain unchanged.
- Inputs are explicit finite real arrays; callers select complex components and
  dataset slices. Results expose named parameters/errors, full physical-unit
  covariance, residuals/RSS and degrees of freedom. The unjustified V1 Nsigma
  statistic is not retained. Bounds and explicit p0 are supported; Matplotlib
  plots are opt-in/lazy, without backend/style changes or automatic show().
- Multi-Gaussian initialization uses resolved positive peaks and physical-coordinate
  half-prominence widths; unresolved/negative components need explicit guesses.
  Optional multistart requires an explicit seed/Generator and selects lowest RSS;
  result components and full covariance are consistently sorted by center. No
  scikit-learn, synthetic sampling, automatic component count, weighted fitting,
  GUI fitter or qubit port. Covariance errors remain conditional local
  least-squares estimates, not proof of identifiability or model validity.
- `ResonatorCircleFit` fits one hanger/notch S21 with constant complex gain and
  linear delay (Hz/s), using circle geometry for initialization and full original
  complex-data refinement. Background phase is at the reported span midpoint.
  Qc = Qc_mag exp(-i phi); Qc_eff = 1/Re(1/Qc); 1/Qi = 1/Ql - 1/Qc_eff.
  It exposes primary covariance and correlated derived-Q covariance, including
  the real/imaginary coupling-Q covariance. Invalid Q/out-of-span/nonconvergence
  fails explicitly; sampling, coverage and covariance limitations warn and remain
  in diagnostics. Local delay initialization is not global alias resolution;
  wrong geometry, variable backgrounds and multiple modes are outside scope.
  `circle_fit` is a separate geometric helper, not Q inference. No V1 list-return
  compatibility or manual post-fit Q override; no absolute-value repair of Qi.

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

Latest full implementation validation (including automatic driver exports/admission):

```sh
MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen python -B -m unittest discover -s tests -q  # 390 passed
```

Six driver-export tests cover canonical export identity, deterministic explicit
selection, invalid/duplicate exports, lazy VISA imports, real IPython from-import
completion (Jedi and non-Jedi), and an isolated new vendor simulator used through
real InstrumentServer/worker IPC without editing initializers or a registry list.
Strict MkDocs build and diff checks passed. No additional V1 drivers were ported.

The 20 analysis tests cover independent hanger loss/coupling equations across
coupling regimes, mismatch/delay/reference conventions, analytic primary and
independently propagated derived covariance, sparse/short-span diagnostics,
unphysical/out-of-span failures, storage and plotting. The documented hanger
example also passed. These are synthetic tests, not real-resonator validation.
Core-fit coverage includes independent peak/dip models, GHz/sub-nanosecond
scales, nonuniform coordinates, analytic polynomial/mixture covariance references,
seeded multistart, unresolved-mixture errors, stored-data fitting, rectangular
2D extension and real Matplotlib rendering. Documented Lorentzian/custom-2D
examples passed; acquisition imports were smoke-tested with SciPy/Matplotlib
unavailable. Numerical evidence does not validate a physical model or calibrate
its error bars for arbitrary lab data. Strict MkDocs build and diff checks passed.

Manual-helper coverage includes complex arrays/units/comments, validation before
overwrite, whole-store replacement, writer locks, dated numbered lookup, dataset
selection, ambiguity, picker cancellation and lazy optional GUI dependencies.
The picker adapter is tested offscreen with its dialog boundary replaced; actual
native folder-picker interaction still needs desktop review. `git diff --check`
passed, along with a strict MkDocs build. Live-plot integration covers actual
spawned filters/writers and a real ExperimentServer/runner, partial/final averages,
late attachment, coalesced frames after a pause, cancellation without publication,
changed-grid/duplicate-position/budget failures isolated from saved data, cache
fallback, GUI focus/close/reopen/suppression, cap policy and shared renderer state.
The live desktop demo still needs attended review; offscreen tests are not that
review. Heatmap regressions check stored non-square complex arrays,
descending/nonuniform axes, component
values/scales, cut snapping, real-only display and release. The user reports
successful desktop visual review. Crash-path resource-tracker,
offscreen Qt and asyncio slow-task diagnostics are known test output, not failed
assertions.

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
authentication, remote plot service, multi-input joins/custom worker admission
or shared-buffer digitizer path. Direct non-server instrument access bypasses lockout.

## Documentation and code navigation

The local-first MkDocs/Material manual has 19 narrative pages; snippets include the
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
| Offline fits | [analysis guide](../docs/components/analysis.md), [hanger guide](../docs/components/resonator-fits.md); `analysis/{fits,resonator_fits}.py`, `tests/analysis/` |
| Live plots | [live plot guide](../docs/utilities/live-plots.md); `filters/{preview,preview_client}.py`, `utils/plot_client.py`, `utils/src/{data_trace,data_heatmap,plot_client_model}.py` |
| Console commands/output/TUI | [utility map](../utils/README.md); `utils/src/{operator_commands,service_output,auspex_tui}.py` |

Module paths above are relative to `src/auspex_v2/`, except `utils/`. Tests mirror
subsystems under `tests/{experiment,data,filters,instruments,analysis}/`. Examples are
trusted authoring templates, not safe-to-run bench configurations without review.
