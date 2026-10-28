# Auspex V2 — development decisions and roadmap

Use [STATUS.md](STATUS.md) for implemented scope, validation and code navigation;
use this file for durable constraints and deferred designs. Neither authorizes
implementation. Agree on the next small slice and its acceptance criteria with
the user before proceeding. Consult [principles.md](principles.md).

## Goal and working rules

Build an inspectable Python 3.13+ laboratory-control system: familiar Python
experiment classes, detached saved plans, independently running local services,
explicit instrument ownership, and bounded parallel processing. Preserve convenient
direct driver/manual-data workflows. V1 lives under `v1/`; do not migrate its
inheritance hierarchies wholesale or promise compatibility with every V1 experiment.

- Humans choose scope and consequential policies. Prefer ordinary idioms, explicit
  data flow, cohesive helpers and visible state over generic supervision,
  transaction, grammar or plugin frameworks.
- Give every class a purpose docstring; nontrivial methods/tests (roughly over
  10–12 lines) need an intent/contract docstring. Reuse integration coverage;
  add a regression only for a concrete gap.
- A cleanup must preserve protocols, timeouts, errors, ordering, recovery policy,
  public APIs and service topology unless a change is explicitly agreed. Report
  correctness defects separately. Terminal wording normalization is approved.
- Inspect `git status` before editing/committing; preserve accumulated work. Do not
  commit, discard changes, enable remote publishing or expand scope unasked.
- This is trusted laboratory Python, not a sandbox, safety controller, distributed
  service, hermetic environment packager or interrupted-run resumption system.

## Durable architecture and lifecycle contracts

### Instruments and local services

- Two independent persistent local services: **InstrumentServer** owns inventory,
  device workers and leases; **ExperimentServer** admits/supervises runs. FilterServer
  is per-run supervision, not a third persistent daemon. Clients may disconnect
  without cancelling a run. IPC is local Unix sockets, not network-wide discovery.
- Registration records an admitted driver and physical identity without I/O;
  connection is explicit. One spawned worker owns each connected physical device,
  not each logical channel. Imports must not probe hardware, load unavailable
  backends eagerly or substitute mocks. Only declared GET/SET/CALL members dispatch.
- A run locks **all registered instruments**, including unused ones, against live
  clients. This does not reset unused hardware. Live use requires CONNECTED;
  closing a live lease releases ownership but leaves the device connected. Direct
  access outside Auspex bypasses this protection.
- Connection/disconnection failures latch UNKNOWN; worker death, command failure
  and termination latch TERMINATED. Recovery explicitly acknowledges inspection
  and returns to DISCONNECTED, not verified-safe or connected. Never silently
  retry uncertain commands. Termination does not release ownership or switch off
  physical outputs; quarantine and orphan-run reconciliation are distinct actions.
- Lost run ownership remains orphaned/locked until explicit reconciliation.
  Reconciliation must retain the old lock if persistence fails. Do not impose a
  blanket transaction policy: existing transitions have different rollback rules.
- Only disconnected, worker-free, unowned inventory may be cleared. Normal service
  exit preserves registrations; forced startup `--clean` discards saved inventory/run
  state without hardware I/O and is not recovery. Restart never resumes a run.
- Discovery enumerates without opening hardware; identification explicitly sends
  `*IDN?` through temporary connections. Suggestions are advisory: the GUI may
  preselect a sole suggestion, but registration requires operator approval of a
  driver/name and never connects. Stable candidate letters differ from inventory IDs.
  Serial, Prologix and known permission denials are hidden by default. Whitelists
  affect visibility only; bulk identification always excludes serial, Prologix and
  registered resources. Ownership restrictions remain server-enforced.
- Physical identity must match the actual driver endpoint. Normalize VISA aliases
  without DNS guesses, preserving case-sensitive components and driver-specific
  bare-IP defaults. Whitelist entries require complete VISA resources. On restart,
  exclude **every** conflicting saved registration and retain originals/errors for
  inspection; never select an arbitrary winner or implicitly connect it.

### Experiment authoring and execution

- `Experiment.compile()` freezes declarations, configured values, typed instructions
  and the attached pipeline. Plans are plain inspectable JSON, not saved source or
  pickled experiment objects. Fresh processes can start supported plans given their
  dependencies; notebook edits cannot alter a compiled plan. Reject old WIP formats
  with recompile guidance rather than silently upgrading.
- Driver-shaped recording uses cached declarations, never driver construction.
  Keep driver kwargs separate from runner timing; settling delays lower to WAIT.
  Fixed/joint sweeps retain V1 first-added-fastest order. Explicit publication is
  one complete window; the default scan encloses the sweep traversal.
- Parameter defaults initialize requested state only, not hardware. Bindings and
  `(experiment, seq, value)` setter functions expand into ordinary instructions at
  compilation. Every explicit update reapplies the association, even at equal values.
  Setup/teardown may update swept parameters; main/axis hooks may not. Preserve
  bounds/type checks, detached declarations and configuration/recursion guards.
- `seq.module(...)` always records synchronous COMPUTE calls, including concrete
  arguments. Numerical/container snapshots and supported indexing are explicit;
  arbitrary objects/callbacks are not supported. Imports/functions are trusted code;
  safety markers are author assertions, dependencies are not frozen, and calls can
  block cancellation until return. Every COMPUTE-containing plan persists/displays
  the whole-plan repeatability warning, even for marked-safe functions.
- Production `Interpreter` owns instruction semantics; `LiveInterpreter` supplies
  I/O; `runner.run()` owns phases, leases and cleanup. Fake executors belong only
  under tests. Do not refactor the sequential runner lifecycle wholesale without
  a separately reviewed need.
- Submission preflights supported topology/budgets and inventory before folder
  writes. After folder claim, operational failures leave records, not reused names.
  The runner acquires/rechecks its lease, starts processing, connects required
  instruments, then obtains a persisted RUNNING acknowledgment before setup.
- Pre-setup failure releases the lease without experiment teardown; completed
  connections stay connected. Once setup starts, ordinary success/failure/cancellation
  attempts teardown. Teardown ignores ordinary cancellation, stops at its first
  error, marks FAILED and leaves the lease orphaned. Successful teardown releases
  ownership **before** processing drain; later publication failure still fails the run.
- Loss of ExperimentServer supervision stops further commands **including teardown**;
  runner death also leaves an orphaned lease. In-flight calls cannot be undone.
  Status is authoritative; failed status persistence degrades admission. Logs are
  best effort, not acknowledged audit records. No watchdog or automatic replay.
- Cancel unfinished scans rather than publish them as complete. Retain already
  published datasets despite later failure. Point progress/publication is not proof
  of overall success. The volatile catalog is not rebuilt by scanning old folders;
  known-folder durable terminal status remains usable after cleanup/service loss.

### Processing and storage

- Pipeline definitions are inert, detached configurations shared by experiments
  and manual processing. Runtime currently requires one nonempty ingress and one
  input per filter; expressible multi-input/empty definitions do not imply execution
  support. Explicit external wiring replaces internal setup with a warning.
- Topology is fixed during execution. Spawn one worker per logical filter; bounded
  shared-memory slots belong to edges. Fan-out copies. No inferred process placement,
  global reference-counting scheme or multi-input join framework.
- Window end is an ordered message, never queue emptiness. Runner owns windows/scans;
  filters own groups, which cannot cross scans. Required completion includes result
  drainage and writer publication acknowledgments. Complex pointwise averaging
  requires exactly equal actual coordinate grids; never silently interpolate.
- Required data never drops: backpressure waits, fail-on-capacity latches failure.
  Only previews are lossy. Configuration updates drain the admitted prefix, apply
  acknowledged filter-local revisions and reject incompatible active groups;
  schema-changing updates remain deferred. Reap workers before reclaiming shared
  memory, respecting admission ownership.
- Multiple logical writer sinks use one physical store owner. Complete scans assemble
  within bounded memory; this is not an out-of-core or digitizer-throughput claim.
  Regular sweeps use named dimensions, irregular paths a point dimension, and
  variable rank-one traces ragged offsets. No general multidimensional ragged API.
- Dataset format and plan format are independent. Validate logical data without
  filesystem I/O; serialization owns snapshots/encoding. Validate ragged offsets by
  adjacent comparisons before int64 conversion (unsigned subtraction can wrap);
  equal offsets represent empty rows. Loaded payloads remain mutable; named metadata
  sequences remain tuple-compatible. No silent repair or V1 data compatibility.

### Operator interfaces and documentation

- `utils/src/operator_commands.py` is the sole operator-command policy for standalone,
  plain-launcher and Textual consoles. Explicit contexts, not socket presence, govern
  availability. Shared results carry text/exit intent; adapters own input, display,
  UI threading and console lifetime. Startup arguments and server dispatch stay separate.
- Standalone confirmed `shutdown`: disconnect all → reject partial failures → clear
  inventory → request service shutdown → exit only on success. Launcher Ctrl+C instead
  cancels/waits and stops ExperimentServer before InstrumentServer, preserving
  registrations; second Ctrl+C skips cooperative waiting, not an in-flight call.
- Launcher owns pipe readers before readiness/UI startup. Keep the newest 5,000
  pending lines and report overflow; wait one second total for EOF after shutdown.
  Warn about inherited-open pipes; never kill descendants merely to obtain stdout EOF.
- GUI scope is inventory/discovery/connection management, not live settings/acquisition.
  Qt/Textual remain optional frontend dependencies. Name-only notebook completion
  uses cached capabilities and must make no server requests.
- Documentation is terse, scientific, local-first Markdown with explicit navigation.
  Preserve source Markdown, V1 documentation and the README funding acknowledgment.
  No API/execution plugin or automatic remote publication; docs workflow stays
  manual-only unless the user explicitly authorizes a change.

## Deferred work — select with the user, not an automatic queue

1. **Attended validation:** bench-review fixed-sweep examples/resources/settings;
   exercise actual terminal/TUI shutdown/output behavior; review manual prose/visuals.
   Hosted Pages deployment requires explicit authorization/settings work. Do not make
   the TUI default based only on headless tests.
2. **Bounded service buffers / incremental acquisition:** choose based on actual use.
   Offline resource helpers exist; service buffer transfer and auto-cleanup do not.
   Retain the reviewed design: reserve a server-wide byte budget before allocation;
   buffers are temporary retained results, not consumer-dependent queues; unattended
   overflow fails; end-of-run transfer is bounded/acknowledged, discarding incomplete
   transfer on failure while retaining disk publications. Release/terminal cleanup
   never deletes files; automatic catalog cleanup must preserve known-folder wait.
3. **Adaptive acquisition / pipeline-result feedback:** define coordinates, admission,
   named required-result access and provenance first. Use pure decision functions
   with explicit configuration/prior state and validated proposed settings/stop;
   randomness needs explicit state. Review custom-policy code/version checks so a
   fresh controller rejects unavailable/changed policies before hardware use; logged
   source is never an execution fallback. Recorded general Python alone is not a
   validated adaptive policy API.
4. **Adaptive VNA milestone:** candidate detection → intermediate narrowing → reviewed
   peak/dip fit with magnitude/power/complex convention, bounds, fit quality,
   no-candidate/edge behavior and refinement limit. A ~2,000-point 1 GHz sweep samples
   a 1 MHz feature only ~twice: do not trust its linewidth fit. An intermediate
   ~20 MHz sweep may be needed before targeting final span ~2× linewidth.
5. **Observability/data extensions:** durable combined lab timeline, run report/events,
   settings history, plot subscriptions/reconnect, appendable data and whole-scan
   filtering. Label commanded/observed/unknown values honestly. Source provenance is
   best-effort human reference, never executable or full notebook history; expose
   capture failures and consider secrets. Current logs are not this audit system.
6. **Digitizers and broader migration:** select measured rates/chunks/channels/duration
   before buffer/transport/placement optimization. Keep high-rate data off inline
   instrument replies. Multi-input joins, custom worker admission, Holzworth-like
   multichannel backend validation, additional V1 drivers/analysis and QGL/qubit
   adapters need separate designs, not blanket porting.
7. **Optional ergonomic work:** recording-module notebook aliases/completion remain
   deferred (executed aliases differ from completion inside unexecuted methods;
   avoid implicit “currently recording” state). API navigation/signatures, richer
   capability schemas, authentication and device-specific safety/readback are not
   prerequisites retroactively imposed on the trusted-lab slice.

## Authorized slice — 2D offline DataBrowser

- [x] Support regular, non-ragged 2D heatmaps with monotonic nonuniform coordinates
  (X last dimension, Y first); leave 1D behavior unchanged.
- [x] Complex component modes: real, imaginary, magnitude, phase, stacked
  real/imaginary, stacked magnitude/phase (default), and four panels. Real data
  uses one heatmap. Shared colormap selector defaults to plasma; independent
  scales: symmetric real/imaginary, zero-based magnitude, phase ±pi, no unwrap.
- [x] Single-panel views have independent, initially disabled X/Y cuts with
  draggable lines snapping to stored coordinates and corresponding plots below.
- [x] Add correctness/integration tests and one complex 2D synthetic visual case.

- [x] User-reviewed follow-up: exclude CET- and PAL-prefixed colormaps from the
  dropdown, retaining plasma as default.

Implementation complete; user reports successful visual review. No further
implementation slice is authorized.

## Authorized slice — manual save_data

- [x] Export `save_data(filename, data, *, comments=None, force_write=False, **axes)`.
  Create one browser-ready `.auspex` folder and return its Path; axes follow keyword
  order, one per dimension. Accept numeric coordinates or Axis objects whose names
  match their keywords, preserving units. Save free-form comments in run metadata.
- [x] Reject existing destinations by default; explicit force replaces an existing
  store. Validate inputs before replacing and respect the store writer lock.
- [x] Document the helper and test round trips, validation and replacement.

Manual save_data is complete. No further implementation slice is authorized.

## Authorized slice — manual open_data

- [x] Export `open_data(*, path=None, number=None, date=None, group=None, dataset=None)`
  returning one loaded Dataset. With no path, show an optional-Qt OS folder picker;
  cancellation returns None. Explicit paths require no GUI dependencies.
- [x] Number lookup requires path and searches only immediate `<name>_N.auspex`
  folders, ignoring zero-padding. Date is None, `today`, or strict YYYY-MM-DD.
  Reject missing/ambiguous folders rather than guessing.
- [x] Select one published dataset, optionally filtered by group/dataset; reject
  missing or ambiguous matches. Document and test real-store loading and selection.

Manual open_data is complete. No further implementation slice is authorized.

## Authorized slice — live plotting

- [x] Share 1D/2D rendering with DataBrowser; preserve view/component/cut state on updates.
- [x] Add compiled trace/map plot endpoints, bounded per-run snapshots and local
  discovery. Partial averages feed traces; completed writer rows feed maps. Late
  clients receive complete acquired state. GUI loss never gates acquisition.
- [x] One three-pane client: newest runs left, tabs center, metadata right. New
  tabs do not steal focus. Closed tabs remain listed; Close plot removes/suppresses
  for this session. Retain terminal plots with a 64-plot cap: oldest terminal
  entries evicted first, reject new plots visibly if all retained entries active.
- [x] Add process-backed integration/correctness coverage and a simulated-pipeline
  visual demo. Document budgets, disconnection, publication and run status limits.

Live plotting is implemented; attended desktop review remains. No further
implementation slice is authorized. The writer owner retains bounded map state;
its private Unix endpoint serves replaceable full frames. `preview.json` provides
local discovery, with best-effort final `.preview/` caches for late viewing after
ordinary closure. These are display artifacts, not dataset publications or run
status. Initial execution is writer-backed, rank-one traces and one monotonic
fixed sweep per map; unsupported maps disable themselves, never required storage.

## Authorized slice — manual-helper MkDocs documentation

- [x] Add save_data/open_data examples and API contracts to the main data guide,
  including axis order/units, comments, destructive opt-in overwrite, GUI picker,
  numbered/date lookup and dataset selection. Correct its stale 1D-only browser
  description. Validate with a strict local MkDocs build.

Documentation is complete; the save/open and comment-reading examples were also
executed successfully in a temporary store. No further slice is authorized.

## Authorized slice — core numerical fits

- [x] Port V1 fits.py's AuspexFit/Auspex2DFit, LorentzFit, GaussianFit,
  MultiGaussianFit and QuadraticFit to auspex_v2.analysis. Preserve familiar
  fit_params/fit_errors/model interfaces; correct initializer/default-model bugs.
- [x] Expose covariance, residuals, squared error and degrees of freedom instead
  of unjustified Nsigma. Add explicit guesses/bounds; keep plotting opt-in with
  lazy Matplotlib imports and no global style changes. SciPy/Matplotlib only in
  an optional analysis extra; no acquisition dependency changes.
- [x] Use deterministic positive-peak multi-Gaussian seeding on actual coordinates,
  explicit guesses when unresolved, optional seeded multistart, center ordering
  with consistently reordered covariance. No scikit-learn.
- [x] Add independent numerical/integration tests and MkDocs usage/contracts.
  Resonator-circle and qubit fits are not part of this slice.

Core fits are implemented. Automatic single-peak seeds now use peak/dip geometry
rather than V1's defective Gaussian regression; explicit p0 bypasses seeding.
Optimizer parameter coordinates are centered/scaled internally for mixed physical
units; returned covariance/parameters remain in caller units. No Nsigma, silent
complex projection, weighted fitting, scikit-learn, or automatic model selection.
Resonator-circle fitting needs a separate scientific review before implementation;
qubit fits remain deferred. No further implementation slice is authorized.

## Authorized slice — aggregate installation extra

- [x] Add `auspex-v2[all]` / `pip install -e ".[all]"` to enable analysis, VISA,
  GUI and TUI extras, without changing base dependencies. README updated;
  requirement syntax and inclusion of every optional extra validated.

## Authorized slice — hanger resonator circle fitting

- [x] Add ResonatorCircleFit(frequency_hz, s21_complex): one hanger/notch S21,
  constant complex gain plus linear cable delay (seconds), phase at a reported
  reference frequency. Circle geometry initializes full original-complex-data
  refinement; no reflection/peak/general-background/qubit expansion.
- [x] Expose fr, loaded/internal Q, complex coupling Q, its magnitude and effective
  dissipative Q, gain/phase/delay, model/residuals and propagated uncertainties.
- [x] Raise on invalid input, failed convergence, out-of-span resonance and
  nonpositive/nonfinite Q. Warn, without repairing results, about weak constraints,
  sparse linewidth sampling and unreliable covariance; no unweighted chi-square.
- [x] Add independent synthetic/correctness and storage/plot integration tests,
  document conventions and operating limits in MkDocs. Real-trace validation is
  a separate attended lab step.

Hanger fitting is implemented. The convention is Qc = Qc_mag exp(-i phi),
Qc_eff = Qc_mag/cos(phi), 1/Qi = 1/Ql - Re(1/Qc); alpha refers to the measured
span midpoint. Seven-parameter covariance and derived real-valued Q covariance
are exposed; complex Qc has a two-component covariance, not a scalar +/- error.
Circle fitting replaces V1's determinant/Newton approach; final inference uses
original complex data. Delay initialization is local, not an alias-resolution
guarantee. No further implementation slice is authorized; qubit fits remain deferred.

## Authorized slice — driver exports and automatic admission

- [x] Each bundled vendor/simulator module declares intended driver classes in
  __all__. Discover those exports once at import, expose them through
  auspex_v2.instruments.drivers and auspex_v2.instruments for completion, and use
  the same catalog for InstrumentServer/ExperimentServer admission.
- [x] Reject duplicate/colliding names and invalid/base/abstract driver exports;
  no driver construction, hardware probing or eager optional backend loading.
  Discovery stays inside the bundled flat driver package, not a plugin system.
- [x] Test actual import completion, catalog validation and a new exported
  simulator through real server/worker use; document the single export contract.
  Actual V1 driver ports are not part of this slice.

Module exports now define both Python visibility and service admission, with no
separate manual registry. Imports are deterministic and fail on broken exports;
newly installed drivers require fresh Python/service processes, not hot reload.
No further implementation slice is authorized.

## Authorized slice — remaining V1 VISA drivers

Revised user-approved scope: retain the 11 completed experimental V1 VISA ports
(including VISA serial); create unavailable stubs for **all 16 remaining models**.
Those models require manual porting and hardware validation, not automatic repair.
Stubs raise NotImplementedError on construction and are absent from module
__all__, top-level exports and server admission. Exclude native USB/DLL, HTTP,
direct-socket-only and Prologix-specific backends. Do not recreate V1 metaclasses,
proxy configuration or dummy backends. Existing four lab-validated V2 drivers
remain unchanged.

- [x] Export `@experimental`: inherited class metadata, normal imports/admission,
  UserWarning on connection, no import-time warning or GUI distinction.
- [x] Verify decorator behavior, imports/catalog and real simulator/server admission;
  no mocked or hardware-driver behavior tests for these ports.
- [x] Port Rohde & Schwarz: SMB100B (`rs.py`).
- [x] Stubs only — Agilent: HP33120A, Agilent33220A, Agilent33500B, Agilent34970A,
  AgilentN9010A. Preserve existing N5183A/E8363C/N5230A unchanged.
- [x] American Magnetics: AMI430.
- [x] BBN: RelayDriver, DigitalAttenuator (not APS/APS2/TDM).
- [x] Stub only — BBN: SpectrumAnalyzer.
- [x] Berkeley Nucleonics: BNC845.
- [x] DS Instruments: DSInstrumentsSG12000, DSInstrumentsSG12000Pro, DAT64H;
  DSInstruments remains an implementation base, not a separate admitted model.
- [x] Keithley: Keithley2400.
- [x] Kepco: BOP2020M.
- [x] Stubs only — Keysight: KeysightM8190A (no waveform/sequence helper port).
- [x] Stubs only — Lake Shore: LakeShore370, LakeShore335.
- [x] Stub only — LeCroy: HDO6104.
- [x] Picosecond: Picosecond10070A.
- [x] Stubs only — Stanford: SR830, SR865.
- [x] Stubs only — Tektronix: DPO72004C, DPO2024, DPO2014, RSA3308A.
- [x] Document implemented ports versus unavailable stubs; check all 16 stub
  constructors and exclusion from exports/catalog, without hardware or mocks.

No Yokogawa work: GS200 is already ported and lab-validated. The inventory above
contains 27 model names: 11 implemented experimental ports and 16 unavailable
stubs. Stub classes are not usable experimental drivers.

**Resolved connection policy:** user approved transport/protocol-only connect;
no output enabling, reference/source-mode/unit/trigger-source changes. BNC845's
active-low pulse/mod/alc mappings are confirmed model limitations and preserved.
BNC845.configure_external_reference selects external 10 MHz and polls lock without
enabling RF; BOP2020M.configure_current_mode selects current mode and VOLT MAX
without enabling output. Existing settings cover explicit Tesla/GPIB selection.
Keithley sweep remains an explicit output-changing operation; restore its I/O
timeout on failure without issuing additional hardware recovery commands.
DS Pro invalid reference now raises instead of silently selecting EXT; DAT64H
uses one numeric attenuation declaration rather than V1's shadowing text property.

The authorized slice is complete; no further automatic porting is authorized.
Known concerns retained for future **manual** work (not an implementation queue):

- Agilent33500B.upload_waveform_binary explicitly says NOT YET WORKING/DO NOT USE;
  Tektronix.get_math_curve is a pass stub. Omit/raise NotImplementedError until
  separately reviewed, or attempt a repair within this port?
- LeCroy fetch_waveform writes COMM_ORDER HI (comment says MSB first) but decodes
  is_big_endian=False. Which wire order should the port use?
- LakeShore335.check_hconf_msg accepts heater-resistance codes 1/2 but documents
  0/1, and applies the 50-ohm limit based on vals[2] (current code), not vals[1]
  (resistance code). Need reviewed model limits rather than guessing protection.
- Agilent33220A.output_sync maps True to OFF and False to ON. Confirm whether this
  is another intentional model quirk (as with BNC) or should be corrected.
- BBN SpectrumAnalyzer.get_voltage retries ValueError/VisaIOError ten times and
  returns None after exhaustion. Decide whether to retain this explicit read-only
  retry behavior or use V2's ordinary single-attempt/propagated-error behavior.

All 16 remaining models are deliberately stubbed, including models without a
specific identified defect. These notes do not authorize repairing or admitting them.

## Authorized slice — nonfatal InstrumentServer announcements

- [x] Make Registry.announce console/file output independently best effort for
  OSError/UnicodeError, without changing dispatch, persistence or lease policy.
- [x] Retire failed stdout to /dev/null so buffered output cannot break subsequent
  worker spawning or interpreter shutdown. Do not introduce asynchronous logging.
- [x] Verify lost stdout with a real simulator/server: commands/readback, live
  ownership, disconnection/reconnection and ordinary worker-failure quarantine;
  check file/console sink independence separately.

Complete. A connected but undrained output pipe can still block; changing that is
outside this slice. Other review findings (Cartesian-product scan allocation and
finished-store EOF handling) remain unfixed and require separate authorization.

Runner lifecycle refactoring remains deferred. Keep current evidence in STATUS.md,
not a growing chronological test-count/change diary here.
