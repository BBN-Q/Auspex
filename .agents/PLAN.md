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

The handoff-note cleanup is complete. No implementation slice is currently authorized;
runner lifecycle refactoring remains deferred. Keep current evidence in STATUS.md,
not a growing chronological test-count/change diary here.
