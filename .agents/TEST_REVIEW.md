# Auspex V2 test audit

## Implementation follow-up — synchronous pipeline retired

Removed core.Pipeline, Collector/DatasetWriter (filters/data.py), exports and
obsolete documentation. Shared contracts, numerical processors and the production
runtime remain unchanged. The ten tests in test_filters.py became five direct
contract tests, without constructing another executor:

| Original scenario | Retained evidence |
| --- | --- |
| Five simulator sweeps | Direct distinct-trace analytical means, partial counts/axes, final result and GroupComplete; graph simulator smoke remains. |
| Independent group sizes | Existing graph fanout, augmented with both branches' group-ID/window assertions. |
| Completed sweeps/average storage | Existing real scan-store average/publication tests; per-window dataset naming was retired with the old writer. |
| Grid mismatch/preview | Direct processor rejection preserving the previous snapshot; no obsolete executor sticky-failure assertion. |
| Acquisition-assigned group | Direct WindowState rejection. |
| Incomplete group/bad ordering | Existing real pipeline incomplete-group tests plus direct WindowState bad-window rejection. |
| Configuration boundary | Direct processor rejects mid-group changes, then emits a new three-window group; production local-revision tests remain. No global acquisition-revision rule is imposed. |
| Writer input ownership | Existing two-writer scan-store test now mutates both submitted arrays before window/scan completion and checks original data/coordinates in both saved destinations. |
| Coordinate memory budget | Existing direct schema test unchanged. |
| Invalid graph/schema/preview failure | Existing graph/definition checks, augmented with schema mismatch, plus existing real optional-preview failure coverage. Collector callback behavior is retired. |

All **388 tests passed** in 85.145 seconds. Documentation now directs callers to
PipelineDefinition with ManualPipeline/FilterServer; no compatibility executor or
wrapper was introduced. Historical notes retaining synchronous APIs below are
superseded by this implementation follow-up.

## Implementation follow-up — standalone FilterWorker retired

Removed the standalone FilterWorker API/private loop and exports. FILTER_TYPES
moved to server.py; the production `_graph_worker`, SharedEdge and synchronous
reference executor/sinks are unchanged. Documentation now directs manual callers
to ManualPipeline or FilterServer, including a single-processor graph.

The nine remaining tests in test_filter_worker.py were handled as follows:

- SharedEdge capacity/reuse moved unchanged to test_transport.py.
- Averaging metadata/axes moved to existing graph fanout and simulator checks;
  preview counts remain covered by live-preview tests.
- Undrained previews remain covered by the graph simulator and live-preview tests.
- Premature window completion now tests public graph ingress. The removed worker
  validated boundaries inside its child; production graph internal edges must
  support empty windows from sparse processors, so injecting a raw empty boundary
  is not an equivalent invalid operation. No runtime behavior was changed.
- Passthrough copy retention assertions were added to the bounded-overlap test.
- Worker death and configuration/incomplete-group failures already have graph tests.
  The standalone API's requirement to manually consume group/window messages
  before configure is intentionally retired, not imposed on FilterServer.
- Coordinate mismatch remains covered by the definition-backed ManualPipeline
  test for changed grids and the synchronous processor tests.

Eight standalone-worker tests were removed and one graph-ingress test added;
shared transport coverage was relocated, not removed. **393 tests pass** in
83.726 seconds. The first run revealed the invalid raw-boundary test migration
assumption above; the corrected full run passed. Synchronous Pipeline/Collector/
DatasetWriter retirement is outside this slice.

## Implementation follow-up — lifecycle evidence migrated

Removed the four PipelineExecutor lifecycle tests named in finding 1 below:

- Invalid publication now runs as a subcase of the real server measurement-failure
  test, checking actual teardown, no published datasets and lease release. The
  existing driver-failure/quarantine scenario remains a separate subcase.
- The real server teardown-failure test now also checks that the following cleanup
  command is not attempted and the lease remains unreleased; its completed-scan
  retention assertion remains.
- One focused test calls production `runner.run()` and records seal, hardware
  teardown, lease release, pipeline drain and terminal notification in that order.
  It covers both successful drain and an injected completion error. Only I/O
  boundaries are mocked, not the production interpreter/lifecycle. This test does
  not claim to reproduce real incomplete averaging: the existing ManualPipeline
  and scan-store tests retain that coverage.

The suite now has **400 tests**, all passing in 84.355 seconds. No production code
or executor implementation changed. API retirement and further cleanup remain
outside the authorized slice. Original audit findings below are historical.

## Implementation follow-up — failure fixtures corrected

The existing nonfatal logging test now causes a real log-open failure by creating
an execution.log directory before runner spawn; it retains the production logger
and checks success, lease release and its diagnostic. The existing truncation
test now removes one payload byte from a valid dataset, verifies the unchanged
header remains readable and checks the specific truncated-array error. Its
incompatible-version check remains. All 403 tests passed in 88.934 seconds, with
no new test methods or production changes. The audit findings below are retained
as the original review baseline.

## Implementation follow-up — first eight candidates complete

The eight candidates in the consolidation table below have now been consolidated.
The original audit and counts are retained below as the review baseline, not the
current inventory. Suite count is now **403**, with all tests passing headless in
85.882 seconds. No production code or public API changed in this slice.

- Preserved definition-backed incomplete-average context-exit coverage in the
  manual pipeline suite and retained the stronger backpressured store-EOF test.
- Moved expected counts and coordinate assertions into the two-window worker test.
- Moved post-publication plan-identity metadata checking into the real server run.
- Retained the production runner FFT test and fresh-process driver import check;
  removed their redundant fake-backend subprocess/vendor-reload counterparts.
- Moved command-only TUI routing assertions into the shared plain-command test.
- Folded regular named-axis/coordinate round-trip assertions into the dual-reader
  test that also prohibits serialization during reads.

Lifecycle-harness and API retirement remain deferred. The two failure-test
fixtures were subsequently corrected as recorded above. No additional
consolidation is authorized.

## Scope and result

Reviewed all **411 test methods in 49 `test*.py` modules** under `tests/`, including
assertions, fixtures and the test-only executors. Followed production references
for suspected obsolete paths. This is a source/contract audit, not mutation
analysis or proof that every remaining assertion catches a distinct defect.
V1's separate historical tests and attended visual scripts are not part of the
411-test suite; visual-script integration tests are included.

The preceding full run passed all 411 tests in 88.681 seconds. No tests or
production code were changed during this audit, and the unchanged suite was not
rerun. Counts below were independently checked from Python syntax trees.

| Area | Modules | Tests |
| --- | ---: | ---: |
| Analysis | 2 | 20 |
| Data/browser | 5 | 33 |
| Experiment/operator interfaces | 21 | 163 |
| Filters | 9 | 93 |
| Instruments | 12 | 102 |
| **Total** | **49** | **411** |

Most tests have a defensible contract. The suite is not simply 411 independent
integration scenarios: some methods are tiny checks, others contain many cases.
Reducing the method count by merging unrelated assertions would not meaningfully
reduce maintenance. The useful reductions are repeated workflows and redundant
execution implementations.

## Main findings

### 1. Several lifecycle assertions test a second, test-only runner

`tests/experiment/fake_executor.py:27` implements phase sequencing and teardown
in `FakeExecutor.run()`. `pipeline_executor.py:19` independently implements
pipeline finalization, partial-scan cancellation and teardown/publication policy.
Neither executes production `experiment.runner.run()`.

The expression/control-flow tests using FakeExecutor are valuable: `_eval`,
`execute`, parameter updates and loops are inherited from the production
Interpreter. **Do not remove FakeExecutor wholesale.** But its lifecycle tests
cannot establish the production runner's cleanup guarantees.

Review these four tests in `test_experiment_publication.py` for migration or
removal of duplicated lifecycle assertions:

- `test_scan_is_sealed_before_teardown_and_awaited_afterward`
- `test_bad_data_fails_and_attempts_teardown_without_publishing`
- `test_teardown_failure_stops_cleanup_but_retains_complete_scan`
- `test_teardown_precedes_wait_for_required_group_completion`

Production server tests already exercise measurement failure, cancellation,
teardown failure with retained publication, startup failure and lost supervision.
Not all assertions are exact substitutes: seal/teardown/wait ordering and stopping
at the first teardown error should be preserved in an existing real-runner test
if those fake-runner tests are retired. Buffer overflow remains a valid manual
pipeline contract even though server submission rejects buffer sinks; it should
be exercised through ManualPipeline rather than attributed to a production
server run.

The same caveat applies to cleanup assertions in compiler/authoring/recorded-
Python tests. Keep their distinct compiler/interpreter cases, but do not count
the FakeExecutor's `finally` block as evidence for live teardown.

### 2. Older filter execution paths are maintained primarily by their tests

Repository reference searches show:

- `core.Pipeline`, `Collector` and `DatasetWriter` are used by
  `test_filters.py`, not the current ManualPipeline/ExperimentServer runtime.
- `FilterWorker` and its `_worker` loop are used by `test_filter_worker.py`.
  FilterServer actually spawns `server._graph_worker`.
- `server.py` still imports `FILTER_TYPES` from `worker.py`; therefore deleting
  that module outright would break the current runtime.
- `SharedEdge` is not obsolete: `GraphEdge` subclasses it. Its direct slot
  ownership/reuse test should remain.
- `PortSchema`, `WindowState`, `Passthrough` and `ComplexAverager` remain active
  runtime contracts even though some tests access them through the old executor.

These are **retirement candidates, not proven dead public APIs**: the older
classes are exported and documented in the source filter guide. External/manual
users cannot be ruled out by searching this repository. If those APIs remain
supported, their separate tests still have a purpose because they execute
separate implementations.

A separately approved retirement could remove the alternate executors and
rewrite the useful synchronous tests as direct numerical/ordering checks of
ComplexAverager and WindowState. It could eliminate most of the nine
FilterWorker-specific tests (retaining the tenth, SharedEdge's direct test).
Do not simply delete tests while retaining these publicly supported paths.

`RetainedRunBuffers` has only two test callers and no service integration. This
is explicitly documented deferred work, not accidentally unreachable code.
Either retain it and those small tests, or retire both together after a scope
decision. The tests do not establish that server-backed buffer transfer exists.

### 3. One failure test replaces the very behavior it claims to verify

`test_experiment_server.py:399`, `test_log_write_failure_is_nonfatal`, patches
`Registry._log` with a function that sets a diagnostic and returns normally.
It never causes an execution-log write to fail and bypasses the production
exception handler entirely. It would still pass if `_log` stopped catching
OSError.

**Recommendation:** revise that existing test to cause a real execution.log open/
write failure while retaining the actual `_log` implementation. Do not add a
second test or retain the current one as failure-path evidence.

Two smaller overclaims:

- `test_data.py:236`, `test_reject_version_and_truncated_array`, replaces the
  entire dataset with `b"bad"`. It tests invalid magic, not a truncated array
  behind a valid header. Correct the fixture or rename the claim.
- `test_scan_store.py:151`, `test_writer_and_required_result_both_gate_scan_completion`,
  drains one unconstrained result and waits successfully. It does not demonstrate
  that either endpoint independently gates completion. The finished-store EOF
  regression immediately following it does, with real backpressure.

## Concrete consolidation candidates

These are proposals, not deletions already made. Where noted, move a small unique
assertion into the retained test before removal.

| Candidate | Retained coverage / action |
| --- | --- |
| `test_pipeline_definitions.py::test_incomplete_average_is_failure` | Same one-trace/two-repeat/context-exit failure as `test_manual_pipeline.py::test_incomplete_group_fails_normal_exit`. Keep one definition-backed ManualPipeline case; compiler topology and successful definition execution are already separately covered. |
| `test_scan_store.py::test_writer_and_required_result_both_gate_scan_completion` | Superseded by `test_finished_writer_eof_allows_a_paused_required_consumer_to_drain`, which checks real publication, blocked required results, ordered drainage and eventual completion. |
| `test_filter_worker.py::test_five_sweep_vna_result_from_spawned_filter` | Repeats the two-window worker averaging test plus ordinary storage round trips and `test_filter_server.py::test_simulated_vna_smoke`. If retaining FilterWorker, fold the expected-count/axis checks into its smaller averaging test; retain one production-DAG simulator smoke. |
| `test_experiment_resources.py::test_preallocated_folder_accepts_pipeline_writer_without_overwriting_metadata` | Repeats ManualPipeline's explicit append handoff and the real server saved-plan/publication run. Preserve the plan-identity metadata assertion in the real server test. Avoid another PipelineExecutor integration just for this handoff. |
| `test_recorded_python.py::test_saved_fft_executes_in_a_fresh_process_without_authoring_class` | The real runner FFT integration already loads a persisted plan in a fresh runner and checks the analytical result and identity. Retain the in-process COMPUTE-specific compile/manifest assertions; this extra fake-backend subprocess is not needed. |
| `test_instruments.py::test_import_does_not_import_pyvisa` | Subsumed by `test_driver_exports.py::test_imports_do_not_require_visa_and_completion_lists_exported_names`, which imports the driver catalog in a fresh process with PyVISA unavailable. The old test reloads vendor modules in the shared interpreter, unnecessarily replacing class identities while the registry retains old classes. Remove this in-process reload test. |
| `test_launcher_tui.py::test_instrument_ids_and_experiment_requests` | Does not run Textual or await anything: it tests shared OperatorCommands with fake clients. Keep genuinely UI-specific tests, and consolidate its useful `runs`/mode dispatch assertions into the shared command tests instead of an async GUI test. |
| `test_data.py::test_named_lookup_round_trips_regular_saved_data_and_standalone_files` | Merge named-lookup identity/unit assertions into the existing dual-reader/read-without-serialization test or regular store round trip. Keep dependent/ragged lookup separately; their semantics differ. |

The four fake-runner lifecycle tests above are an additional consolidation area,
not four unconditional deletions. The expected first pass is a **modest reduction
of roughly 8–12 test methods**, with some assertions moved, rather than an
arbitrary target such as halving the suite.

Additional trimming within tests, without deleting useful scenarios:

- Consolidate obsolete-plan version matrices in `test_experiment_compiler.py`;
  other tests need not repeat rejection of version 4/5 merely because those
  features arrived at that version. Keep one file-load case and complete direct
  document validation, including the immediately previous version.
- `test_interpreter_boundary.py`'s import dependency check is valuable. Exact
  method identity and absence of arbitrary instance attributes are refactor-
  sensitive; retain only what is needed to prohibit duplicated interpretation
  and production dependencies on test infrastructure.
- Exact numbered instruction IDs in `test_experiment_review.py` need not be a
  separate implementation-shape contract if deterministic saved identity and
  usable instruction diagnostics remain verified. Its read-only-validation
  assertion has independent value.
- Repeated literal format-version, palette-color, export-identity and default
  values can live in one relevant contract test rather than every integration.
  Do not remove behavioral assertions just to lower assertion counts.
- The demo-generator test's numerical-noise thresholds are low-value compared
  with its actual-store creation, reuse and refusal-to-overwrite checks. Retain
  an executable visual-demo smoke, not a detailed specification of demo noise.

## Coverage that should remain separate

- **Numerical correctness versus transport integration:** independent fit models,
  covariance checks, byte-identical scan layout comparisons, and shared-memory
  ownership tests fail for different reasons. Similar data values do not make
  these duplicates.
- **Graph cancellation versus worker death:** the trigger and wakeup paths differ;
  blocked producer and blocked consumer cases are meaningful.
- **Normal store EOF versus unacknowledged store death:** opposite completion
  policies, both essential.
- **Scan-end incomplete groups versus run-end incomplete groups:** ordered
  boundaries have different contracts, even when they reject the same average.
- **GUI versus server ownership restrictions:** server tests prove enforcement;
  GUI tests prove the operator is offered the right actions and retains feedback.
  Keep widget behavior, but avoid retesting shared command dispatch without UI.
- **Plain SIGINT shutdown versus Textual Ctrl+C:** separate frontend paths.
  Retain first/second interrupt behavior in both; do not delete TUI lifetime/output
  checks because plain-launcher shutdown succeeds.
- **Save/open helpers versus the underlying store:** keyword axis order,
  overwrite-before-validation, filename/date resolution, ambiguity and Qt
  optionality are adapter contracts that storage round trips do not cover.
- **Driver declarations versus service admission:** import-only checks cannot
  replace a real worker instantiated from a newly exported vendor module.
- **Different uncertainty failures:** setup never started, teardown failed,
  supervision lost, instrument worker failed, and status persistence failed are
  not interchangeable failure tests.

## Complete module disposition

Names are relative to each area under `tests/`. “Keep” means no whole-test removal
identified, not a guarantee that every assertion is indispensable.

### Analysis — 20

| Module | Tests | Disposition |
| --- | ---: | --- |
| `analysis/test_fits.py` | 12 | Keep: independent analytical references, physical units, covariance, seeded mixtures, validation and real plotting/storage. |
| `analysis/test_resonator_fits.py` | 8 | Keep: independent loss convention, circle geometry, primary/derived covariance, diagnostics and physical failures. Not duplicates of generic fits. |

### Data — 33

| Module | Tests | Disposition |
| --- | ---: | --- |
| `data/test_data.py` | 12 | Keep core contracts; merge regular named-lookup round trip; repair/rename truncated-array claim. Keep chunk validation and unsigned ragged regression. |
| `data/test_data_browser.py` | 7 | Keep distinct folder/tab/tree/watch behaviors. Simplify demo-data assertions; watched-folder tests overlap in setup but differ in multi-root, nonrecursive and pinned-run behavior. |
| `data/test_heatmap.py` | 2 | Keep: real non-square/nonuniform complex rendering, cuts and unsupported grids. |
| `data/test_open_data.py` | 7 | Keep: resolution, ambiguity, detached loading, optional picker and ragged pass-through. Picker boundary mock is appropriate. |
| `data/test_save_data.py` | 5 | Keep: axis order, units, replacement, pre-deletion validation, writer lock and unrelated-folder protection. |

### Experiment/operator interfaces — 163

| Module | Tests | Disposition |
| --- | ---: | --- |
| `experiment/test_auspex_paths.py` | 2 | Keep: shared service root and explicit fallback explanation. |
| `experiment/test_experiment_authoring.py` | 22 | Keep compiler/parameter/proxy cases; distinguish test-only cleanup assertions from real lifecycle evidence. |
| `experiment/test_experiment_compiler.py` | 13 | Keep sweep order, explicit wait semantics, loops and availability; consolidate old-version repetitions. |
| `experiment/test_experiment_proxy.py` | 6 | Keep: syntax equivalence, no construction, permissions, kwargs and detached recording lifetime. |
| `experiment/test_experiment_publication.py` | 19 | Keep wiring/publication compiler contracts; consolidate four fake-runner lifecycle cases as discussed. |
| `experiment/test_experiment_resources.py` | 8 | Keep allocation/admission/failure numbering; consolidate append handoff; two retained-buffer tests depend on preserving deferred helper API. Browser import smoke is cheap and still checks a supported alias. |
| `experiment/test_experiment_review.py` | 12 | Keep distinct initialization/mutation/type cases; trim exact implementation-shape assertions. Historical filename is not evidence of redundant behavior. |
| `experiment/test_experiment_server.py` | 18 | Keep production lifecycle integration; repair ineffective log-write-failure injection. |
| `experiment/test_interpreter_boundary.py` | 3 | Keep dependency direction and explicit unsupported backend actions; reduce brittle identity/attribute checks if restructuring. |
| `experiment/test_launcher.py` | 5 | Keep real startup, duplicate launch, logs/restart, shutdown and unexpected service exit. Duplicate-launch setup appears twice but log preservation adds a distinct contract. |
| `experiment/test_launcher_tui.py` | 4 | Keep actual UI routing/confirmation tests; move/remove the command-only async test. |
| `experiment/test_operator_commands.py` | 2 | Keep shared command/context policy and plain adapter confirmation; natural destination for command-only TUI assertions. |
| `experiment/test_operator_status.py` | 4 | Keep bounded unavailable-service behavior, blocker display and CLI JSON. |
| `experiment/test_parameter_setters.py` | 11 | Keep: recorder association/inheritance/recursion/phase/detachment contracts differ from simple binding. Defaults and validation may share fixtures with authoring but need both association forms. |
| `experiment/test_plot_client.py` | 3 | Keep retention policy, renderer update state and executable visual-client demo. These are not just repeats of offline rendering. |
| `experiment/test_progress.py` | 5 | Keep point semantics and totals; trim duplicate obsolete-version assertion. |
| `experiment/test_real_experiment_examples.py` | 2 | Keep hardware-free authoring checks of distinct bench examples; simulator runs cannot validate their specific setup instruction order. |
| `experiment/test_recorded_python.py` | 14 | Keep COMPUTE-specific contracts; remove redundant fresh-process fake FFT; move any unique assertions before removal. |
| `experiment/test_service_output.py` | 3 | Keep real-pipe flood, startup readiness and inherited-EOF budget cases. |
| `experiment/test_terminal_logging.py` | 3 | Keep persistence/message relationship and nonfatal output; prefix-only assertion can be folded into the existing status/logging scenario, but savings are negligible. |
| `experiment/test_tui_shutdown.py` | 4 | Keep real UI lifetime, first/second Ctrl+C, inherited stdout and idle shutdown. |

### Filters — 93

| Module | Tests | Disposition |
| --- | ---: | --- |
| `filters/test_filter_inspection.py` | 3 | Keep acknowledged configuration snapshots, private socket ownership and GUI refresh/geometry. |
| `filters/test_filter_server.py` | 11 | Keep production DAG fanout, capacity, cancellation, failure wakeups, local revisions and simulator smoke. |
| `filters/test_filter_worker.py` | 10 | Retain SharedEdge test. Remaining nine are tied to older public single-worker runtime; merge repeated five-sweep smoke now, retire others only with that API. |
| `filters/test_filters.py` | 10 | Useful direct contract/numerical cases, currently coupled to synchronous reference executor. Rewrite targeted cases if retiring that executor; do not remove shared schema/window-state coverage. |
| `filters/test_live_preview.py` | 7 | Keep lossy display versus required storage, late attach, partial/final distinction, cache and isolated preview failure. |
| `filters/test_manual_pipeline.py` | 13 | Keep standalone entry-point, buffers, handoff, cancellation and ownership semantics. Some setup repeats store tests but return-value/lifetime and context-exit contracts are distinct. |
| `filters/test_pipeline_definitions.py` | 20 | Keep inert authoring/schema/serialization/lowering. Consolidate duplicate incomplete-average execution. |
| `filters/test_scan_assembly.py` | 9 | Keep independent layout/encoding, dtype/endian/empty-row cases, bounded allocations and failure cleanup. These are stronger than shape-only end-to-end smoke tests. |
| `filters/test_scan_store.py` | 10 | Keep process owner, multiple writers/scans, sparse emissions, capacity and EOF/death; remove superseded unconstrained result/writer gate smoke. |

### Instruments — 102

| Module | Tests | Disposition |
| --- | ---: | --- |
| `instruments/test_bulk.py` | 4 | Keep framing, byte order, unsupported payloads and truncated frames; real VNA IPC tests cover a different layer. |
| `instruments/test_discovery.py` | 12 | Keep discovery ownership, inert enumeration, hints, whitelist/filter and shared terminal policies. Fake VISA is appropriate for deterministic no-hardware probes. |
| `instruments/test_discovery_gui.py` | 10 | Keep real UI/IPC behaviors: selection preservation, cancellation, queue stopping, feedback isolation and filter state. Trim repeated service-policy assertions rather than delete widget scenarios. |
| `instruments/test_driver_exports.py` | 6 | Keep real import/admission contract, fresh imports, isolated new vendor and IPython completion. |
| `instruments/test_driver_stubs.py` | 2 | Keep: absence from admission and failure on direct construction are distinct promises. Sixteen subcases are only two test methods. |
| `instruments/test_experimental.py` | 4 | Keep warning timing/inheritance and intended experimental model list; redundant generic export-identity assertions can be trimmed. |
| `instruments/test_instrument_cli.py` | 8 | Keep frontend child lifetime, optional dependencies, flag forwarding and shutdown/confirmation policy; mocks are concentrated at process/UI boundaries. |
| `instruments/test_instrument_gui.py` | 1 | Keep inventory-only actions, gating and persistent feedback. One scenario already covers many related transitions. |
| `instruments/test_instrument_server.py` | 20 | Keep real ownership, quarantine, recovery, IPC arrays, restart and failure paths. Closed-stdout regression overlaps the happy path intentionally to establish nonfatal output. |
| `instruments/test_instruments.py` | 22 | Keep command/driver SCPI contracts; remove redundant module-reload lazy-import test. Existing four driver tests are not validation of the eleven experimental ports. |
| `instruments/test_registry_lifecycle.py` | 6 | Keep deterministic blocked lifecycle/late-reply and failed reconciliation cases; real IPC tests do not reliably force those races. Sink-independence test is complementary to real closed-stdout integration. |
| `instruments/test_visa_addresses.py` | 7 | Keep explicit alias equivalence, endpoint/ownership agreement and rejected-inventory preservation. Normalization unit tests do not replace integration with registration/discovery. |

## Suggested next slice

1. Repair the ineffective logging-failure test and the misleading truncated-array
   fixture/claim, without adding test methods.
2. Consolidate the eight named duplicate/overlap candidates, preserving the few
   unique assertions where specified.
3. Decide separately whether synchronous Pipeline/FilterWorker and deferred
   RetainedRunBuffers remain supported. Their retirement is an API/code decision,
   not test deletion justified by a count.
4. Address test-only lifecycle duplication without building a new elaborate
   harness: reuse existing real-runner integrations and keep FakeExecutor focused
   on deterministic instruction semantics.

No coverage percentage or target test count is proposed. The aim is less repeated
policy and more trustworthy evidence, not fewer green dots.
