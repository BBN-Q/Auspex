"""Offline saved-plan/filter integration harness, not a live hardware runner.

Commands are injected Python callbacks and WAIT is virtual. There is no instrument
lease, service supervision, cancellation API, or durable execution log here.
"""
from tests.experiment.fake_executor import FakeExecutor
from auspex_v2.experiment._experiment_plan import PLAN_VERSION
from auspex_v2.filters import ManualPipeline, PipelineError
from auspex_v2.filters.definitions import CompiledPipeline


class PipelineExecutor(FakeExecutor):
    """Execute a current-version plan into one filter pipeline using fake instrument results.

    Teardown is attempted before waiting for final processing/storage. It stops on
    error. Completed scans can still publish if teardown failed; a failed partial
    scan is cancelled. This is an integration test harness, not ExperimentServer.
    """
    def run(self, plan, *, store_path=None, **pipeline_options):
        """Admit supported topology before startup, then finalize after teardown."""
        document = plan.document
        attachments = document.get("pipelines", [])
        if document["version"] != PLAN_VERSION or len(attachments) != 1:
            raise PipelineError("Offline pipeline runner requires one attached current-version pipeline")
        attachment = attachments[0]
        compiled = CompiledPipeline.from_document(attachment["pipeline"])
        compiled.runtime()  # Reject multi-input/empty recipes before opening storage.
        output = next(iter(attachment["connections"].values()))
        pipeline_options.setdefault("buffer_full", "fail")
        pipeline = ManualPipeline(compiled, store_path=store_path, **pipeline_options)
        previous_retention = self.retain_windows
        self.retain_windows = False
        self._active_pipeline = pipeline
        previous_publisher = self.publisher

        def submit(name, data, axes, coordinates):
            if name != output:
                raise PipelineError(f"Output {name} is not routed to the active pipeline")
            units = {key: document["parameters"][key]["unit"] for key in coordinates}
            pipeline.submit(data, axes=axes, sweep=coordinates, sweep_units=units)

        self.publisher = submit
        self.buffers = pipeline.buffers
        try:
            super().run(plan)
            # A normal early termination closes its admitted prefix; incomplete
            # required filter groups still fail. A failed partial scan is discarded.
            if self.outcome == "SUCCEEDED" or self.scan_complete:
                try:
                    pipeline.close()
                except Exception as exc:
                    self.outcome = "FAILED"
                    self.reason += f"; pipeline finalization: {exc}"
            else:
                pipeline.cancel()
            self.trace.append(f"{self.time:.6f} PIPELINE_FINISHED {self.outcome} {self.reason}")
            return self
        finally:
            self.publisher = previous_publisher
            self.retain_windows = previous_retention
            pipeline.cancel()  # Idempotent after normal finalization; cleans up unexpected errors.
            self._active_pipeline = None

    def _scan_end(self, node):
        """Close the scan before teardown, but defer completion waits until afterward."""
        self._active_pipeline._seal_scan()
        super()._scan_end(node)
