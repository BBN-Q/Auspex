"""Offline saved-plan/filter integration harness, not a live hardware runner.

Commands are injected Python callbacks and WAIT is virtual. There is no instrument
lease, service supervision, cancellation API, or durable execution log here.
"""
from .fake import FakeExecutor
from ._experiment_plan import walk_instructions
from ..filters import ManualPipeline, PipelineError
from ..filters.definitions import CompiledPipeline


class PipelineExecutor(FakeExecutor):
    """Execute a v2 plan into one real filter pipeline using fake instrument results.

    Teardown is attempted before waiting for final processing/storage. It stops on
    error. Completed scans can still publish if teardown failed; a failed partial
    scan is cancelled. This is an integration test harness, not ExperimentServer.
    """
    def run(self, plan, *, store_path=None, **pipeline_options):
        """Admit supported topology before startup, then finalize after teardown."""
        document = plan.document
        attachments = document.get("pipelines", [])
        if document["version"] != 2 or len(attachments) != 1:
            raise PipelineError("Offline pipeline runner requires one attached v2 pipeline")
        for phase in ("setup", "main", "teardown"):
            if any(node["op"] == "ACQUIRE" for node in walk_instructions(document[phase])):
                raise PipelineError("Legacy ACQUIRE has no publication adapter; use CALL and explicit PUBLISH")
        attachment = attachments[0]
        compiled = CompiledPipeline.from_document(attachment["pipeline"])
        compiled.runtime()  # Reject multi-input/empty recipes before opening storage.
        output = next(iter(attachment["connections"].values()))
        pipeline = ManualPipeline(compiled, store_path=store_path, **pipeline_options)
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
            pipeline.cancel()  # Idempotent after normal finalization; cleans up unexpected errors.
            self._active_pipeline = None

    def _scan_end(self, node):
        """Close the scan before teardown, but defer completion waits until afterward."""
        self._active_pipeline._seal_scan()
        super()._scan_end(node)
