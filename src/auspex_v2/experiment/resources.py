"""Run-folder and volatile buffer ownership primitives shared with ExperimentServer.

The daemon and runner live in service/runner; automatic folder discovery is not
implemented. These helpers make the agreed handoffs explicit and testable.
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tempfile
import threading

from ..data import AuspexStore
from ..data.data import _component
from ..filters.buffer import snapshot
from ..filters.core import PipelineError
from ..filters.definitions import CompiledPipeline
from ..filters.server import FilterServer, ResultSpec
from ..instruments.server.bulk import MAX_BULK_BYTES
from .compiler import CompiledPlan
from ._experiment_plan import PLAN_VERSION, swept_parameters


def _atomic_json(path, value):
    """Replace one control file, syncing its contents and directory entry."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".run-", delete=False,
                                         mode="w", encoding="utf8") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preflight_run(plan, *, slots=2, result_capacity=2,
                  memory_budget_bytes=64 * 1024 * 1024,
                  store_scan_budget_bytes=256 * 1024 * 1024):
    """Check supported saved topology and graph budgets before claiming a folder.

    No workers, store or instrument session are created here. Recorded Python
    imports are trusted code and may have side effects. Runtime failures remain possible.
    """
    if not isinstance(plan, CompiledPlan) or plan.document["version"] != PLAN_VERSION:
        raise ValueError("Preflight requires a validated current-version plan")
    from .computations import resolve_operation
    for spec in plan.document["operations"].values():
        resolve_operation(spec)
    attachments = plan.document["pipelines"]
    if len(attachments) != 1:
        raise PipelineError("Current runner requires one attached pipeline")
    compiled = CompiledPipeline.from_document(attachments[0]["pipeline"])
    schema, nodes, edges, writers, buffers, _ = compiled.runtime()
    if writers and (set(schema.axis_names) & swept_parameters(plan.document) or "point" in schema.axis_names):
        raise PipelineError("Writer sweep and measurement coordinate names must be distinct (point is reserved)")
    for name, spec in buffers.items():
        if nodes[spec.node].schema.max_bytes > MAX_BULK_BYTES:
            raise PipelineError(f"Buffer {name!r} exceeds the inline transfer limit")
    results = {name: ResultSpec(spec.node, spec.port) for name, spec in buffers.items()}
    FilterServer.validate_graph(schema, nodes, edges, results, writers, buffers,
                                slots=slots, result_capacity=result_capacity,
                                memory_budget_bytes=memory_budget_bytes,
                                store_scan_budget_bytes=store_scan_budget_bytes)
    return compiled


@dataclass(frozen=True)
class RunFolder:
    """A claimed .auspex directory with a saved plan and server-owned control files."""
    path: Path

    @classmethod
    def allocate(cls, plan, *, name, directory):
        """Claim a never-overwritten numbered folder; return only after ACCEPTED."""
        if not isinstance(plan, CompiledPlan) or plan.document["version"] != PLAN_VERSION:
            raise ValueError("Run allocation requires a validated current-version plan")
        name = _component(name)
        parent = Path(directory).expanduser().resolve()
        if not parent.is_dir():
            raise FileNotFoundError(parent)
        pattern = re.compile(re.escape(name) + r"_(\d{3,})\.auspex\Z")
        numbers = [int(match.group(1)) for entry in parent.iterdir()
                   if (match := pattern.fullmatch(entry.name))]
        number = max(numbers, default=-1) + 1
        while True:
            path = parent / f"{name}_{number:03d}.auspex"
            try:
                with AuspexStore(path, mode="w"):
                    _atomic_json(path / "status.json", {"state": "UNINITIALIZED", "run_id": path.stem})
                break  # Manifest, stub and exclusive store lock are initialized.
            except FileExistsError:
                number += 1
        run = cls(path)
        try:
            _atomic_json(path / "plan.json", plan.document)
            with AuspexStore(path, mode="a") as store:
                store.write_run_metadata({"name": name, "plan_identity": plan.identity,
                                          "experiment": plan.document["metadata"]})
            _atomic_json(path / "status.json", {"state": "ACCEPTED", "run_id": path.stem,
                                                 "plan_identity": plan.identity})
        except Exception as exc:
            try:
                _atomic_json(path / "status.json", {"state": "FAILED", "run_id": path.stem,
                                                     "reason": f"Run initialization failed: {exc}"})
            except OSError:
                pass  # UNINITIALIZED remains inspectable if disk I/O itself failed.
            raise
        return run

    def status(self):
        """Inspect a saved run by path, without adding it to a server catalog."""
        return json.loads((self.path / "status.json").read_text(encoding="utf8"))


@dataclass
class _Retention:
    reserved: int
    specs: dict
    batches: dict = field(default_factory=dict)
    terminal: bool = False


class RetainedRunBuffers:
    """Bounded server-owned, volatile completed results keyed by run ID.

    Reserve worst-case bytes from each buffer's capacity and schema at admission.
    A runner transfers independent completed batches on finalization. This class
    does not supply IPC; a future service must transport batches to its process.
    """
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Server-wide buffer limit must be a positive byte count")
        self.max_bytes = max_bytes
        self._runs = {}
        self._lock = threading.RLock()

    @property
    def reserved_bytes(self):
        with self._lock:
            return sum(run.reserved for run in self._runs.values())

    def reserve(self, run_id, plan):
        """Budget the run before startup; never evict a previous run implicitly."""
        if not isinstance(plan, CompiledPlan) or plan.document["version"] != PLAN_VERSION:
            raise ValueError("Buffer reservation requires a validated current-version plan")
        specs = {}
        for attachment in plan.document["pipelines"]:
            compiled = CompiledPipeline.from_document(attachment["pipeline"])
            definition = compiled.document["definition"]
            if len(definition["inputs"]) != 1 or compiled.document["empty_inputs"]:
                raise PipelineError("Current runner needs one nonempty pipeline input")
            schema = definition["inputs"][0]["schema"]
            for node in definition["nodes"]:
                if node["kind"] != "buffer":
                    continue
                if node["name"] in specs:
                    raise PipelineError(f"Ambiguous run buffer name: {node['name']}")
                specs[node["name"]] = (node["config"]["capacity"], schema["max_bytes"])
        required = sum(capacity * size for capacity, size in specs.values())
        with self._lock:
            if run_id in self._runs:
                raise ValueError(f"Run already registered: {run_id}")
            if self.reserved_bytes + required > self.max_bytes:
                raise PipelineError("Server-wide buffer memory limit exceeded; release old run buffers")
            self._runs[run_id] = _Retention(required, specs)
        return required

    def finish(self, run_id, batches):
        """Transfer completed batches once, before reporting the run terminal."""
        with self._lock:
            run = self._runs[run_id]
            if run.terminal:
                raise ValueError("Run already finalized")
            if set(batches) - set(run.specs):
                raise PipelineError("Unknown buffer in run transfer")
            captured = {}
            for name, items in batches.items():
                capacity, max_bytes = run.specs[name]
                if len(items) > capacity:
                    raise PipelineError(f"Buffer {name!r} capacity exceeded; no data discarded")
                for batch in items:
                    size = batch.data.nbytes + sum(axis.points.nbytes for axis in batch.axes)
                    if size > max_bytes:
                        raise PipelineError(f"Buffer {name!r} exceeds its declared schema size")
                captured[name] = tuple(snapshot(batch) for batch in items)
            run.batches = captured
            run.terminal = True

    def view(self, run_id, name):
        """Return copies; consumers cannot mutate the retained server results."""
        with self._lock:
            run = self._runs[run_id]
            if not run.terminal:
                raise RuntimeError("Buffers are available after run finalization")
            if name not in run.specs or run.reserved == 0:
                raise KeyError(name)
            return [snapshot(batch) for batch in run.batches.get(name, ())]

    def release_buffers(self, run_id):
        """Free results and reservation, preserving the terminal run's catalog entry."""
        with self._lock:
            run = self._runs[run_id]
            if not run.terminal:
                raise RuntimeError("Cannot release buffers from an active run")
            run.batches.clear()
            run.reserved = 0

    def cleanup(self, run_id):
        """Forget a terminal run; never touch its independently readable folder."""
        with self._lock:
            if not self._runs[run_id].terminal:
                raise RuntimeError("Cannot clean up an active run")
            del self._runs[run_id]
