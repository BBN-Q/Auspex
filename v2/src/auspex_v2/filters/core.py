"""Small, synchronous reference implementation of the V2 filter contract.

The graph and filters here do not own processes or instrument connections. A future
transport must preserve these ordering and failure semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np




class PipelineError(Exception):
    """Reject invalid graph messages or incomplete processing rather than emit bad data."""


@dataclass(frozen=True)
class PortSchema:
    """Stable data meaning; shape and actual axis points may vary by window."""

    dtype: str
    rank: int
    axis_names: tuple[str, ...]
    axis_units: tuple[str | None, ...]
    max_bytes: int
    value_name: str = "value"
    value_unit: str | None = None

    def __post_init__(self):
        if (self.rank < 0 or self.max_bytes < 1 or len(self.axis_names) != self.rank or
                len(self.axis_units) != self.rank or np.dtype(self.dtype).kind not in "biufc" or
                not self.value_name or (self.value_unit is not None and not isinstance(self.value_unit, str))):
            raise ValueError("Invalid port schema")

    def check(self, batch):
        """Check a batch against the declared port and its actual coordinates."""
        data = np.asarray(batch.data)
        if (data.dtype != np.dtype(self.dtype) or data.ndim != self.rank or
                len(batch.axes) != self.rank or data.nbytes > self.max_bytes):
            raise PipelineError("Batch dtype, rank or size violates port schema")
        total_bytes = data.nbytes
        for dimension, (axis, name, unit) in enumerate(zip(batch.axes, self.axis_names, self.axis_units)):
            points = np.asarray(axis.points)
            total_bytes += points.nbytes
            if (axis.name != name or axis.unit != unit or points.ndim != 1 or
                    len(points) != data.shape[dimension] or points.dtype.kind not in "biuf" or
                    not np.all(np.isfinite(points))):
                raise PipelineError("Batch coordinate axis violates port schema")
        if total_bytes > self.max_bytes:
            raise PipelineError("Batch including coordinates exceeds port limit")


@dataclass(frozen=True)
class SweepPosition:
    """An actual swept parameter value and its unit for one acquisition window."""

    name: str
    value: float
    unit: str | None = None


@dataclass(frozen=True)
class Batch:
    """One array and its actual coordinates, tagged with ownership identifiers."""

    group: str | None  # None on acquisition input; filters assign their own group IDs.
    window: int
    sequence: int
    revision: int
    data: np.ndarray
    axes: tuple[Axis, ...]
    count: int | None = None
    expected_count: int | None = None
    scan: str = "default"
    sweep: tuple[SweepPosition, ...] = ()


@dataclass(frozen=True)
class Emission:
    """A batch emitted from a declared output port."""

    port: str
    batch: Batch


class GroupComplete:
    """A filter-owned boundary emitted after its final result for this group."""

    def __init__(self, group):
        self.group = group


class WindowState:
    """Validate experiment-owned windows, independent of filter-owned groups."""

    def __init__(self):
        self.window = None
        self.last_window = -1
        self.sequence = 0
        self.revision = 0
        self.finished = False
        self.scan = None
        self.closed_scans = set()

    def admit(self, batch):
        if self.finished or batch.group is not None:
            raise PipelineError("Only the pipeline assigns groups to acquisition batches")
        if (not isinstance(batch.scan, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", batch.scan)
                or batch.scan in (".", "..") or batch.scan in self.closed_scans):
            raise PipelineError("Invalid or closed scan")
        if self.scan is None:
            self.scan = batch.scan
        if self.scan != batch.scan:
            raise PipelineError("Finish the scan before starting another")
        if len({p.name for p in batch.sweep}) != len(batch.sweep) or any(
                not isinstance(p, SweepPosition) or not p.name or not np.isfinite(p.value)
                or (p.unit is not None and not isinstance(p.unit, str)) for p in batch.sweep):
            raise PipelineError("Invalid sweep position")
        if self.window is None:
            if batch.window != self.last_window + 1:
                raise PipelineError("Unexpected window")
            self.window = batch.window
            self.sequence = 0
        if ((batch.window, batch.sequence, batch.revision) !=
                (self.window, self.sequence, self.revision)):
            raise PipelineError("Out-of-order or differently configured batch")
        self.sequence += 1

    def end_window(self, window):
        if self.finished or window != self.window:
            raise PipelineError("Wrong window completion")
        self.last_window = window
        self.window = None

    def can_update(self):
        if self.finished or self.window is not None:
            raise PipelineError("Finish the acquisition window before changing parameters")

    def end_scan(self, scan):
        """Close a scan only between windows; no later batch may reopen it."""
        self.can_update()
        if scan != self.scan:
            raise PipelineError("Wrong scan completion")
        self.closed_scans.add(scan)
        self.scan = None

    def finish(self):
        self.can_update()
        if self.scan is not None and self.scan != "default":
            raise PipelineError("Finish the scan before ending the run")
        self.finished = True


class Filter:
    """A one-input, testable processor independent of process and transport code."""

    input_schema: PortSchema | None = None
    output_schemas: dict[str, PortSchema] = {}
    required_ports: tuple[str, ...] = ()
    best_effort = False

    def on_batch(self, batch):
        return ()

    def on_window_end(self, window):
        return ()

    def on_group_end(self, group):
        return ()

    def on_run_end(self):
        return ()

    def on_scan_end(self, scan):
        return ()

    def configure(self, changes):
        """Reject changes unless this filter explicitly declares a safe boundary update."""
        raise PipelineError("Filter does not accept parameter changes")


class Passthrough(Filter):
    """Forward validated batches unchanged through a named output."""

    def __init__(self, schema):
        self.input_schema = schema
        self.output_schemas = {"out": schema}

    def on_batch(self, batch):
        return (Emission("out", batch),)


class Pipeline:
    """Validate and run a one-input DAG synchronously as a transport reference."""

    def __init__(self, source_schema, nodes, edges):
        """Connect (producer, port, consumer) edges; '$input' is the sole source."""
        self.source_schema = source_schema
        self.nodes = dict(nodes)
        self.edges = {}
        incoming = {name: 0 for name in nodes}
        for producer, port, consumer in edges:
            if consumer not in nodes or producer not in nodes and producer != "$input":
                raise PipelineError("Unknown graph node")
            outputs = {"out": source_schema} if producer == "$input" else nodes[producer].output_schemas
            if port not in outputs or outputs[port] != nodes[consumer].input_schema:
                raise PipelineError("Incompatible port schemas")
            incoming[consumer] += 1
            self.edges.setdefault((producer, port), []).append(consumer)
        if any(count != 1 for count in incoming.values()):
            raise PipelineError("Every node must have exactly one connected input")
        for name, node in nodes.items():
            if any((name, port) not in self.edges for port in node.required_ports):
                raise PipelineError("Required filter output must be connected")
        if not nodes or not self.edges.get(("$input", "out")):
            raise PipelineError("Graph needs a source and at least one node")
        remaining = set(nodes)
        visited = {"$input"}
        while remaining:
            available = {node for node in remaining if any(
                producer in visited and node in consumers
                for (producer, _), consumers in self.edges.items())}
            if not available:
                raise PipelineError("Cycle or unreachable graph node")
            visited.update(available)
            remaining.difference_update(available)
        self.state = WindowState()
        self.failed = False

    def _deliver(self, producer, port, batch):
        for consumer in self.edges.get((producer, port), ()):
            node = self.nodes[consumer]
            try:
                node.input_schema.check(batch)
                emitted = node.on_batch(batch)
            except Exception:
                if not node.best_effort:
                    raise
                continue  # Only explicitly best-effort sinks may fail independently.
            self._emit(consumer, emitted)

    def _emit(self, producer, emissions):
        for emission in emissions:
            if isinstance(emission, GroupComplete):
                self._event(producer, "group", emission.group)
            else:
                schema = self.nodes[producer].output_schemas[emission.port]
                schema.check(emission.batch)
                self._deliver(producer, emission.port, emission.batch)

    def _event(self, producer, event, group=None, window=None):
        """Forward an ordered boundary only after a node has emitted its results."""
        for (source, _), consumers in self.edges.items():
            if source != producer:
                continue
            for consumer in consumers:
                node = self.nodes[consumer]
                try:
                    emitted = (node.on_window_end(window) if event == "window" else
                               node.on_group_end(group) if event == "group" else
                               node.on_scan_end(group) if event == "scan" else
                               node.on_run_end())
                    self._emit(consumer, emitted)
                    self._event(consumer, event, group, window)
                except Exception:
                    if not node.best_effort:
                        raise

    def _guard(self, operation):
        if self.failed:
            raise PipelineError("Pipeline failed; start a new run")
        try:
            operation()
        except Exception:
            self.failed = True
            raise

    def push(self, batch):
        """Admit ordered batches for one window; errors permanently fail this run."""
        def apply():
            self.source_schema.check(batch)
            self.state.admit(batch)
            self._deliver("$input", "out", batch)
        self._guard(apply)

    def update(self, node, changes):
        """Acknowledge one validated parameter change at a group/window boundary."""
        if self.failed:
            raise PipelineError("Pipeline failed; start a new run")
        self.state.can_update()
        if node not in self.nodes:
            raise PipelineError("Unknown filter node")
        self.nodes[node].configure(changes)
        self.state.revision += 1
        return self.state.revision

    def end_window(self, window):
        """Complete an experiment-owned window after its last ordered batch."""
        def apply():
            self.state.end_window(window)
            self._event("$input", "window", window=window)
        self._guard(apply)

    def end_scan(self, scan):
        """Propagate a scan boundary after the last window and validate filter state."""
        def apply():
            self.state.end_scan(scan)
            self._event("$input", "scan", group=scan)
        self._guard(apply)

    def finish(self):
        """Fail if any required filter still holds an incomplete group."""
        def apply():
            self.state.finish()
            self._event("$input", "run", window=None)
        self._guard(apply)
