"""Coordinate-aware complex averaging across complete acquisition windows."""

import numpy as np

from ..data import Axis
from .core import Batch, Emission, Filter, GroupComplete, PipelineError


class ComplexAverager(Filter):
    """Average identical complex grids across a bounded group of complete windows."""

    def __init__(self, schema, *, repeats=5, group_name="average"):
        if (np.dtype(schema.dtype).kind != "c" or schema.rank != 1 or
                type(repeats) is not int or repeats < 1 or not isinstance(group_name, str) or not group_name):
            raise ValueError("Averager needs 1D complex data, a name and a positive repeat count")
        self.input_schema = schema
        self.output_schemas = {"partial": schema, "final": schema}
        self.required_ports = ("final",)
        self.repeats = repeats
        self.group_name = group_name
        self.groups_completed = 0
        self.group = None
        self.window = None
        self.count = 0
        self.total = None
        self.axes = None
        self.last_revision = None
        self.sweep = None
        self.scan = None

    def configure(self, changes):
        """Change the repeat target only after the previous group has ended."""
        if self.group is not None or set(changes) != {"repeats"} or type(changes["repeats"]) is not int or changes["repeats"] < 1:
            raise PipelineError("Averaging repeats can change only between groups")
        self.repeats = changes["repeats"]

    def on_batch(self, batch):
        """Accept exactly one trace per window, without mixing unlike coordinates."""
        if self.window is not None or self.count >= self.repeats:
            raise PipelineError("Extra trace in window or averaging group")
        if self.last_revision is not None and batch.revision != self.last_revision:
            raise PipelineError("Configuration changed during averaging group")
        if self.group is not None and (batch.scan != self.scan or batch.sweep != self.sweep):
            raise PipelineError("Averaging group crossed scan or sweep position")
        if self.axes is not None:
            if any(a.name != b.name or a.unit != b.unit or
                   not np.array_equal(np.asarray(a.points), np.asarray(b.points))
                   for a, b in zip(batch.axes, self.axes)):
                raise PipelineError("Actual frequency grids differ between sweeps")
        else:
            self.axes = tuple(Axis(a.name, np.array(a.points, copy=True), a.unit) for a in batch.axes)
        if self.group is None:
            self.group = f"{self.group_name}-{self.groups_completed}"
            self.scan, self.sweep = batch.scan, batch.sweep
        self.window, self.last_revision = batch.window, batch.revision
        if self.total is None:
            self.total = np.zeros(batch.data.shape, dtype=np.complex128)
        self.total += batch.data
        return ()

    def on_window_end(self, window):
        """Emit a preview, and close this filter's group at its configured count."""
        if window != self.window:
            raise PipelineError("Window ended without its expected trace")
        self.count += 1
        self.window = None
        snapshot = (self.total / self.count).astype(np.dtype(self.input_schema.dtype))
        partial = Emission("partial", Batch(self.group, window, 0, self.last_revision,
                                             snapshot, self.axes, self.count, self.repeats,
                                             self.scan, self.sweep))
        if self.count < self.repeats:
            return (partial,)
        final = Emission("final", Batch(self.group, window, 0, self.last_revision,
                                         snapshot.copy(), self.axes, self.count, self.repeats,
                                         self.scan, self.sweep))
        complete = GroupComplete(self.group)
        self.group, self.window, self.count = None, None, 0
        self.total, self.axes, self.last_revision = None, None, None
        self.scan, self.sweep = None, None
        self.groups_completed += 1
        return (partial, final, complete)

    def on_scan_end(self, scan):
        """A group must finish within its owning scan."""
        if self.group is not None:
            raise PipelineError("Averaging group is incomplete at scan end")
        return ()

    def on_run_end(self):
        """Never treat an unfinished average as a valid final result."""
        if self.group is not None:
            raise PipelineError("Averaging group is incomplete")
        return ()
