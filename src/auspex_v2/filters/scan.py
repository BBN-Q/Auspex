"""Bounded disk-staged assembly of one named graph output into a scan dataset."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
import tempfile

import numpy as np

from ..data import Axis, Coordinate
from .core import PipelineError


class ScanAssembler:
    """Stage window-complete batches; publish only at an explicit scan boundary."""

    def __init__(self, schema, *, max_scan_bytes=64 * 1024 * 1024,
                 max_scan_windows=8192, scratch_dir=None):
        self.schema = schema
        self.max_scan_bytes = max_scan_bytes
        self.max_scan_windows = max_scan_windows
        self.scratch = tempfile.TemporaryDirectory(prefix=".auspex-scan-", dir=scratch_dir)
        self.rows = []
        self.total_bytes = 0
        self.pending = None
        self.scan = None
        self.last_window = -1

    def on_batch(self, batch):
        """Retain one trace until its ordered window boundary validates completion."""
        self.schema.check(batch)
        if self.pending is not None:
            raise PipelineError("Writer expects one batch per window")
        if self.scan is None:
            self.scan = batch.scan
        if batch.scan != self.scan:
            raise PipelineError("Writer crossed scan without completion")
        self.pending = replace(batch, data=np.array(batch.data, copy=True),
                               axes=tuple(Axis(a.name, np.array(a.points, copy=True), a.unit)
                                          for a in batch.axes))

    def on_window_end(self, window):
        """Spool the validated batch so input shared-memory slots can be reused."""
        batch = self.pending
        if window <= self.last_window or (batch is not None and batch.window != window):
            raise PipelineError("Out-of-order writer window")
        self.last_window = window
        if batch is None:
            return  # A group-producing filter need not emit at every window.
        size = batch.data.nbytes + sum(np.asarray(a.points).nbytes for a in batch.axes)
        if len(self.rows) >= self.max_scan_windows or self.total_bytes + size > self.max_scan_bytes:
            raise PipelineError("Scan exceeds declared writer capacity")
        index = len(self.rows)
        file = Path(self.scratch.name) / f"{index}.npz"
        # np.savez writes numeric arrays; load always disables pickle.
        with file.open("wb") as stream:
            np.savez(stream, value=batch.data, **{f"axis_{i}": a.points for i, a in enumerate(batch.axes)})
        self.rows.append((file, batch.sweep, batch.scan))
        self.total_bytes += size
        self.pending = None

    def publish(self, scan, store, group, name, *, label=None, source=None, input_name=None):
        """Assemble actual coordinates without regridding and publish one dataset."""
        if scan != self.scan or self.pending is not None or not self.rows:
            raise PipelineError("Incomplete or empty scan")
        sweeps = [s for _, s, _ in self.rows]
        if any(tuple((p.name, p.unit) for p in s) != tuple((p.name, p.unit) for p in sweeps[0]) for s in sweeps):
            raise PipelineError("Sweep parameters changed within a scan")
        # One outer observation dimension preserves arbitrary sweep paths, repeats and adaptivity.
        outer = "point"
        if outer in self.schema.axis_names:
            raise PipelineError("Measurement axis conflicts with point dimension")
        values = []
        axes = [[] for _ in self.schema.axis_names]
        for file, _, _ in self.rows:
            with np.load(file, allow_pickle=False) as loaded:
                values.append(loaded["value"])
                for i, dest in enumerate(axes):
                    dest.append(loaded[f"axis_{i}"])
        shapes = {value.shape for value in values}
        coords = [Coordinate(p.name, np.array([s[i].value for s in sweeps]), (outer,), p.unit)
                  for i, p in enumerate(sweeps[0])]
        if len({c.name for c in coords}) != len(coords) or any(c.name in self.schema.axis_names for c in coords):
            raise PipelineError("Sweep and data coordinate names collide")
        kwargs = {"value_name": self.schema.value_name, "value_unit": self.schema.value_unit,
                  "label": label, "source": source, "input_name": input_name}
        if len(shapes) == 1:
            # Only promote sweep settings to independent dimensions when every
            # Cartesian combination occurs exactly once in acquisition order.
            grids = [list(dict.fromkeys(s[i].value for s in sweeps))
                     for i in range(len(sweeps[0]))]
            rectangular = bool(grids) and list(product(*grids)) == [tuple(p.value for p in s) for s in sweeps]
            if rectangular:
                outer_dims = tuple(p.name for p in sweeps[0])
                coords = [Coordinate(p.name, np.asarray(grid), (p.name,), p.unit)
                          for p, grid in zip(sweeps[0], grids)]
                leading_shape = tuple(map(len, grids))
            else:
                outer_dims = (outer,)
                leading_shape = (len(values),)
                coords.insert(0, Coordinate(outer, np.arange(len(values)), (outer,)))
            result = np.stack(values).reshape(leading_shape + values[0].shape)
            dims = outer_dims + self.schema.axis_names
            for i, (axis_name, unit) in enumerate(zip(self.schema.axis_names, self.schema.axis_units)):
                if all(np.array_equal(axes[i][0], a) for a in axes[i][1:]):
                    coords.append(Coordinate(axis_name, axes[i][0], (axis_name,), unit))
                else:
                    points = np.stack(axes[i]).reshape(leading_shape + axes[i][0].shape)
                    coords.append(Coordinate(axis_name, points, outer_dims + (axis_name,), unit))
            store.write_dataset(group, name, result, coordinates=coords, dimensions=dims, **kwargs)
        else:
            if self.schema.rank != 1:
                raise PipelineError("Variable-length measurements require rank-one input")
            sizes = [len(v) for v in values]
            offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
            coords.insert(0, Coordinate(outer, np.arange(len(values)), (outer,)))
            coords.append(Coordinate(self.schema.axis_names[0], np.concatenate(axes[0]),
                                     ("sample",), self.schema.axis_units[0]))
            store.write_dataset(group, name, np.concatenate(values), coordinates=coords,
                                dimensions=(outer,), offsets=offsets, **kwargs)
        self.reset()

    def reset(self):
        """Discard the completed scan's staging files and start a new one."""
        for file, _, _ in self.rows:
            file.unlink(missing_ok=True)
        self.rows.clear()
        self.total_bytes = 0
        self.scan = None
        self.last_window = -1

    def close(self):
        """Remove unpublished staging files after cancellation or completion."""
        self.scratch.cleanup()
