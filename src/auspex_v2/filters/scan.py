"""Bounded disk-staged assembly of one named graph output into a scan dataset."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import math
from pathlib import Path
import tempfile

import numpy as np

from ..data import Axis, Coordinate
from ..data.data import _CoordinateLayout
from .core import Batch, PipelineError


class ScanAssembler:
    """Stage complete windows; inspect layout and copy bounded chunks at scan end."""

    def __init__(self, schema, *, max_scan_bytes=512 * 1024 * 1024,
                 max_scan_windows=16384, scratch_dir=None):
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

    def _read_window(self, file):
        """Load and validate one staged window, never an assembled scan."""
        with np.load(file, allow_pickle=False) as loaded:
            data = loaded["value"]
            axes = tuple(Axis(name, loaded[f"axis_{i}"], unit)
                         for i, (name, unit) in enumerate(zip(self.schema.axis_names, self.schema.axis_units)))
        self.schema.check(Batch(None, 0, 0, 0, data, axes))
        return data, tuple(a.points for a in axes)

    def _inspect_windows(self):
        """Retain shapes/dtypes and one reference grid; discard each inspected payload.

        Coordinate equality deliberately ignores dtype as np.array_equal did in
        whole-scan assembly. Shared axes keep the first dtype; dependent/ragged
        axes use NumPy's common dtype across all observed coordinate dtypes.
        """
        shapes = []
        reference = None
        dtypes = [set() for _ in self.schema.axis_names]
        shared = [True] * self.schema.rank
        for file, _, _ in self.rows:
            data, axes = self._read_window(file)
            shapes.append(data.shape)
            if reference is None:
                reference = axes
            for i, points in enumerate(axes):
                dtypes[i].add(points.dtype)
                shared[i] = shared[i] and np.array_equal(reference[i], points)
            del data, axes
        return shapes, reference, tuple(np.result_type(*types) for types in dtypes), shared

    def _outer_layout(self, sweeps, *, allow_cartesian):
        """Classify only acquired positions, without expanding a hypothetical grid."""
        grids = [list(dict.fromkeys(s[i].value for s in sweeps))
                 for i in range(len(sweeps[0]))] if allow_cartesian else []
        rectangular = (bool(grids) and math.prod(map(len, grids)) == len(sweeps)
                       and all(expected == tuple(p.value for p in actual)
                               for expected, actual in zip(product(*grids), sweeps, strict=True)))
        if rectangular:
            dims = tuple(p.name for p in sweeps[0])
            coords = [Coordinate(p.name, np.asarray(grid), (p.name,), p.unit)
                      for p, grid in zip(sweeps[0], grids)]
            return dims, tuple(map(len, grids)), coords
        coords = [Coordinate("point", np.arange(len(sweeps)), ("point",))]
        coords.extend(Coordinate(p.name, np.asarray([s[i].value for s in sweeps]), ("point",), p.unit)
                      for i, p in enumerate(sweeps[0]))
        return ("point",), (len(sweeps),), coords

    def publish(self, scan, store, group, name, *, label=None, source=None, input_name=None):
        """Inspect layout, then copy windows into a temporary final file before publication.

        Numerical working memory is bounded by a few windows and a reference grid;
        sweep coordinates, shapes and offsets are bounded per-row metadata. This
        does not bound OS file caching or whole-dataset reads by later clients.
        """
        if scan != self.scan or self.pending is not None or not self.rows:
            raise PipelineError("Incomplete or empty scan")
        sweeps = [s for _, s, _ in self.rows]
        if any(tuple((p.name, p.unit) for p in s) != tuple((p.name, p.unit) for p in sweeps[0]) for s in sweeps):
            raise PipelineError("Sweep parameters changed within a scan")
        if "point" in self.schema.axis_names:
            raise PipelineError("Measurement axis conflicts with point dimension")
        names = [p.name for p in sweeps[0]]
        if len(set(names)) != len(names) or set(names) & set(self.schema.axis_names):
            raise PipelineError("Sweep and data coordinate names collide")
        shapes, reference, axis_dtypes, shared = self._inspect_windows()
        uniform = all(shape == shapes[0] for shape in shapes)
        if not uniform and self.schema.rank != 1:
            raise PipelineError("Variable-length measurements require rank-one input")
        outer_dims, leading_shape, coords = self._outer_layout(sweeps, allow_cartesian=uniform)
        offsets = None
        if uniform:
            dimensions = outer_dims + self.schema.axis_names
            shape = leading_shape + shapes[0]
        else:
            offsets = np.empty(len(shapes) + 1, dtype="<i8")
            offsets[0] = 0
            np.cumsum([s[0] for s in shapes], out=offsets[1:])
            shape, dimensions = (int(offsets[-1]),), outer_dims
        layouts = [_CoordinateLayout(c.name, np.shape(c.values), np.asarray(c.values).dtype,
                                     c.dimensions, c.unit) for c in coords]
        for i, (axis_name, unit) in enumerate(zip(self.schema.axis_names, self.schema.axis_units)):
            if not uniform:
                layout = _CoordinateLayout(axis_name, shape, axis_dtypes[i], ("sample",), unit)
            elif shared[i]:
                layout = _CoordinateLayout(axis_name, reference[i].shape, reference[i].dtype, (axis_name,), unit)
            else:
                layout = _CoordinateLayout(axis_name, leading_shape + (shapes[0][i],),
                                           axis_dtypes[i], outer_dims + (axis_name,), unit)
            layouts.append(layout)

        # np.stack/concatenate previously normalized non-native value byte order.
        value_dtype = np.result_type(self.schema.dtype)

        def chunks():
            """Supply values and dependent coordinates together from each reloaded window."""
            for coordinate in coords:
                yield coordinate.name, coordinate.values
            if uniform:
                for i, axis_name in enumerate(self.schema.axis_names):
                    if shared[i]:
                        yield axis_name, reference[i]
            for row, (file, _, _) in enumerate(self.rows):
                data, axes = self._read_window(file)
                if data.shape != shapes[row]:
                    raise PipelineError("Staged window shape changed during publication")
                yield None, data.astype(value_dtype, copy=False)
                for i, axis_name in enumerate(self.schema.axis_names):
                    if not uniform or not shared[i]:
                        yield axis_name, axes[i].astype(axis_dtypes[i], copy=False)

        store._write_dataset_chunks(group, name, shape=shape, dtype=value_dtype,
            dimensions=dimensions, coordinates=layouts, chunks=chunks(), offsets=offsets,
            value_name=self.schema.value_name, value_unit=self.schema.value_unit,
            label=label, source=source, input_name=input_name)
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
