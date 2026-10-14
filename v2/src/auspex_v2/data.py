"""Versioned V2 .auspex dataset storage; no V1 file compatibility is implied."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

FORMAT_VERSION = 1
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class Axis:
    """A dataset dimension with its actual numeric coordinates and optional unit."""

    name: str
    points: object
    unit: str | None = None


@dataclass(frozen=True)
class Dataset:
    """A completed array and the coordinates published with it."""

    data: np.ndarray
    axes: tuple[Axis, ...]


class AuspexStore:
    """Read published datasets or exclusively write completed datasets in a V2 folder."""

    def __init__(self, path: str | Path, *, mode: str = "r"):
        self.path = Path(path)
        if self.path.suffix != ".auspex":
            raise ValueError("Store path must end in .auspex")
        if mode not in ("r", "w", "a"):
            raise ValueError("Mode must be r, w or a")
        self.mode = mode
        self._lock = None
        if mode == "w":
            self.path.mkdir(parents=True, exist_ok=False)
        elif not self.path.is_dir():
            raise FileNotFoundError(self.path)
        try:
            if mode != "r":
                self._lock = (self.path / ".writer.lock").open("a+b")
                try:
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Another writer owns this .auspex folder") from exc
            manifest = self.path / "manifest.json"
            if mode == "w":
                self._manifest = {"format_version": FORMAT_VERSION, "datasets": {}}
                self._publish_manifest()
            else:
                with manifest.open(encoding="utf-8") as stream:
                    self._manifest = json.load(stream)
                if self._manifest.get("format_version") != FORMAT_VERSION:
                    raise ValueError("Unsupported .auspex format version")
                if not isinstance(self._manifest.get("datasets"), dict):
                    raise ValueError("Invalid .auspex manifest")
        except BaseException:
            self.close()
            raise

    def close(self):
        """Release this store's exclusive writer lock, if any."""
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or not _NAME.fullmatch(value) or value in (".", ".."):
            raise ValueError("Group and dataset names must be simple path components")
        return value

    def list_datasets(self):
        """List published group/dataset pairs, excluding incomplete or orphaned files."""
        return [(group, name) for group, entries in self._manifest["datasets"].items()
                for name in entries]

    def write_dataset(self, group: str, name: str, data, axes: tuple[Axis, ...] | list[Axis]):
        """Publish a complete numeric array and matching coordinate axes atomically."""
        if self.mode == "r" or self._lock is None:
            raise PermissionError("Store is not open for writing")
        group, name = self._name(group), self._name(name)
        if name in self._manifest["datasets"].get(group, {}):
            raise FileExistsError(f"Dataset already exists: {group}/{name}")
        array = np.asarray(data)
        if array.dtype.kind not in "biufc" or array.dtype.hasobject:
            raise TypeError("Only numeric and boolean arrays are supported")
        array = np.array(array, order="C", copy=True)  # preserves scalar (zero-dimensional) arrays
        if len(axes) != array.ndim:
            raise ValueError("Provide one coordinate axis per array dimension")
        described = []
        for axis, extent in zip(axes, array.shape):
            if not isinstance(axis, Axis) or not isinstance(axis.name, str) or not axis.name:
                raise ValueError("Each axis must have a nonempty name")
            if axis.unit is not None and not isinstance(axis.unit, str):
                raise ValueError("Axis unit must be a string or None")
            points = np.asarray(axis.points)
            if points.ndim != 1 or len(points) != extent or points.dtype.kind not in "biuf":
                raise ValueError("Axis points must be a numeric vector matching its dimension")
            values = points.tolist()
            if not all(np.isfinite(points)):
                raise ValueError("Axis points must be finite")
            described.append({"name": axis.name, "unit": axis.unit, "points": values})
        entry = {"file": f"{group}/{name}.dat", "shape": list(array.shape),
                 "dtype": array.dtype.str, "axes": described}
        directory = self.path / group
        directory.mkdir(exist_ok=True)
        target = self.path / entry["file"]
        if target.exists():
            raise FileExistsError(target)  # never overwrite an orphan from an interrupted write
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".array-", delete=False) as stream:
                tmp = Path(stream.name)
                stream.write(array.tobytes(order="C"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
            tmp = None
            updated = {**self._manifest, "datasets": {**self._manifest["datasets"]}}
            updated["datasets"][group] = {**updated["datasets"].get(group, {}), name: entry}
            self._publish_manifest(updated)
            self._manifest = updated
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def read_dataset(self, group: str, name: str) -> Dataset:
        """Read only a published dataset, checking its binary length against the descriptor."""
        group, name = self._name(group), self._name(name)
        entry = self._manifest["datasets"][group][name]
        shape = tuple(entry["shape"])
        dtype = np.dtype(entry["dtype"])
        if dtype.kind not in "biufc" or dtype.hasobject or any(n < 0 for n in shape):
            raise ValueError("Invalid dataset descriptor")
        file = self.path / entry["file"]
        if file != self.path / group / f"{name}.dat":
            raise ValueError("Invalid dataset path")
        count = int(np.prod(shape, dtype=np.int64))
        if file.stat().st_size != count * dtype.itemsize:
            raise ValueError("Dataset file size does not match manifest")
        array = np.fromfile(file, dtype=dtype).reshape(shape)
        axes = tuple(Axis(ax["name"], ax["points"], ax["unit"]) for ax in entry["axes"])
        return Dataset(array, axes)

    def _publish_manifest(self, content=None):
        """Replace the manifest last so readers never see an unpublished dataset."""
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path, prefix=".manifest-",
                                             encoding="utf-8", delete=False) as stream:
                tmp = Path(stream.name)
                json.dump(content if content is not None else self._manifest, stream,
                          allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path / "manifest.json")
            tmp = None
            fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
