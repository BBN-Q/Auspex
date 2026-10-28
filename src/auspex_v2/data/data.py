"""Self-describing, versioned V2 .auspex datasets (no legacy-format reader)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

import numpy as np

FORMAT_VERSION = 2
MAGIC = b"AUSPEX2\n"
MAX_HEADER = 1024 * 1024
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class Axis:
    """A dimension with actual numeric coordinates and an optional unit."""

    name: str
    points: object
    unit: str | None = None


@dataclass(frozen=True)
class Coordinate:
    """A numeric coordinate indexed by named dimensions (possibly dependent)."""

    name: str
    values: object
    dimensions: tuple[str, ...]
    unit: str | None = None


@dataclass(frozen=True)
class _CoordinateLayout:
    """Private coordinate description for scan assembly without a full value array."""

    name: str
    shape: tuple[int, ...]
    dtype: object
    dimensions: tuple[str, ...]
    unit: str | None = None


class _NamedSequence(tuple):
    """An immutable metadata sequence with unambiguous, case-sensitive name lookup."""

    __slots__ = ()

    def __new__(cls, items):
        result = super().__new__(cls, items)
        names = set()
        for item in result:
            if item.name in names:
                raise ValueError(f"Duplicate metadata name: {item.name!r}")
            names.add(item.name)
        return result

    def __getitem__(self, key):
        if isinstance(key, str):
            for item in self:
                if item.name == key:
                    return item
            raise KeyError(key)
        return super().__getitem__(key)


@dataclass(frozen=True)
class Dataset:
    """A complete value array, its coordinates and optional ragged offsets."""

    data: np.ndarray
    axes: tuple[Axis, ...]
    coordinates: tuple[Coordinate, ...] = ()
    offsets: np.ndarray | None = None
    dimensions: tuple[str, ...] = ()
    value_name: str = "value"
    value_unit: str | None = None
    label: str | None = None
    source: str | None = None
    input_name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "axes", _NamedSequence(self.axes))
        object.__setattr__(self, "coordinates", _NamedSequence(self.coordinates))

    def trace(self, index: int):
        """Return one row of a ragged scan without constructing object arrays."""
        if self.offsets is None:
            raise ValueError("Dataset is not ragged")
        return self.data[int(self.offsets[index]):int(self.offsets[index + 1])]


def _component(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value in (".", ".."):
        raise ValueError("Names must be simple path components")
    return value


def _numeric(value):
    """Inspect numeric data without copying existing arrays; writers own their snapshots."""
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or array.dtype.hasobject:
        raise TypeError("Only numeric and boolean arrays are supported")
    return array


def _validated_dataset(dataset: Dataset):
    """Resolve layout defaults and validate array views without mutation or filesystem I/O."""
    data = _numeric(dataset.data)
    dimensions = tuple(dataset.dimensions) or tuple(axis.name for axis in dataset.axes)
    coordinates = dataset.coordinates or tuple(
        Coordinate(a.name, a.points, (a.name,), a.unit) for a in dataset.axes)
    if len(dimensions) != data.ndim or len(set(dimensions)) != len(dimensions):
        raise ValueError("Value dimensions do not match its rank")
    for name in dimensions:
        _component(name)
    if not isinstance(dataset.value_name, str) or not dataset.value_name:
        raise ValueError("Value needs a name")
    for unit in (dataset.value_unit, *(c.unit for c in coordinates)):
        if unit is not None and not isinstance(unit, str):
            raise ValueError("Units must be strings or None")
    for text in (dataset.label, dataset.source, dataset.input_name):
        if text is not None and not isinstance(text, str):
            raise ValueError("Labels and sources must be strings or None")
    ragged = dataset.offsets is not None
    offsets = None
    if ragged:
        if data.ndim != 1 or len(dimensions) != 1:
            raise ValueError("Ragged values must be flat and have an outer axis")
        offsets = np.asarray(dataset.offsets)
        if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 1:
            raise ValueError("Invalid ragged offsets: expected a nonempty one-dimensional integer array")
        if offsets[0] != 0 or offsets[-1] != len(data):
            raise ValueError("Invalid ragged offsets: must start at zero and end at the data length")
        # Direct comparison avoids unsigned underflow. Together with the endpoints,
        # this bounds offsets for the writer's later conversion to int64.
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("Invalid ragged offsets: must be nondecreasing")
    normalized_coordinates = []
    names = set()
    for coordinate in coordinates:
        _component(coordinate.name)
        if coordinate.name in names:
            raise ValueError("Duplicate coordinate name")
        names.add(coordinate.name)
        arr = _numeric(coordinate.values)
        if arr.dtype.kind not in "biuf" or not np.all(np.isfinite(arr)):
            raise ValueError("Coordinates must be finite real numbers")
        if not coordinate.dimensions or len(set(coordinate.dimensions)) != len(coordinate.dimensions):
            raise ValueError("Invalid coordinate dimensions")
        if any(dim not in dimensions and (not ragged or dim != "sample") for dim in coordinate.dimensions):
            raise ValueError("Unknown coordinate dimension")
        if ragged:
            expected = tuple(len(offsets) - 1 if d == dimensions[0] else len(data) for d in coordinate.dimensions)
            # A ragged sample coordinate is stored flat; it must not pretend to be rectangular.
            if "sample" in coordinate.dimensions and coordinate.dimensions != ("sample",):
                raise ValueError("Ragged sample coordinates must be flat")
        else:
            expected = tuple(data.shape[dimensions.index(d)] for d in coordinate.dimensions)
        if arr.shape != expected:
            raise ValueError("Coordinate shape does not match its dimensions")
        normalized_coordinates.append(Coordinate(coordinate.name, arr, tuple(coordinate.dimensions),
                                                 coordinate.unit))
    if ragged:
        if not any(c.dimensions == (dimensions[0],) for c in coordinates) or not any(c.dimensions == ("sample",) for c in coordinates):
            raise ValueError("Ragged values need outer and sample coordinates")
    elif any(not any(c.name == dim and dim in c.dimensions for c in coordinates) for dim in dimensions):
        raise ValueError("Every value dimension needs a named coordinate")
    return replace(dataset, data=data, dimensions=dimensions,
                   coordinates=tuple(normalized_coordinates), offsets=offsets)


def _pack_dataset(stream, dataset: Dataset):
    """Validate and snapshot a dataset, then encode its bounded header and numeric payload."""
    dataset = _validated_dataset(dataset)
    ragged = dataset.offsets is not None
    arrays = {"value": np.array(dataset.data, order="C", copy=True)}
    if ragged:
        arrays["offsets"] = np.asarray(dataset.offsets, dtype="<i8")
    described = []
    for i, coordinate in enumerate(dataset.coordinates):
        key = f"coordinate_{i}"
        arrays[key] = np.array(coordinate.values, order="C", copy=True)
        described.append({"name": coordinate.name, "dimensions": list(coordinate.dimensions),
                          "unit": coordinate.unit, "array": key})
    entries = {}
    position = 0
    for name, arr in arrays.items():
        entries[name] = {"dtype": arr.dtype.str, "shape": list(arr.shape),
                         "offset": position, "bytes": arr.nbytes}
        position += arr.nbytes
    header = {"format_version": FORMAT_VERSION, "dimensions": list(dataset.dimensions),
              "value_name": dataset.value_name, "value_unit": dataset.value_unit,
              "label": dataset.label, "source": dataset.source, "input_name": dataset.input_name,
              "coordinates": described,
              "arrays": entries, "ragged": ragged}
    encoded = json.dumps(header, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_HEADER:
        raise ValueError("Dataset header is too large")
    stream.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
    for arr in arrays.values():
        stream.write(arr.tobytes(order="C"))


def _scan_header(shape, dtype, dimensions, coordinates, offsets,
                 value_name, value_unit, label, source, input_name):
    """Validate scan layout metadata without allocating arrays of the final shape.

    Numerical coordinates are checked as chunks arrive; offsets are bounded row
    metadata. Keep these layout rules consistent with _validated_dataset.
    """
    shape, dimensions = tuple(shape), tuple(dimensions)
    if (any(type(n) is not int or n < 0 for n in shape)
            or len(dimensions) != len(shape) or len(set(dimensions)) != len(dimensions)):
        raise ValueError("Value dimensions do not match its rank")
    for dim in dimensions:
        _component(dim)
    if not isinstance(value_name, str) or not value_name:
        raise ValueError("Value needs a name")
    if value_unit is not None and not isinstance(value_unit, str):
        raise ValueError("Units must be strings or None")
    for text in (label, source, input_name):
        if text is not None and not isinstance(text, str):
            raise ValueError("Labels and sources must be strings or None")
    arrays = {}
    position = 0

    def add_array(key, dtype, shape):
        nonlocal position
        dtype = np.dtype(dtype)
        if dtype.kind not in "biufc" or dtype.hasobject:
            raise TypeError("Only numeric and boolean arrays are supported")
        size = math.prod(shape) * dtype.itemsize
        arrays[key] = {"dtype": dtype.str, "shape": list(shape), "offset": position, "bytes": size}
        position += size

    add_array("value", dtype, shape)
    ragged = offsets is not None
    if ragged:
        offsets = np.asarray(offsets)
        if (len(shape) != 1 or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
                or len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != shape[0]
                or np.any(offsets[1:] < offsets[:-1]) or shape[0] > np.iinfo(np.int64).max):
            raise ValueError("Invalid ragged offsets")
        add_array("offsets", "<i8", offsets.shape)
    described = []
    names = set()
    for i, c in enumerate(coordinates):
        _component(c.name)
        if c.name in names:
            raise ValueError("Duplicate coordinate name")
        names.add(c.name)
        if c.unit is not None and not isinstance(c.unit, str):
            raise ValueError("Units must be strings or None")
        if (not c.dimensions or len(set(c.dimensions)) != len(c.dimensions)
                or any(d not in dimensions and (not ragged or d != "sample") for d in c.dimensions)):
            raise ValueError("Invalid coordinate dimensions")
        if ragged:
            if "sample" in c.dimensions and c.dimensions != ("sample",):
                raise ValueError("Ragged sample coordinates must be flat")
            expected = tuple(len(offsets) - 1 if d == dimensions[0] else shape[0] for d in c.dimensions)
        else:
            expected = tuple(shape[dimensions.index(d)] for d in c.dimensions)
        if tuple(c.shape) != expected:
            raise ValueError("Coordinate shape does not match its dimensions")
        if np.dtype(c.dtype).kind not in "biuf":
            raise ValueError("Coordinates must be finite real numbers")
        key = f"coordinate_{i}"
        add_array(key, c.dtype, expected)
        described.append({"name": c.name, "dimensions": list(c.dimensions), "unit": c.unit, "array": key})
    if ragged:
        if not any(c.dimensions == (dimensions[0],) for c in coordinates) or not any(
                c.dimensions == ("sample",) for c in coordinates):
            raise ValueError("Ragged values need outer and sample coordinates")
    elif any(not any(c.name == d and d in c.dimensions for c in coordinates) for d in dimensions):
        raise ValueError("Every value dimension needs a named coordinate")
    return {"format_version": FORMAT_VERSION, "dimensions": list(dimensions),
            "value_name": value_name, "value_unit": value_unit, "label": label,
            "source": source, "input_name": input_name, "coordinates": described,
            "arrays": arrays, "ragged": ragged}


def _pack_scan_chunks(stream, header, chunks, offsets):
    """Fill declared payload sections from bounded, possibly interleaved chunks.

    Seeking lets one staged window supply values and dependent coordinates in one
    visit. Every section must be filled exactly before the file can be published.
    """
    encoded = json.dumps(header, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_HEADER:
        raise ValueError("Dataset header is too large")
    stream.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
    start = stream.tell()
    arrays = header["arrays"]
    written = dict.fromkeys(arrays, 0)
    keys = {None: "value", **{c["name"]: c["array"] for c in header["coordinates"]}}

    def write(key, values):
        array = _numeric(values)
        spec = arrays[key]
        if array.dtype != np.dtype(spec["dtype"]) or written[key] + array.nbytes > spec["bytes"]:
            raise ValueError("Scan chunk dtype or byte count does not match its layout")
        if key.startswith("coordinate_") and not np.all(np.isfinite(array)):
            raise ValueError("Coordinates must be finite real numbers")
        stream.seek(start + spec["offset"] + written[key])
        stream.write(array.tobytes(order="C"))
        written[key] += array.nbytes

    if offsets is not None:
        write("offsets", np.asarray(offsets, dtype="<i8"))
    for name, values in chunks:
        if name not in keys:
            raise ValueError(f"Unknown scan chunk coordinate: {name!r}")
        write(keys[name], values)
    if any(written[key] != spec["bytes"] for key, spec in arrays.items()):
        raise ValueError("Incomplete scan payload")
    stream.truncate(start + sum(spec["bytes"] for spec in arrays.values()))


def _read_header(stream):
    """Read the bounded file directory without loading the numerical arrays."""
    if stream.read(len(MAGIC)) != MAGIC:
        raise ValueError("Not a V2 self-describing dataset")
    length_bytes = stream.read(8)
    if len(length_bytes) != 8:
        raise ValueError("Truncated dataset header")
    length = struct.unpack("<Q", length_bytes)[0]
    if length > MAX_HEADER:
        raise ValueError("Oversized dataset header")
    encoded = stream.read(length)
    if len(encoded) != length:
        raise ValueError("Truncated dataset header")
    header = json.loads(encoded)
    if not isinstance(header, dict) or header.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported dataset version")
    if not isinstance(header.get("arrays"), dict) or "value" not in header["arrays"]:
        raise ValueError("Invalid dataset header")
    return header


def _unpack_dataset(path):
    """Validate all array bounds and relationships before exposing a recovered file."""
    with open(path, "rb") as stream:
        header = _read_header(stream)
        start = stream.tell()
        size = os.fstat(stream.fileno()).st_size - start
        arrays = {}
        cursor = 0
        for key, spec in header["arrays"].items():
            dtype = np.dtype(spec["dtype"])
            shape = tuple(spec["shape"])
            count = int(np.prod(shape, dtype=np.int64))
            if (dtype.kind not in "biufc" or dtype.hasobject or any(type(n) is not int or n < 0 for n in shape)
                    or spec["offset"] != cursor or count * dtype.itemsize != spec["bytes"]):
                raise ValueError("Invalid array descriptor")
            stream.seek(start + cursor)
            raw = stream.read(spec["bytes"])
            if len(raw) != spec["bytes"]:
                raise ValueError("Truncated dataset array")
            arrays[key] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            cursor += spec["bytes"]
        if cursor != size:
            raise ValueError("Dataset length does not match header")
    coordinates = tuple(Coordinate(c["name"], arrays[c["array"]], tuple(c["dimensions"]), c["unit"])
                        for c in header["coordinates"])
    data = arrays["value"]
    dimensions = tuple(header["dimensions"])
    ragged = header["ragged"]
    axes = tuple(Axis(c.name, c.values.tolist(), c.unit) for c in coordinates
                 if c.dimensions == (c.name,) and c.name in dimensions)
    result = Dataset(data, axes, coordinates, arrays.get("offsets"), dimensions,
                     header["value_name"], header["value_unit"], header["label"], header["source"],
                     header.get("input_name"))
    result = _validated_dataset(result)
    if ragged != (result.offsets is not None):
        raise ValueError("Invalid ragged layout")
    return result


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
        """Release the writer lock."""
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def list_datasets(self):
        """List only datasets present in the published manifest."""
        return [(g, n) for g, entries in self._manifest["datasets"].items() for n in entries]

    def read_manifest(self) -> dict:
        """Return an independent snapshot of the published dataset index."""
        return json.loads(json.dumps(self._manifest))

    def write_dataset(self, group: str, name: str, data, axes=(), *, coordinates=None,
                      offsets=None, dimensions=None, value_name="value", value_unit=None,
                      label=None, source=None, input_name=None):
        """Publish a complete self-describing dataset and then its manifest entry."""
        with self._dataset_file(group, name) as stream:
            dataset = Dataset(np.asarray(data), tuple(axes), tuple(coordinates or ()), offsets,
                              tuple(dimensions) if dimensions is not None else (),
                              value_name, value_unit, label, source, input_name)
            _pack_dataset(stream, dataset)

    def _write_dataset_chunks(self, group, name, *, shape, dtype, dimensions,
                              coordinates, chunks, offsets=None, value_name="value",
                              value_unit=None, label=None, source=None, input_name=None):
        """Publish a private scan layout from window-sized numerical chunks.

        Chunk keys are None for values, or a coordinate name. Chunks may interleave
        arrays, but must fill each array in C order with its declared dtype. Small
        ragged offsets are supplied separately as row metadata. No payload snapshot
        or scan-sized validation array is constructed here.
        """
        with self._dataset_file(group, name) as stream:
            header = _scan_header(shape, dtype, dimensions, coordinates, offsets,
                                  value_name, value_unit, label, source, input_name)
            _pack_scan_chunks(stream, header, chunks, offsets)

    @contextmanager
    def _dataset_file(self, group, name):
        """Share exclusive, temporary-file publication between ordinary and scan writes.

        Sync file contents and the final directory entry before publishing the manifest.
        A directory-sync or manifest failure can leave a complete but unlisted file.
        """
        if self.mode == "r" or self._lock is None:
            raise PermissionError("Store is not open for writing")
        group, name = _component(group), _component(name)
        if name in self._manifest["datasets"].get(group, {}):
            raise FileExistsError(f"Dataset already exists: {group}/{name}")
        directory = self.path / group
        directory.mkdir(exist_ok=True)
        target = directory / f"{name}.dat"
        if target.exists():
            raise FileExistsError(target)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".array-", delete=False) as stream:
                tmp = Path(stream.name)
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
            tmp = None
            # File fsync preserves bytes, not the rename in its parent directory.
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            updated = {**self._manifest, "datasets": {**self._manifest["datasets"]}}
            updated["datasets"][group] = {**updated["datasets"].get(group, {}), name: {"file": f"{group}/{name}.dat"}}
            self._publish_manifest(updated)
            self._manifest = updated
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def write_run_metadata(self, metadata: dict):
        """Atomically save JSON-compatible run settings separately from datasets."""
        if self.mode == "r" or self._lock is None:
            raise PermissionError("Store is not open for writing")
        if not isinstance(metadata, dict):
            raise TypeError("Run metadata must be a dictionary")
        encoded = json.dumps(metadata, allow_nan=False, indent=2).encode("utf-8")
        if len(encoded) > MAX_HEADER:
            raise ValueError("Run metadata exceeds size limit")
        if (self.path / "metadata.json").exists():
            raise FileExistsError("Run metadata already exists")
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path, prefix=".metadata-", delete=False) as stream:
                tmp = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path / "metadata.json")
            tmp = None
            fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def read_run_metadata(self):
        """Read the optional run-settings snapshot, independent of the manifest."""
        with (self.path / "metadata.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    def read_dataset(self, group: str, name: str) -> Dataset:
        """Read and validate a published self-contained file."""
        group, name = _component(group), _component(name)
        entry = self._manifest["datasets"][group][name]
        if entry["file"] != f"{group}/{name}.dat":
            raise ValueError("Invalid dataset path")
        return self.read_file(self.path / entry["file"])

    @staticmethod
    def read_file(path: str | Path) -> Dataset:
        """Recover a dataset directly from its .dat file without a manifest."""
        return _unpack_dataset(path)

    @staticmethod
    def read_file_header(path: str | Path) -> dict:
        """Inspect a dataset's small self-describing header without loading array bytes.

        The header is useful for listing datasets; read_file/read_dataset fully validate
        the arrays before numerical use.
        """
        with open(path, "rb") as stream:
            return _read_header(stream)

    def _publish_manifest(self, content=None):
        """Atomically replace the index after the data file has been synced."""
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path, prefix=".manifest-",
                                             encoding="utf-8", delete=False) as stream:
                tmp = Path(stream.name)
                json.dump(content if content is not None else self._manifest, stream, allow_nan=False, indent=2)
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
