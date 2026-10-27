"""Bounded, optional display snapshots. Never an acquisition/result transport.

One store owner retains map rows before coalescing frames. A private local socket
serves complete frames; clients cannot change processing. Final display caches are
best effort and are not dataset publications or authoritative run status.
"""

from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import tempfile
import threading
import time
import warnings

import numpy as np

from ..data import Axis, Dataset
from ..data.data import _validated_dataset

MAX_PREVIEW_BYTES = 64 * 1024 * 1024
MAX_HEADER = 65536


@dataclass(frozen=True)
class PlotSpec:
    """One optional trace or writer-backed map with an explicit array byte limit."""

    node: str
    port: str
    kind: str = "trace"
    writer: str | None = None
    max_bytes: int = 8 * 1024 * 1024


def validate_plots(plots, writers, schemas):
    """Reject unsupported declarations before workers/files, not during hardware use."""
    if len(plots) > 64 or sum(p.max_bytes for p in plots.values()) > MAX_PREVIEW_BYTES:
        raise ValueError("Plots exceed 64 endpoints or the 64 MiB display-state budget")
    for name, spec in plots.items():
        if (not name or not name.replace("_", "").replace("-", "").replace(".", "").isalnum()
                or spec.kind not in ("trace", "map") or type(spec.max_bytes) is not int or spec.max_bytes < 1):
            raise ValueError("Invalid plot declaration")
        if spec.node not in schemas or schemas[spec.node].rank != 1:
            raise ValueError("Plots require a rank-one trace source")
        if spec.kind == "map":
            writer = writers.get(spec.writer)
            if writer is None or (writer.node, writer.port) != (spec.node, spec.port):
                raise ValueError("Map plots require a writer on the same finalized output")


def planned_sweep(document):
    """Recover the sole fixed sweep from recorded instructions, not author objects."""
    sweeps = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("op") == "SWEEP":
                sweeps.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document["main"])
    if len(sweeps) != 1 or len(sweeps[0]["parameters"]) != 1:
        raise ValueError("Live maps require one independent fixed sweep")
    sweep = sweeps[0]
    name = sweep["parameters"][0]
    points = np.asarray(sweep["points"], dtype=float)
    if len(points) > 1 and not (np.all(np.diff(points) > 0) or np.all(np.diff(points) < 0)):
        raise ValueError("Live map sweep must be strictly monotonic without repeats")
    return Axis(name, points, document["parameters"][name]["unit"])


def encode_frame(dataset, metadata):
    """Serialize owned numeric display arrays; no pickle or executable metadata."""
    coordinates = [{"name": a.name, "unit": a.unit} for a in dataset.axes]
    header = {**metadata, "coordinates": coordinates, "value_name": dataset.value_name,
              "value_unit": dataset.value_unit}
    encoded = json.dumps(header, allow_nan=False)
    if len(encoded.encode()) > MAX_HEADER:
        raise ValueError("Preview metadata too large")
    stream = io.BytesIO()
    np.savez(stream, metadata=np.asarray(encoded), data=dataset.data,
             **{f"axis_{i}": a.points for i, a in enumerate(dataset.axes)})
    return stream.getvalue()


def decode_frame(payload):
    """Restore a detached display Dataset and metadata from one bounded frame."""
    if len(payload) > MAX_PREVIEW_BYTES + MAX_HEADER:
        raise ValueError("Preview frame too large")
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        axes = tuple(Axis(a["name"], arrays[f"axis_{i}"], a["unit"])
                     for i, a in enumerate(metadata["coordinates"]))
        data = arrays["data"]
    dataset = Dataset(data, axes, dimensions=tuple(a.name for a in axes),
                      value_name=metadata["value_name"], value_unit=metadata["value_unit"])
    return _validated_dataset(dataset), metadata


def _receive_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            raise ConnectionError("Preview disconnected")
        result.extend(block)
    return bytes(result)


def request_preview(endpoint, plot=None, *, timeout=2):
    """Fetch a catalog or complete binary frame; GUI-independent and read-only."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(endpoint))
        client.sendall(json.dumps({"version": 1, "plot": plot}).encode() + b"\n")
        with client.makefile("rb") as stream:
            size = struct.unpack("!Q", _receive_exact(stream, 8))[0]
            if size > MAX_PREVIEW_BYTES + MAX_HEADER:
                raise ValueError("Preview reply exceeds limit")
            data = _receive_exact(stream, size)
    return json.loads(data) if plot is None else data


class _Handler(socketserver.StreamRequestHandler):
    """Serve one bounded request; a stalled viewer can occupy only this server thread."""

    def handle(self):
        self.request.settimeout(1)
        try:
            raw = self.rfile.readline(4097)
            if len(raw) > 4096 or not raw.endswith(b"\n"):
                return
            request = json.loads(raw)
            if set(request) != {"version", "plot"} or request["version"] != 1:
                return
            hub = self.server.hub
            with hub.lock:
                payload = (json.dumps(hub.catalog(), allow_nan=False).encode()
                           if request["plot"] is None else hub.frames.get(request["plot"], b""))
            self.wfile.write(struct.pack("!Q", len(payload)) + payload)
        except (OSError, ValueError, TypeError):
            pass  # A broken/slow viewer never propagates into acquisition.


class PreviewHub:
    """Retain bounded trace/map state inside the physical writer owner process."""

    def __init__(self, path, plots, schemas, sweep=None):
        """Reserve declared limits before retaining rows; keep sockets in a private directory."""
        self.path, self.plots, self.schemas = Path(path), plots, schemas
        self.sweep = sweep
        self.frames, self.datasets, self.metadata, self.errors = {}, {}, {}, {}
        self.dirty = set()
        self.map_positions = {}
        self.sequence, self.last_sent = {}, {}
        self.closed = False
        self.lock = threading.RLock()
        self.private = tempfile.TemporaryDirectory(prefix="auspex-preview-")
        self.endpoint = str(Path(self.private.name) / "p.sock")
        self.server = socketserver.UnixStreamServer(self.endpoint, _Handler)
        self.server.hub = self
        os.chmod(self.endpoint, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)
        try:
            self._write_catalog()
            self.thread.start()
        except BaseException:
            self.server.server_close()
            self.private.cleanup()
            raise

    def catalog(self):
        """Describe sources and revisions; no hardware readback or run-success inference."""
        return {"version": 1, "endpoint": self.endpoint, "closed": self.closed,
                "plots": {name: {**asdict(spec), "schema": asdict(self.schemas[spec.node]),
                                  "sequence": self.sequence.get(name, 0),
                                  "error": self.errors.get(name)} for name, spec in self.plots.items()}}

    def _write_catalog(self):
        target = self.path / "preview.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.catalog(), allow_nan=False))
        os.replace(temporary, target)

    def disable(self, name, error):
        """Disable only the affected optional display, retaining an explicit diagnostic."""
        with self.lock:
            self.errors[name] = str(error)
            self.datasets.pop(name, None)
        warnings.warn(f"Plot {name} disabled: {error}", RuntimeWarning)

    def update(self, name, batch):
        """Accept a trace snapshot or finalized row before any display-frame coalescing."""
        if name in self.errors:
            return
        try:
            with self.lock:
                spec = self.plots[name]
                schema = self.schemas[spec.node]
                previous = self.metadata.get(name, {})
                if (batch.window, batch.count or 0) < (previous.get("window", -1), previous.get("count") or 0):
                    return
                if spec.kind == "trace":
                    dataset = Dataset(np.array(batch.data, copy=True), batch.axes,
                                      dimensions=schema.axis_names, value_name=schema.value_name,
                                      value_unit=schema.value_unit)
                else:
                    dataset = self._map_row(name, batch, schema)
                size = dataset.data.nbytes + sum(np.asarray(a.points).nbytes for a in dataset.axes)
                if size > spec.max_bytes:
                    raise ValueError("Display array budget exceeded")
                self.datasets[name] = dataset
                self.dirty.add(name)
                self.metadata[name] = {"scan": batch.scan, "window": batch.window,
                    "sweep": [asdict(p) for p in batch.sweep], "count": batch.count,
                    "expected_count": batch.expected_count, "revision": batch.revision,
                    "received_at": time.time(), "publication": "not acknowledged"}
                self._frame(name)
        except Exception as exc:
            self.disable(name, exc)

    def _map_row(self, name, batch, schema):
        """Update exactly one sweep position; changed grids never masquerade as a heatmap."""
        if len(batch.sweep) != 1:
            raise ValueError("Map preview requires exactly one swept parameter")
        position = batch.sweep[0]
        trace_axis = batch.axes[0]
        old = self.datasets.get(name)
        if old is not None and self.metadata[name]["scan"] != batch.scan:
            old = None
        if old is None:
            self.map_positions[name] = set()
            points = np.asarray(self.sweep.points if self.sweep is not None else [position.value], dtype=float)
            if self.sweep is not None and (self.sweep.name, self.sweep.unit) != (position.name, position.unit):
                raise ValueError("Declared sweep disagrees with acquired coordinates")
            required = len(points) * len(batch.data) * np.dtype(np.result_type(batch.data.dtype, np.float64)).itemsize
            if required + points.nbytes + np.asarray(trace_axis.points).nbytes > self.plots[name].max_bytes:
                raise ValueError("Display array budget exceeded")
            dtype = np.result_type(batch.data.dtype, np.float64)
            missing = complex(np.nan, np.nan) if dtype.kind == "c" else np.nan
            rows = np.full((len(points), len(batch.data)), missing, dtype=dtype)
        else:
            if not np.array_equal(old.axes[1].points, trace_axis.points):
                raise ValueError("Trace grid changed; dependent maps are not supported")
            points, rows = np.asarray(old.axes[0].points), old.data
        if position.value in self.map_positions[name]:
            raise ValueError("Multiple finalized rows at one sweep position are not a regular map")
        indices = np.flatnonzero(points == position.value)
        if not indices.size:
            if self.sweep is not None:
                raise ValueError("Acquired position absent from declared sweep")
            if rows.nbytes + rows.shape[1] * rows.dtype.itemsize + points.nbytes + 8 + np.asarray(trace_axis.points).nbytes > self.plots[name].max_bytes:
                raise ValueError("Display array budget exceeded")
            points = np.r_[points, position.value]
            missing = complex(np.nan, np.nan) if rows.dtype.kind == "c" else np.nan
            rows = np.concatenate((rows, np.full((1, rows.shape[1]), missing, dtype=rows.dtype)))
            indices = [len(points) - 1]
        if len(points) > 1 and not (np.all(np.diff(points) > 0) or np.all(np.diff(points) < 0)):
            raise ValueError("Map sweep must be strictly monotonic")
        rows[indices[0]] = batch.data
        self.map_positions[name].add(position.value)
        return Dataset(rows, (Axis(position.name, points, position.unit), trace_axis),
                       dimensions=(position.name, trace_axis.name), value_name=schema.value_name,
                       value_unit=schema.value_unit)

    def _frame(self, name, force=False):
        """Coalesce display serialization, never the authoritative acquired-row state."""
        now = time.monotonic()
        if not force and now - self.last_sent.get(name, -1e9) < 0.2:
            return
        self.sequence[name] = self.sequence.get(name, 0) + 1
        self.frames[name] = encode_frame(self.datasets[name],
                                        {**self.metadata[name], "sequence": self.sequence[name]})
        self.last_sent[name] = now
        self.dirty.discard(name)

    def flush(self):
        """Publish the newest dirty state even if acquisition pauses after a burst."""
        with self.lock:
            for name in list(self.dirty):
                if name not in self.errors:
                    try:
                        self._frame(name)
                    except Exception as exc:
                        self.disable(name, exc)

    def published(self, writer):
        """Publication acknowledgment is separate from having acquired all visible rows."""
        with self.lock:
            for name, spec in self.plots.items():
                if spec.writer == writer and name in self.metadata:
                    self.metadata[name]["publication"] = "published"
                    self.dirty.add(name)

    def close(self):
        """Cache final bounded displays for late clients; cache failure is not run failure."""
        try:
            with self.lock:
                self.closed = True
                cache = self.path / ".preview"
                cache.mkdir(exist_ok=True)
                for name in self.datasets:
                    self._frame(name, force=True)
                    (cache / f"{name}.npz").write_bytes(self.frames[name])
                self._write_catalog()
        except Exception as exc:
            warnings.warn(f"Final preview cache unavailable: {exc}", RuntimeWarning)
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
            self.private.cleanup()
