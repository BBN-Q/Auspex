"""Spawn-safe bounded shared-memory edge for one producer and one consumer.

Only byte-framed JSON control messages cross pipes; NumPy data and coordinates
stay in reusable shared-memory slots. The creating process owns cleanup.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from ..data import Axis
from .core import Batch, PipelineError

MAX_MESSAGE = 16_384


def _send(connection, message):
    payload = json.dumps(message, allow_nan=False).encode()
    if len(payload) > MAX_MESSAGE:
        raise PipelineError("Filter edge message exceeds limit")
    connection.send_bytes(payload)


def _receive(connection):
    payload = connection.recv_bytes(MAX_MESSAGE)
    return json.loads(payload)


class SharedEdge:
    """Own a fixed number of reusable slots and one ordered duplex message pipe."""

    def __init__(self, *, slots=2, slot_bytes=64_000, context=None):
        if slots < 1 or slot_bytes < 1:
            raise ValueError("Edge capacity must be positive")
        self.context = context or mp.get_context("spawn")
        self.slot_bytes = slot_bytes
        self.slots = [shared_memory.SharedMemory(create=True, size=slot_bytes) for _ in range(slots)]
        self.writer_connection, self.reader_connection = self.context.Pipe(duplex=True)
        self._free = list(range(slots))
        self._closed = False

    def send_batch(self, batch, *, timeout=5, best_effort=False):
        """Publish an array; skip only an explicitly best-effort edge when full."""
        arrays = [np.array(batch.data, order="C", copy=True)]
        arrays.extend(np.array(axis.points, order="C", copy=True) for axis in batch.axes)
        used = sum(array.nbytes for array in arrays)
        if used > self.slot_bytes:
            raise PipelineError("Batch and coordinates exceed edge slot capacity")
        while not self._free:
            if not self.writer_connection.poll(0 if best_effort else timeout):
                if best_effort:
                    return False
                raise PipelineError("Timed out waiting for a free edge slot")
            reply = _receive(self.writer_connection)
            if (reply.get("type") != "released" or
                    not isinstance(reply.get("slot"), int) or
                    reply["slot"] < 0 or reply["slot"] >= len(self.slots) or
                    reply["slot"] in self._free):
                raise PipelineError("Invalid slot release")
            self._free.append(reply["slot"])
        slot = self._free.pop()
        buffer = self.slots[slot].buf
        offset = 0
        specs = []
        for array in arrays:
            size = array.nbytes
            buffer[offset:offset + size] = array.tobytes()
            specs.append({"offset": offset, "shape": list(array.shape),
                          "dtype": array.dtype.str, "bytes": size})
            offset += size
        _send(self.writer_connection, {"type": "batch", "slot": slot, "arrays": specs,
              "axes": [{"name": axis.name, "unit": axis.unit} for axis in batch.axes],
              "group": batch.group, "window": batch.window, "sequence": batch.sequence,
              "revision": batch.revision, "count": batch.count,
              "expected_count": batch.expected_count})
        return True

    def send_event(self, kind, group, window=None):
        """Order completion after previously published batches on this edge."""
        if kind not in ("window_end", "group_end", "run_end", "stop"):
            raise ValueError("Unknown edge event")
        _send(self.writer_connection, {"type": kind, "group": group, "window": window})

    def receive(self):
        """Return (message type, batch or message, slot); caller releases any slot."""
        message = _receive(self.reader_connection)
        if message["type"] != "batch":
            return message["type"], message, None
        slot = message["slot"]
        if not isinstance(slot, int) or slot < 0 or slot >= len(self.slots):
            raise PipelineError("Invalid slot ID")
        arrays = []
        for spec in message["arrays"]:
            dtype = np.dtype(spec["dtype"])
            if dtype.kind not in "biufc" or dtype.hasobject or any(n < 0 for n in spec["shape"]):
                raise PipelineError("Invalid array descriptor")
            offset, size = spec["offset"], spec["bytes"]
            if (offset < 0 or size < 0 or offset + size > self.slot_bytes or
                    int(np.prod(spec["shape"], dtype=np.int64)) * dtype.itemsize != size):
                raise PipelineError("Invalid array bounds")
            arrays.append(np.ndarray(spec["shape"], dtype=dtype,
                                     buffer=self.slots[slot].buf, offset=offset))
        if len(arrays) != len(message["axes"]) + 1:
            raise PipelineError("Coordinate count does not match array rank")
        axes = tuple(Axis(axis["name"], points, axis["unit"])
                     for axis, points in zip(message["axes"], arrays[1:]))
        batch = Batch(message["group"], message["window"], message["sequence"],
                      message["revision"], arrays[0], axes,
                      message["count"], message["expected_count"])
        return "batch", batch, slot

    def release(self, slot):
        """Allow the producer to reuse a consumed slot; views must not be retained."""
        if not isinstance(slot, int) or slot < 0 or slot >= len(self.slots):
            raise PipelineError("Invalid slot release")
        _send(self.reader_connection, {"type": "released", "slot": slot})

    def close(self):
        """Close pipes and unlink all shared memory in the creating process."""
        if not self._closed:
            self.writer_connection.close()
            self.reader_connection.close()
            for slot in self.slots:
                slot.close()
                slot.unlink()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
