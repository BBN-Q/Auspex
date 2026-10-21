"""Bounded, consuming/non-consuming access to completed required outputs."""
from collections import deque
from dataclasses import dataclass, replace
import queue
import threading
import time
import warnings

import numpy as np

from ..data import Axis
from .core import PipelineError


@dataclass(frozen=True)
class BufferSpec:
    node: str
    port: str
    capacity: int = 16


class BufferFullWarning(UserWarning):
    """A required buffer needs a consumer before more output can be retained."""


def snapshot(batch):
    return replace(batch, data=np.array(batch.data, copy=True),
                   axes=tuple(Axis(a.name, np.array(a.points, copy=True), a.unit) for a in batch.axes))


class BufferSink:
    """Retain completed batches in FIFO order, independent of graph lifetime.

    receive consumes one batch; drain consumes all currently retained batches;
    view copies without consuming. No partial outputs are exposed. Producer and
    nonblocking queue operations are private FilterServer implementation details.
    """
    def __init__(self, name, capacity, *, warn_on_full=True):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Buffer capacity must be a positive integer")
        self.name, self.capacity = name, capacity
        self.warn_on_full = warn_on_full
        self._items = deque()
        self._condition = threading.Condition()
        self._done, self._error, self._warned = False, None, False

    def full(self):
        with self._condition:
            return len(self._items) >= self.capacity

    def empty(self):
        with self._condition:
            return not self._items

    def _put_completed(self, batch):
        warn = False
        with self._condition:
            if len(self._items) >= self.capacity:
                raise queue.Full
            self._items.append(snapshot(batch))
            if self.warn_on_full and len(self._items) == self.capacity and not self._warned:
                self._warned = True
                warn = True
            self._condition.notify_all()
        if warn:
            warnings.warn(f"Buffer {self.name!r} is full ({self.capacity} outputs); "
                          "consume with receive()/drain(). Further output may block or fail.",
                          BufferFullWarning, stacklevel=2)

    def _get_nowait(self):
        with self._condition:
            if not self._items:
                raise queue.Empty
            item = self._items.popleft()
            self._warned = False
            return item  # Storage owns copies; removing transfers ownership to this caller.

    def _finish(self, error=None):
        with self._condition:
            self._done, self._error = True, error
            self._condition.notify_all()

    def receive(self, *, timeout=10):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                if self._error is not None:
                    raise PipelineError(self._error)
                if self._done:
                    raise EOFError(f"Buffer {self.name!r} is exhausted")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for buffer {self.name!r}")
                self._condition.wait(remaining)
            return self._get_nowait()

    def view(self):
        with self._condition:
            return [snapshot(batch) for batch in self._items]

    def drain(self):
        with self._condition:
            items, self._items = self._items, deque()
            self._warned = False
            return list(items)

    def clear(self):
        self.drain()
