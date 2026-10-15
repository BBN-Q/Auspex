"""Bounded graph notifications with cancellable waits and shared-memory arrays.

The small queues contain only the existing framed JSON messages, never arrays.
Unlike a pipe send, a full notification queue can be interrupted by cancellation.
"""

import queue

from .core import PipelineError
from .transport import SharedEdge


class _Connection:
    """Adapt two bounded message queues to the edge's framed-message interface."""

    def __init__(self, outgoing, incoming, stop, backpressure, label):
        self.outgoing = outgoing
        self.incoming = incoming
        self.stop = stop
        self.backpressure = backpressure
        self.label = label
        self.pending = None

    def send_bytes(self, payload):
        while not self.stop.is_set():
            try:
                self.outgoing.put(payload, timeout=0.05 if self.backpressure else 0)
                return
            except queue.Full:
                if not self.backpressure:
                    raise PipelineError(f"Capacity exhausted: {self.label} notifications")
        raise PipelineError("Graph cancelled")

    def poll(self, timeout=0):
        if self.pending is not None:
            return True
        try:
            self.pending = self.incoming.get(timeout=timeout)
            return True
        except queue.Empty:
            return False

    def recv_bytes(self, maximum):
        while not self.poll(0.05):
            if self.stop.is_set():
                raise PipelineError("Graph cancelled")
        payload, self.pending = self.pending, None
        if len(payload) > maximum:
            raise PipelineError("Oversized graph message")
        return payload

    def close(self):
        pass  # The supervisor owns the queues.


class GraphEdge(SharedEdge):
    """An edge whose data and boundary notifications both have bounded capacity."""

    def __init__(self, *, context, stop, backpressure, label, slots, slot_bytes):
        super().__init__(context=context, slots=slots, slot_bytes=slot_bytes)
        self.writer_connection.close()
        self.reader_connection.close()
        self.notifications = context.Queue(maxsize=slots + 2)
        self.releases = context.Queue(maxsize=slots)
        self.writer_connection = _Connection(self.notifications, self.releases, stop,
                                             backpressure, label)
        self.reader_connection = _Connection(self.releases, self.notifications, stop,
                                             True, label)
        self.stop = stop
        self.backpressure = backpressure
        self.label = label

    def send_batch(self, batch, *, best_effort=False):
        """Wait for a slot, drop a preview, or fail immediately according to policy."""
        while not self.stop.is_set():
            if super().send_batch(batch, timeout=0, best_effort=True):
                return True
            if best_effort:
                return False
            if not self.backpressure:
                raise PipelineError(f"Capacity exhausted: {self.label} data slots")
            self.stop.wait(0.005)
        raise PipelineError("Graph cancelled")

    def close(self):
        if not self._closed:
            super().close()
            for messages in (self.notifications, self.releases):
                messages.cancel_join_thread()
                messages.close()
