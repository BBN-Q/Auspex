"""One spawned worker hosting a built-in filter; no filter object crosses processes."""

from __future__ import annotations

import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time

import numpy as np

from ..data import Axis
from .averager import ComplexAverager
from .core import Batch, GroupComplete, Passthrough, PipelineError, PortSchema, WindowState
from .transport import SharedEdge, _receive, _send

FILTER_TYPES = {"passthrough": Passthrough, "complex_average": ComplexAverager}


def _worker(kind, schema, config, input_edge, outputs, status, control):
    """Construct the filter inside its child and preserve per-edge event order."""
    try:
        processor = FILTER_TYPES[kind](schema, **json.loads(config))
        _send(status, {"type": "ready"})
        state = WindowState()
        supervisor_pid = os.getppid()
        while True:
            ready = wait([input_edge.reader_connection, control], timeout=1)
            if not ready:
                if os.getppid() != supervisor_pid:
                    return  # the Filter Server vanished; do not leave an orphan worker
                continue
            if input_edge.reader_connection not in ready and control in ready:
                request = _receive(control)
                try:
                    state.can_update()
                    processor.configure(request["changes"])
                    state.revision += 1
                    _send(control, {"type": "updated", "revision": state.revision})
                except (KeyError, TypeError, ValueError, PipelineError) as exc:
                    _send(control, {"type": "rejected", "error": str(exc)})
                continue
            event, payload, slot = input_edge.receive()
            if event == "stop":
                return
            try:
                if event == "batch":
                    processor.input_schema.check(payload)
                    state.admit(payload)
                    emitted = processor.on_batch(payload)
                elif event == "window_end":
                    state.end_window(payload["window"])
                    emitted = processor.on_window_end(payload["window"])
                elif event == "run_end":
                    state.finish()
                    emitted = processor.on_run_end()
                else:
                    raise PipelineError("Unknown edge event")
                for item in emitted:
                    if isinstance(item, GroupComplete):
                        for port, output in outputs.items():
                            if port != "partial":
                                output.send_event("group_end", item.group)
                    else:
                        if item.port not in outputs:
                            raise PipelineError("Undeclared output port")
                        processor.output_schemas[item.port].check(item.batch)
                        outputs[item.port].send_batch(item.batch, best_effort=item.port == "partial")
                if event != "batch":
                    for port, output in outputs.items():
                        if port != "partial":  # previews are snapshots, never required completion
                            output.send_event(event, payload["group"], payload["window"]) 
            finally:
                if slot is not None:
                    input_edge.release(slot)  # emitted views may refer to this input slot
    except Exception as exc:
        _send(status, {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        status.close()
        control.close()


class FilterWorker:
    """Supervise one child and own its input/output shared-memory edges."""

    def __init__(self, kind: str, schema: PortSchema, *, config=None, slots=2,
                 slot_bytes=64_000):
        if kind not in FILTER_TYPES:
            raise ValueError("Unknown built-in filter")
        config_json = json.dumps(config or {}, allow_nan=False)  # never pickle filter configuration
        self.context = mp.get_context("spawn")
        self.input = SharedEdge(slots=slots, slot_bytes=slot_bytes, context=self.context)
        ports = ("out",) if kind == "passthrough" else ("partial", "final")
        self.outputs = {port: SharedEdge(slots=slots, slot_bytes=slot_bytes, context=self.context)
                        for port in ports}
        self.status_parent, status_child = self.context.Pipe(duplex=False)
        self.control_parent, control_child = self.context.Pipe(duplex=True)
        self.process = self.context.Process(target=_worker,
                    args=(kind, schema, config_json, self.input, self.outputs, status_child, control_child))
        self._started = False
        self.state = WindowState()
        self._boundary_ready = True
        self._group_completed = False
        try:
            self.process.start()
            self._started = True
            status_child.close()
            control_child.close()
            if not self.status_parent.poll(5):
                raise PipelineError("Filter worker failed to become ready")
            response = _receive(self.status_parent)
            if response["type"] != "ready":
                raise PipelineError(response.get("error", "Worker startup failed"))
        except BaseException:
            status_child.close()
            control_child.close()
            self.close()
            raise

    def check(self):
        """Report worker failure before waiting indefinitely for a required result."""
        if self.status_parent.poll():
            try:
                message = _receive(self.status_parent)
            except EOFError as exc:
                raise PipelineError("Filter worker closed its status channel") from exc
            if message.get("type") == "error":
                raise PipelineError(message["error"])
        if not self.process.is_alive():
            raise PipelineError("Filter worker exited")

    def push(self, batch):
        """Validate and enqueue a batch; no configuration may overtake pending input."""
        self.state.admit(batch)
        self._boundary_ready = False
        self.input.send_batch(batch)

    def end_window(self, window):
        """Enqueue an experiment-owned window boundary on its ordered input edge."""
        self.state.end_window(window)
        self.input.send_event("window_end", None, window)

    def finish(self):
        """Request worker validation of any unfinished filter-owned group."""
        self.state.finish()
        self.input.send_event("run_end", None)

    def configure(self, changes, *, timeout=5):
        """Request a revision only after the previous group's output has completed."""
        if not self._boundary_ready:
            raise PipelineError("Wait for required group completion before changing parameters")
        _send(self.control_parent, {"changes": changes})
        deadline = time.monotonic() + timeout
        while not self.control_parent.poll(0.05):
            self.check()
            if time.monotonic() >= deadline:
                raise PipelineError("Timed out waiting for filter configuration")
        reply = _receive(self.control_parent)
        if reply.get("type") != "updated":
            raise PipelineError(reply.get("error", "Filter rejected configuration"))
        self.state.revision = reply["revision"]
        return self.state.revision

    def receive(self, port, *, timeout=5):
        """Wait for one output, copying it before returning the shared slot."""
        edge = self.outputs[port]
        deadline = time.monotonic() + timeout
        while not edge.reader_connection.poll(0.05):
            self.check()
            if time.monotonic() >= deadline:
                raise PipelineError(f"Timed out waiting for {port}")
        event, payload, slot = edge.receive()
        if slot is None:
            if port == "final" and event == "group_end":
                self._group_completed = True
            if port == "final" and event == "window_end" and self._group_completed:
                self._boundary_ready = True
                self._group_completed = False
            return event, payload
        try:
            snapshot = Batch(payload.group, payload.window, payload.sequence, payload.revision,
                np.array(payload.data, copy=True),
                tuple(Axis(a.name, np.array(a.points, copy=True), a.unit) for a in payload.axes),
                payload.count, payload.expected_count)
            return event, snapshot
        finally:
            edge.release(slot)

    def close(self):
        """Terminate if needed and reclaim owned memory even after worker failure."""
        if self._started:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()
            else:
                self.process.join(timeout=5)
        self.status_parent.close()
        self.control_parent.close()
        self.input.close()
        for edge in self.outputs.values():
            edge.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
