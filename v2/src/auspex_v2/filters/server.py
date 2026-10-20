"""Per-run, thread-supervised spawned filter DAG (one acquisition ingress).

Public submission is serialized, but does not wait for downstream completion.
Arrays travel directly between workers. Only terminal results enter the supervisor.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
import queue
import re
import threading
import time

import numpy as np

from ..data import Axis
from .core import GroupComplete, PipelineError, PortSchema, WindowState
from .graph_transport import GraphEdge
from .transport import _receive, _send
from .worker import FILTER_TYPES
from .store_owner import store_owner
from .inspection import GraphInspector
from .buffer import BufferSink

# Keep these in sync with the supported worker constructors; the inspector must
# show effective values even when NodeSpec omits optional constructor arguments.
FILTER_DEFAULTS = {"passthrough": {},
                   "complex_average": {"repeats": 5, "group_name": "average"}}


@dataclass(frozen=True)
class NodeSpec:
    """Plain construction recipe; no live filter instance crosses a process boundary."""

    kind: str
    schema: PortSchema
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WriterSpec:
    """Required sink in one run folder: prefix + scan ID, or an exact dataset name."""

    node: str
    port: str
    group: str = "data"
    prefix: str = ""
    max_scan_bytes: int = 64 * 1024 * 1024
    max_scan_windows: int = 8192
    dataset: str | None = None  # Exact name; None retains legacy prefix + scan naming.


@dataclass(frozen=True)
class ResultSpec:
    """A named terminal port, optionally an explicitly lossy preview sink."""

    node: str
    port: str
    preview: bool = False


def _snapshot(batch):
    return replace(batch, data=np.array(batch.data, copy=True),
                   axes=tuple(Axis(a.name, np.array(a.points, copy=True), a.unit)
                              for a in batch.axes))


def _graph_worker(name, spec, input_edge, outputs, status, control, stop):
    """Run one sequential processor; downstream backpressure is cancellable."""
    context = "startup"
    try:
        processor = FILTER_TYPES[spec.kind](spec.schema, **spec.config)
        revision = 0
        supervisor_pid = os.getppid()
        _send(status, {"type": "ready"})

        def emit(items):
            for item in items:
                if isinstance(item, GroupComplete):
                    forward("group_end", item.group)
                else:
                    processor.output_schemas[item.port].check(item.batch)
                    batch = replace(item.batch, revision=revision)
                    for edge, preview in outputs.get(item.port, ()):
                        edge.send_batch(batch, best_effort=preview)

        def forward(event, group=None, window=None):
            for destinations in outputs.values():
                for edge, preview in destinations:
                    if not preview:
                        edge.send_event(event, group, window)

        while not stop.is_set():
            if os.getppid() != supervisor_pid:
                stop.set()
                return
            if control.poll():
                request = _receive(control)
                try:
                    processor.configure(request["changes"])
                    revision += 1
                    _send(control, {"type": "updated", "revision": revision})
                except (PipelineError, ValueError, TypeError, KeyError) as exc:
                    _send(control, {"type": "rejected", "error": str(exc)})
            if not input_edge.reader_connection.poll(0.01):
                continue
            event, payload, slot = input_edge.receive()
            context = (f"window {payload.window}" if event == "batch" else
                       f"{event} window={payload['window']} group={payload['group']}")
            try:
                if event == "batch":
                    processor.input_schema.check(payload)
                    emit(processor.on_batch(payload))
                elif event == "window_end":
                    emit(processor.on_window_end(payload["window"]))
                elif event == "group_end":
                    emit(processor.on_group_end(payload["group"]))
                elif event == "scan_end":
                    emit(processor.on_scan_end(payload["group"]))
                elif event == "run_end":
                    emit(processor.on_run_end())
                else:
                    raise PipelineError(f"Unknown event: {event}")
                if event != "batch":
                    forward(event, payload["group"], payload["window"])
                # Remain alive after run end until the supervisor drains terminals.
            finally:
                if slot is not None:
                    input_edge.release(slot)
    except Exception as exc:
        if not stop.is_set():
            _send(status, {"type": "error", "error": f"{name} ({context}): {type(exc).__name__}: {exc}"})
    finally:
        status.close()
        control.close()
        # Never wait for queue feeder threads after a failed/cancelled graph.
        input_edge.releases.cancel_join_thread()
        for destinations in outputs.values():
            for edge, _ in destinations:
                edge.notifications.cancel_join_thread()


class FilterServer:
    """Own a bounded DAG and supervise it independently of blocked callers.

    `finish()` submits run end; `wait_finished()` confirms drainage. Consumers must
    drain required results concurrently when they exceed `result_capacity`.
    `close()` cancels an unfinished run; it never manufactures successful completion.
    """

    def __init__(self, schema, nodes, edges, results, *, writers=None, buffer_sinks=None, store_path=None,
                 run_metadata=None, input_name=None, inspect_socket=None,
                 backpressure=True, slots=2, result_capacity=2, max_windows=8,
                 startup_timeout=10, memory_budget_bytes=64 * 1024 * 1024,
                 store_scan_budget_bytes=256 * 1024 * 1024):
        self.schema = schema
        self.input_name = input_name if input_name is not None else schema.value_name
        if not isinstance(self.input_name, str) or not self.input_name:
            raise ValueError("Pipeline input needs a name")
        self.nodes = {name: replace(spec, config=json.loads(json.dumps(spec.config, allow_nan=False)))
                      for name, spec in nodes.items()}
        self.edges = list(edges)
        self.results = dict(results)
        self.buffer_specs = dict(buffer_sinks or {})
        if self.results.keys() & self.buffer_specs.keys():
            raise PipelineError("Buffer and result names must be distinct")
        self.buffer_sinks = {name: BufferSink(name, spec.capacity) for name, spec in self.buffer_specs.items()}
        self.results.update({name: ResultSpec(spec.node, spec.port) for name, spec in self.buffer_specs.items()})
        self.writers = dict(writers or {})
        self.store_path = store_path
        self.run_metadata = (json.loads(json.dumps(run_metadata, allow_nan=False))
                             if run_metadata is not None else None)
        if self.run_metadata is not None and (not isinstance(self.run_metadata, dict) or not self.writers):
            raise ValueError("Run metadata requires a store and a mapping")
        if bool(self.writers) != (store_path is not None):
            raise ValueError("Writers require a store path, and vice versa")
        if min(slots, result_capacity, max_windows) < 1 or type(backpressure) is not bool:
            raise ValueError("Positive capacities and boolean backpressure are required")
        self._validate()
        if store_scan_budget_bytes < 1 or sum(w.max_scan_bytes for w in self.writers.values()) > store_scan_budget_bytes:
            raise PipelineError("Writer staging exceeds scan budget")
        edge_bytes = sum(self.nodes[c].schema.max_bytes * slots for _, _, c in self.edges)
        edge_bytes += sum(self.nodes[r.node].schema.max_bytes * slots for r in self.results.values())
        edge_bytes += sum(self.nodes[w.node].schema.max_bytes * slots for w in self.writers.values())
        retained_bytes = sum(self.nodes[r.node].schema.max_bytes *
                             (self.buffer_specs[name].capacity + 1 if name in self.buffer_specs else result_capacity)
                             for name, r in self.results.items())
        if edge_bytes + retained_bytes > memory_budget_bytes:
            raise PipelineError("Graph transport and result retention exceed memory budget")
        self.backpressure = backpressure
        self.slots = slots
        self.max_windows = max_windows
        self.context = mp.get_context("spawn")
        self.stop = self.context.Event()
        self.condition = threading.Condition()
        self.admission = threading.Lock()
        self.state = WindowState()
        self.error = None
        self.lifecycle = "STARTING"
        self.completed_window = -1
        self.revisions = {name: 0 for name in nodes}
        self.current_configs = {name: {**FILTER_DEFAULTS[spec.kind], **spec.config}
                                for name, spec in self.nodes.items()}
        self.inspector = None
        self.buffers = {name: queue.Queue(maxsize=result_capacity) for name in results}
        self.buffers.update(self.buffer_sinks)
        self.processes = {}
        self.controls = {}
        self.channels = []
        self.ready = threading.Event()
        self.failed = threading.Event()
        self.scan_acks = {}
        self.published = []
        self.thread = threading.Thread(target=self._supervise, name="auspex-filter-server")
        self.thread.start()
        if not self.ready.wait(startup_timeout):
            self._fail("Filter graph startup timed out")
            self.close()
        try:
            self.check()
        except PipelineError:
            self.close()
            raise
        if inspect_socket is not None:
            try:
                self.inspector = GraphInspector(inspect_socket, self.inspection_snapshot)
            except BaseException:
                self.close()
                raise

    def inspection_snapshot(self):
        """Return a read-only, JSON-compatible graph and acknowledged configurations."""
        with self.condition:
            nodes = {name: {"kind": spec.kind,
                            "config": json.loads(json.dumps(self.current_configs[name])),
                            "revision": self.revisions[name], "schema": asdict(spec.schema),
                            "inputs": ["in"],
                            "outputs": ["out"] if spec.kind == "passthrough" else ["partial", "final"]}
                     for name, spec in sorted(self.nodes.items())}
            lifecycle = self.lifecycle
        edges = [{"source": producer, "port": port, "target": consumer,
                  "target_port": "in", "preview": False}
                 for producer, port, consumer in self.edges]
        for name, spec in self.results.items():
            edges.append({"source": spec.node, "port": spec.port, "target": f"result:{name}",
                          "target_port": "in", "preview": spec.preview})
        for name, spec in self.writers.items():
            edges.append({"source": spec.node, "port": spec.port, "target": f"writer:{name}",
                          "target_port": "in", "preview": False})
        return {"snapshot_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
                "lifecycle": lifecycle, "input": {"name": self.input_name, "schema": asdict(self.schema)},
                "nodes": nodes, "edges": edges,
                "results": {name: asdict(spec) for name, spec in sorted(self.results.items())},
                "writers": {name: asdict(spec) for name, spec in sorted(self.writers.items())},
                "buffers": {name: asdict(spec) for name, spec in sorted(self.buffer_specs.items())}}

    def _validate(self):
        """Reject unsupported topology before allocating memory or spawning children."""
        if not self.nodes or not (self.writers or any(not r.preview for r in self.results.values())):
            raise PipelineError("Graph needs workers and at least one required result or writer")
        ports = {}
        for name, spec in self.nodes.items():
            if name == "$input" or spec.kind not in FILTER_TYPES:
                raise PipelineError("Unknown filter kind or reserved node name")
            ports[name] = ("out",) if spec.kind == "passthrough" else ("partial", "final")
        incoming = {name: 0 for name in self.nodes}
        used = set()
        for producer, port, consumer in self.edges:
            if consumer not in self.nodes:
                raise PipelineError("Unknown consumer")
            source = self.schema if producer == "$input" else self.nodes.get(producer)
            if source is None or port not in (("out",) if producer == "$input" else ports[producer]):
                raise PipelineError("Unknown producer port")
            source_schema = source if producer == "$input" else source.schema
            if source_schema != self.nodes[consumer].schema:
                raise PipelineError("Incompatible port schemas")
            if port == "partial":
                raise PipelineError("Partial outputs support preview sinks only")
            incoming[consumer] += 1
            used.add((producer, port))
        if any(count != 1 for count in incoming.values()):
            raise PipelineError("Every worker needs exactly one input")
        visited = {"$input"}
        while len(visited) <= len(self.nodes):
            available = {c for p, _, c in self.edges if p in visited and c not in visited}
            if not available:
                raise PipelineError("Cycle or unreachable worker")
            visited.update(available)
        for result in self.results.values():
            if result.node not in ports or result.port not in ports[result.node]:
                raise PipelineError("Unknown result port")
            if (result.port == "partial") != result.preview:
                raise PipelineError("Only partial outputs may be preview sinks")
            used.add((result.node, result.port))
        destinations = set()
        for writer in self.writers.values():
            if writer.node not in ports or writer.port not in ports[writer.node] or writer.port == "partial":
                raise PipelineError("Writer needs a declared required output port")
            if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", writer.group)
                    or writer.group in (".", "..") or
                    (writer.prefix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", writer.prefix)) or
                    (writer.dataset is not None and
                     (not isinstance(writer.dataset, str) or
                      not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", writer.dataset) or writer.prefix)) or
                    type(writer.max_scan_bytes) is not int or writer.max_scan_bytes < 1 or
                    type(writer.max_scan_windows) is not int or writer.max_scan_windows < 1):
                raise PipelineError("Invalid writer destination or capacity")
            destination = (writer.group, writer.dataset if writer.dataset is not None else writer.prefix)
            if destination in destinations:
                raise PipelineError("Writer destinations may collide")
            destinations.add(destination)
            used.add((writer.node, writer.port))
        for name, spec in self.nodes.items():
            required = "out" if spec.kind == "passthrough" else "final"
            if (name, required) not in used:
                raise PipelineError(f"Unconnected required output: {name}.{required}")

    def _fail(self, message):
        with self.condition:
            if self.error is None:
                self.error = str(message)
                self.lifecycle = "FAILED"
            self.failed.set()
            self.stop.set()
            self.condition.notify_all()

    def check(self):
        if self.error is not None:
            raise PipelineError(self.error)

    def _new_edge(self, label, schema):
        edge = GraphEdge(context=self.context, stop=self.stop,
                         backpressure=self.backpressure, label=label,
                         slots=self.slots, slot_bytes=schema.max_bytes)
        self.channels.append(edge)
        return edge

    def _supervise(self):
        """Spawn, drain terminals, detect errors, and always reclaim graph resources."""
        statuses = {}
        child_handles = []
        try:
            outputs = {name: {} for name in self.nodes}
            inputs = {}
            self.ingress = []
            for producer, port, consumer in self.edges:
                edge = self._new_edge(f"{producer}.{port}->{consumer}", self.nodes[consumer].schema)
                inputs[consumer] = edge
                if producer == "$input":
                    self.ingress.append(edge)
                else:
                    outputs[producer].setdefault(port, []).append((edge, False))
            terminals = {}
            for name, spec in self.results.items():
                edge = self._new_edge(f"result:{name}", self.nodes[spec.node].schema)
                terminals[name] = edge
                outputs[spec.node].setdefault(spec.port, []).append((edge, spec.preview))
            writer_edges = {}
            for name, spec in self.writers.items():
                edge = self._new_edge(f"writer:{name}", self.nodes[spec.node].schema)
                writer_edges[name] = edge
                outputs[spec.node].setdefault(spec.port, []).append((edge, False))
            if self.writers:
                owner_status, child_status = self.context.Pipe(duplex=False)
                statuses["$store"] = owner_status
                child_handles.append(child_status)
                process = self.context.Process(target=store_owner, name="auspex-store-owner",
                    args=(self.store_path, self.writers,
                          {n: self.nodes[w.node].schema for n, w in self.writers.items()},
                          writer_edges, child_status, self.stop, self.run_metadata, self.input_name))
                process.start()
                self.processes["$store"] = process
                child_status.close()
            for name, spec in self.nodes.items():
                parent_status, child_status = self.context.Pipe(duplex=False)
                parent_control, child_control = self.context.Pipe()
                statuses[name] = parent_status
                self.controls[name] = parent_control
                child_handles.extend((child_status, child_control))
                process = self.context.Process(target=_graph_worker, name=f"filter:{name}",
                    args=(name, spec, inputs[name], outputs[name], child_status, child_control, self.stop))
                process.start()
                self.processes[name] = process
                child_status.close()
                child_control.close()
            pending = set(statuses)
            while pending and not self.stop.is_set():
                for name in list(pending):
                    if statuses[name].poll(0.01):
                        try:
                            response = _receive(statuses[name])
                        except (EOFError, OSError) as exc:
                            raise PipelineError(f"Worker status closed during startup: {name}") from exc
                        if response["type"] != "ready":
                            raise PipelineError(response.get("error", "Worker startup failed"))
                        pending.remove(name)
                    elif not self.processes[name].is_alive():
                        raise PipelineError(f"Worker exited during startup: {name}")
            if self.stop.is_set():
                return
            self.lifecycle = "RUNNING"
            self.ready.set()
            windows = {name: -1 for name, spec in self.results.items() if not spec.preview}
            windows.update({f"writer:{name}": -1 for name in self.writers})
            finished = set()
            pending_buffers = {}  # A batch is not readable until its ordered window end.
            while not self.stop.is_set():
                for name, connection in statuses.items():
                    if connection.poll():
                        try:
                            response = _receive(connection)
                        except (EOFError, OSError) as exc:
                            raise PipelineError(f"Worker status channel closed: {name}") from exc
                        if name == "$store" and response["type"] == "window":
                            key = f"writer:{response['writer']}"
                            if response["window"] != windows[key] + 1:
                                raise PipelineError("Out-of-order store window")
                            windows[key] = response["window"]
                            with self.condition:
                                self.completed_window = min(windows.values())
                                self.condition.notify_all()
                            continue
                        if name == "$store" and response["type"] == "scan":
                            self._ack_scan(response["scan"])
                            self.published.append(response)
                            continue
                        if name == "$store" and response["type"] == "finished":
                            finished.add("$store")
                            continue
                        raise PipelineError(response.get("error", f"Unexpected status: {name}"))
                    if not self.processes[name].is_alive() and not (name == "$store" and name in finished):
                        raise PipelineError(f"Worker exited: {name}")
                for name, edge in terminals.items():
                    if not edge.reader_connection.poll():
                        continue
                    preview = self.results[name].preview
                    is_batch = json.loads(edge.reader_connection.pending)["type"] == "batch"
                    if is_batch and self.buffers[name].full() and not preview:
                        if not self.backpressure:
                            raise PipelineError(f"Capacity exhausted: result:{name} retention")
                        continue
                    event, payload, slot = edge.receive()
                    if event == "batch":
                        try:
                            if preview and self.buffers[name].full():
                                try:
                                    self.buffers[name].get_nowait()
                                except queue.Empty:
                                    pass  # A client consumed the snapshot concurrently.
                            if name in self.buffer_sinks:
                                if name in pending_buffers:
                                    raise PipelineError(f"Buffer {name} expects one batch per output window")
                                pending_buffers[name] = _snapshot(payload)
                            else:
                                self.buffers[name].put_nowait(_snapshot(payload))
                        finally:
                            edge.release(slot)
                    elif event == "window_end":
                        if payload["window"] != windows[name] + 1:
                            raise PipelineError(f"Out-of-order terminal window: {name}")
                        if name in pending_buffers:
                            batch = pending_buffers.pop(name)
                            if batch.window != payload["window"]:
                                raise PipelineError(f"Buffer {name} window mismatch")
                            self.buffer_sinks[name]._put_completed(batch)
                        windows[name] = payload["window"]
                    elif event == "scan_end" and not preview:
                        self._ack_scan(payload["group"])
                    elif event == "run_end":
                        finished.add(name)
                    with self.condition:
                        self.completed_window = min(windows.values())
                        self.condition.notify_all()
                required = {name for name, spec in self.results.items() if not spec.preview}
                if finished == required | ({"$store"} if self.writers else set()):
                    with self.condition:
                        self.lifecycle = "SUCCEEDED"
                        self.condition.notify_all()
                    return
                self.stop.wait(0.002)
        except Exception as exc:
            self._fail(f"Filter graph: {exc}")
        finally:
            self.stop.set()
            self.ready.set()
            # A shared deadline bounds cooperative cleanup even with many workers.
            deadline = time.monotonic() + 1
            for process in self.processes.values():
                process.join(max(0, deadline - time.monotonic()))
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
            # A cancelled submission must finish touching shared memory first.
            with self.admission:
                for edge in self.channels:
                    edge.close()
            for connection in list(statuses.values()) + list(self.controls.values()) + child_handles:
                connection.close()
            for sink in self.buffer_sinks.values():
                sink._finish(self.error)
            with self.condition:
                self.condition.notify_all()

    def _ack_scan(self, scan):
        """Count terminal drainage and durable writer publication for one scan."""
        with self.condition:
            if scan not in self.scan_acks or self.scan_acks[scan] <= 0:
                raise PipelineError(f"Unexpected scan acknowledgement: {scan}")
            self.scan_acks[scan] -= 1
            self.condition.notify_all()

    def _wait(self, predicate, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while not predicate():
                self.check()
                if deadline is not None and time.monotonic() >= deadline:
                    raise PipelineError("Timed out waiting for graph progress")
                self.condition.wait(0.02)
            self.check()

    def push(self, batch):
        """Copy input before returning; capacity may block, but never the supervisor."""
        with self.admission:
            self.check()
            try:
                new_window = self.state.window is None
                self.schema.check(batch)
                self.state.admit(batch)
                if new_window:
                    available = lambda: batch.window <= self.completed_window + self.max_windows
                    if not self.backpressure and not available():
                        raise PipelineError("Capacity exhausted: outstanding windows")
                    self._wait(available, None)
                for edge in self.ingress:
                    edge.send_batch(batch)
            except Exception as exc:
                self._fail(exc)
                raise

    def end_window(self, window):
        with self.admission:
            self.check()
            try:
                self.state.end_window(window)
                for edge in self.ingress:
                    edge.send_event("window_end", None, window)
            except Exception as exc:
                self._fail(exc)
                raise

    def end_scan(self, scan):
        """Close the runner-owned scan after its last window and await required sinks."""
        with self.admission:
            self.check()
            try:
                self.state.end_scan(scan)
                with self.condition:
                    self.scan_acks[scan] = sum(not spec.preview for spec in self.results.values()) + len(self.writers)
                for edge in self.ingress:
                    edge.send_event("scan_end", scan)
            except Exception as exc:
                self._fail(exc)
                raise

    def wait_scan(self, scan, *, timeout=10):
        """Wait for all required paths, including manifest publication."""
        self._wait(lambda: self.scan_acks.get(scan) == 0, timeout)

    def finish(self):
        """Submit ordered run end without waiting for pending terminal results."""
        with self.admission:
            self.check()
            try:
                self.state.finish()
                self.lifecycle = "DRAINING"
                for edge in self.ingress:
                    edge.send_event("run_end", None)
            except Exception as exc:
                self._fail(exc)
                raise

    def wait_window(self, window, *, timeout=10):
        self._wait(lambda: self.completed_window >= window, timeout)

    def wait_finished(self, *, timeout=10):
        self._wait(lambda: self.lifecycle == "SUCCEEDED", timeout)

    def receive(self, name, *, timeout=10):
        """Consume one owned terminal batch; errors take precedence over stale data."""
        deadline = time.monotonic() + timeout
        while True:
            self._wait(lambda: not self.buffers[name].empty() or self.lifecycle == "SUCCEEDED",
                       max(0, deadline - time.monotonic()))
            try:
                if name in self.buffer_sinks:
                    return self.buffer_sinks[name]._get_nowait()
                return self.buffers[name].get_nowait()
            except queue.Empty as exc:
                if self.lifecycle == "SUCCEEDED":
                    raise PipelineError(f"No more results: {name}") from exc
                # Another consumer or preview replacement won the race; wait again.
                if time.monotonic() >= deadline:
                    raise PipelineError(f"Timed out waiting for result: {name}") from exc

    def configure(self, node, changes, *, timeout=10):
        """Pause admission, drain the admitted prefix, and update one local revision."""
        with self.admission:
            self.check()
            self.state.can_update()
            if node not in self.nodes:
                raise PipelineError("Unknown filter node")
            self.wait_window(self.state.last_window, timeout=timeout)
            connection = self.controls[node]
            changes = json.loads(json.dumps(changes, allow_nan=False))
            _send(connection, {"changes": changes})
            try:
                self._wait(lambda: connection.poll(), timeout)
                reply = _receive(connection)
            except Exception as exc:
                self._fail(f"Unknown configuration outcome for {node}: {exc}")
                raise
            if reply["type"] != "updated":
                raise PipelineError(reply["error"])
            with self.condition:
                self.current_configs[node].update(changes)
                self.revisions[node] = reply["revision"]
            return reply["revision"]

    def cancel(self):
        """Wake blocked producers and waiters; cancellation is not successful run end."""
        if self.lifecycle != "SUCCEEDED":
            self._fail("Filter graph cancelled")

    def close(self):
        if self.inspector is not None:
            self.inspector.close()
            self.inspector = None
        self.cancel()
        self.thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
