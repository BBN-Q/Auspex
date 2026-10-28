"""Standalone storage and whole-window manual processing, no instrument services."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
import warnings

import numpy as np

from auspex_v2.data import Axis, AuspexStore
from auspex_v2.filters import (Batch, BufferFullWarning, BufferSpec, FilterServer,
    ManualPipeline, NodeSpec, PipelineError, PortSchema, WriterSpec)
from auspex_v2.filters.buffer import BufferSink


SCHEMA = PortSchema("complex128", 1, ("frequency",), ("Hz",), 1024)
AXES = (Axis("frequency", np.array([1., 2.]), "Hz"),)
DATA = np.array([1 + 2j, 3 + 4j])
NODES = {"pass": NodeSpec("passthrough", SCHEMA)}
EDGES = [("$input", "out", "pass")]


def batch(window=0):
    return Batch(None, window, 0, 0, DATA, AXES)


class StandaloneStoreTest(unittest.TestCase):
    def test_manual_store_and_buffer_to_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.auspex"
            with AuspexStore(path, mode="w") as store:
                store.write_run_metadata({"operator": "test", "source": "ordinary Python"})
                store.write_dataset("data", "trace", DATA, AXES)
            with AuspexStore(path) as store:
                np.testing.assert_array_equal(store.read_dataset("data", "trace").data, DATA)
                self.assertEqual(store.read_run_metadata()["source"], "ordinary Python")
            with ManualPipeline(SCHEMA, NODES, EDGES,
                                buffers={"memory": BufferSpec("pass", "out")}) as pipeline:
                pipeline.submit(DATA, axes=AXES)
            with AuspexStore(path, mode="a") as store:
                result = pipeline.buffers["memory"].receive()
                store.write_dataset("processed", "trace", result.data, result.axes)
            with AuspexStore(path) as store:
                self.assertEqual(len(store.list_datasets()), 2)


class BufferTest(unittest.TestCase):
    def test_copies_fifo_view_drain_and_warning_episodes(self):
        sink = BufferSink("test", 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BufferFullWarning)
            sink._put_completed(batch(0))
            sink._put_completed(batch(1))
            self.assertEqual(len(caught), 1)
            view = sink.view()
            view[0].data[:] = 99
            view[0].axes[0].points[:] = 99
            np.testing.assert_array_equal(sink.view()[0].data, DATA)
            np.testing.assert_array_equal(sink.view()[0].axes[0].points, AXES[0].points)
            self.assertEqual(sink.receive().window, 0)
            sink._put_completed(batch(2))
            self.assertEqual(len(caught), 2)
            self.assertEqual([b.window for b in sink.drain()], [1, 2])
        sink._finish()
        with self.assertRaises(EOFError): sink.receive()

    def test_buffer_exposes_only_window_completed_data(self):
        with FilterServer(SCHEMA, NODES, EDGES, {},
                          buffer_sinks={"memory": BufferSpec("pass", "out")}) as graph:
            graph.push(batch())
            with self.assertRaises(TimeoutError): graph.buffer_sinks["memory"].receive(timeout=0.1)
            graph.end_window(0)
            result = graph.buffer_sinks["memory"].receive(timeout=5)
            np.testing.assert_array_equal(result.data, DATA)
            graph.finish()
            graph.wait_finished()

    def test_backpressure_warns_then_resumes_without_loss(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BufferFullWarning)
            with ManualPipeline(SCHEMA, NODES, EDGES, buffers={"memory": BufferSpec("pass", "out", 1)}) as pipe:
                pipe.submit(DATA, axes=AXES)
                pipe.submit(DATA * 2, axes=AXES)
                sink = pipe.buffers["memory"]
                deadline = time.monotonic() + 5
                while not sink.full() and time.monotonic() < deadline: time.sleep(0.01)
                self.assertTrue(sink.full())
                np.testing.assert_array_equal(sink.receive().data, DATA)
                np.testing.assert_array_equal(sink.receive(timeout=5).data, DATA * 2)
            self.assertTrue(any(isinstance(w.message, BufferFullWarning) for w in caught))

    def test_overflow_fails_without_backpressure(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BufferFullWarning)
            graph = FilterServer(SCHEMA, NODES, EDGES, {}, backpressure=False,
                                 buffer_sinks={"memory": BufferSpec("pass", "out", 1)})
            try:
                graph.push(batch(0))
                graph.end_window(0)
                graph.wait_window(0)
                graph.push(batch(1))
                graph.end_window(1)
                self.assertTrue(graph.failed.wait(5))
                with self.assertRaises(PipelineError): graph.check()
                self.assertEqual(len(graph.buffer_sinks["memory"].view()), 1)
            finally:
                graph.close()

    def test_invalid_preview_and_budget(self):
        nodes = {"avg": NodeSpec("complex_average", SCHEMA)}
        edges = [("$input", "out", "avg")]
        with self.assertRaises(PipelineError):
            FilterServer(SCHEMA, nodes, edges, {}, buffer_sinks={"bad": BufferSpec("avg", "partial")})
        with self.assertRaises(PipelineError):
            FilterServer(SCHEMA, NODES, EDGES, {}, memory_budget_bytes=1024,
                         buffer_sinks={"large": BufferSpec("pass", "out", 100)})


class ManualPipelineTest(unittest.TestCase):
    def test_average_completed_endpoint_and_lifetime(self):
        nodes = {"avg": NodeSpec("complex_average", SCHEMA, {"repeats": 2})}
        with ManualPipeline(SCHEMA, nodes, [("$input", "out", "avg")],
                            buffers={"average": BufferSpec("avg", "final")}) as pipe:
            pipe.submit(DATA, axes=AXES)
            self.assertEqual(pipe.buffers["average"].view(), [])
            pipe.submit(DATA * 3, axes=AXES)
        output = pipe.buffers["average"].receive()
        np.testing.assert_array_equal(output.data, DATA * 2)
        self.assertEqual(output.count, 2)
        with self.assertRaises(EOFError): pipe.buffers["average"].receive()
        with self.assertRaises(RuntimeError): pipe.submit(DATA, axes=AXES)

    def test_explicit_existing_store_handoff_preserves_server_metadata(self):
        """An append-only owner publishes into a preallocated store without replacing metadata."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claimed.auspex"
            with AuspexStore(path, mode="w") as store:
                store.write_run_metadata({"name": "server-owned"})
            with ManualPipeline(SCHEMA, NODES, EDGES,
                                writers={"disk": WriterSpec("pass", "out")},
                                store_path=path, existing_store=True) as pipe:
                pipe.submit(DATA, axes=AXES)
            with AuspexStore(path) as store:
                self.assertEqual(store.read_run_metadata(), {"name": "server-owned"})
                self.assertEqual(store.list_datasets(), [("data", "sweep_000")])
            with self.assertRaises(PipelineError):
                ManualPipeline(SCHEMA, NODES, EDGES,
                               writers={"disk": WriterSpec("pass", "out")},
                               store_path=Path(directory) / "missing.auspex", existing_store=True)
            with self.assertRaises(ValueError):
                ManualPipeline(SCHEMA, NODES, EDGES,
                               writers={"disk": WriterSpec("pass", "out")},
                               store_path=path, existing_store=True, run_metadata={"wrong": True})

    def test_writer_and_buffer_multiple_scans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            with ManualPipeline(SCHEMA, NODES, EDGES,
                                writers={"disk": WriterSpec("pass", "out")},
                                buffers={"memory": BufferSpec("pass", "out")}, store_path=path) as pipe:
                for bias in (0., 1.):
                    pipe.submit(DATA, axes=AXES, sweep={"bias": bias}, sweep_units={"bias": "V"})
                self.assertEqual(pipe.end_scan(), "sweep_000")
                with AuspexStore(path) as store:
                    self.assertEqual(store.list_datasets(), [("data", "sweep_000")])
                pipe.submit(DATA, axes=AXES, sweep={"bias": 2.}, sweep_units={"bias": "V"})
            self.assertEqual([b.window for b in pipe.buffers["memory"].drain()], [0, 1, 2])
            with AuspexStore(path) as store:
                self.assertEqual(len(store.list_datasets()), 2)

    def test_exception_cancels_unpublished_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.auspex"
            with self.assertRaisesRegex(RuntimeError, "measurement failed"):
                with ManualPipeline(SCHEMA, NODES, EDGES,
                                    writers={"disk": WriterSpec("pass", "out")}, store_path=path) as pipe:
                    pipe.submit(DATA, axes=AXES)
                    pipe.end_scan()
                    pipe.submit(DATA, axes=AXES)
                    raise RuntimeError("measurement failed")
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [("data", "sweep_000")])

    def test_full_buffer_completion_timeout_cleans_up(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BufferFullWarning)
            with self.assertRaises(PipelineError):
                with ManualPipeline(SCHEMA, NODES, EDGES, timeout=0.2,
                                    buffers={"memory": BufferSpec("pass", "out", 1)}) as pipe:
                    pipe.submit(DATA, axes=AXES)
                    pipe.submit(DATA, axes=AXES)
            self.assertTrue(pipe._closed)
            self.assertFalse(pipe._graph.thread.is_alive())

    def test_cancel_wakes_waiting_buffer_consumer(self):
        pipe = ManualPipeline(SCHEMA, NODES, EDGES,
                              buffers={"memory": BufferSpec("pass", "out")})
        failures = []
        def consume():
            try:
                pipe.buffers["memory"].receive(timeout=None)
            except PipelineError as exc:
                failures.append(str(exc))
        thread = threading.Thread(target=consume)
        thread.start()
        try:
            pipe.cancel()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertTrue(failures)
        finally:
            pipe.cancel()
            thread.join(5)

    def test_incomplete_group_fails_normal_exit(self):
        """Definition-backed context exit cannot publish an unfinished required average."""
        from auspex_v2.filters.definitions import (
            ComplexAverage, DataBuffer, PipelineDefinition, PipelineInput,
        )

        source = PipelineInput("trace", SCHEMA)
        average = ComplexAverage(name="average", repeats=2)
        memory = DataBuffer(name="memory")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, average.sink), (average.final_average, memory.sink)])
        with self.assertRaisesRegex(PipelineError, "incomplete"):
            with ManualPipeline(definition) as pipe:
                pipe.submit(DATA, axes=AXES)
