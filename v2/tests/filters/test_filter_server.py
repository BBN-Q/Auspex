"""Offline multi-process graph tests; no hardware or throughput claims."""

import multiprocessing as mp
import threading
import unittest
from multiprocessing import shared_memory

import numpy as np

from auspex_v2.data import Axis
from auspex_v2.filters import Batch, FilterServer, NodeSpec, PipelineError, PortSchema, ResultSpec
from auspex_v2.filters.graph_transport import GraphEdge
from auspex_v2.instruments.drivers.simulated import SimulatedVNA

SCHEMA = PortSchema("<c8", 1, ("frequency",), ("Hz",), 128)


def batch(window, value=None):
    return Batch(None, window, 0, 0,
                 np.array([window if value is None else value], dtype=np.complex64),
                 (Axis("frequency", [1.], "Hz"),))


def simple(**kwargs):
    return FilterServer(SCHEMA, {"pass": NodeSpec("passthrough", SCHEMA)},
                        [("$input", "out", "pass")],
                        {"raw": ResultSpec("pass", "out")}, **kwargs)


class FilterServerTest(unittest.TestCase):
    """Prove bounded overlap, independent groups, error propagation and ownership."""

    def test_fanout_independent_groups(self):
        nodes = {"pass": NodeSpec("passthrough", SCHEMA),
                 "two": NodeSpec("complex_average", SCHEMA, {"repeats": 2, "group_name": "two"}),
                 "three": NodeSpec("complex_average", SCHEMA, {"repeats": 3, "group_name": "three"})}
        edges = [("$input", "out", "pass"), ("pass", "out", "two"), ("pass", "out", "three")]
        results = {"two": ResultSpec("two", "final"), "three": ResultSpec("three", "final"),
                   "preview": ResultSpec("two", "partial", True)}
        with FilterServer(SCHEMA, nodes, edges, results, result_capacity=4) as graph:
            for window in range(6):
                graph.push(batch(window))
                graph.end_window(window)
            graph.finish()
            graph.wait_finished()
            twos = [graph.receive("two") for _ in range(3)]
            threes = [graph.receive("three") for _ in range(2)]
            self.assertEqual([b.group for b in twos], ["two-0", "two-1", "two-2"])
            self.assertEqual([b.window for b in threes], [2, 5])
            np.testing.assert_allclose([b.data[0] for b in twos], [0.5, 2.5, 4.5])
            np.testing.assert_allclose([b.data[0] for b in threes], [1, 4])
            names = [slot.name for edge in graph.channels for slot in edge.slots]
        self.assertTrue(all(not p.is_alive() for p in graph.processes.values()))
        for name in names:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_overlap_and_bounded_retention(self):
        """A full result buffer holds window 1 pending while window 2 is admitted."""
        with simple(slots=2, result_capacity=1) as graph:
            graph.push(batch(0))
            graph.end_window(0)
            graph.wait_window(0)  # Result 0 occupies the only retained-result slot.
            graph.push(batch(1))
            graph.end_window(1)
            graph.push(batch(2))
            graph.end_window(2)
            self.assertEqual(graph.completed_window, 0)
            for window in range(3):
                self.assertEqual(graph.receive("raw").window, window)
            graph.finish()
            graph.wait_finished()

    def test_nonbackpressure_result_overflow(self):
        with simple(backpressure=False, slots=4, result_capacity=1) as graph:
            graph.push(batch(0))
            graph.end_window(0)
            graph.wait_window(0)
            graph.push(batch(1))
            # Do not consume the retained result. The next result must fail.
            with self.assertRaisesRegex(PipelineError, "Capacity exhausted: result:raw retention"):
                graph.wait_window(1)
            graph.thread.join(5)
            self.assertFalse(graph.thread.is_alive())

    def test_cancel_unblocks_producer(self):
        """Window credit exhaustion cannot prevent cancellation or resource cleanup."""
        with simple(result_capacity=1, max_windows=1) as graph:
            graph.push(batch(0))
            graph.end_window(0)
            graph.wait_window(0)
            graph.push(batch(1))
            graph.end_window(1)
            started = threading.Event()
            errors = []

            def submit():
                started.set()
                try:
                    graph.push(batch(2))
                except PipelineError as exc:
                    errors.append(str(exc))

            producer = threading.Thread(target=submit)
            producer.start()
            self.assertTrue(started.wait(2))
            self.assertTrue(producer.is_alive())
            graph.cancel()
            producer.join(5)
            self.assertFalse(producer.is_alive())
            self.assertTrue(errors)

    def test_worker_death_wakes_waiters(self):
        with simple() as graph:
            graph.processes["pass"].kill()
            with self.assertRaises(PipelineError):
                graph.receive("raw")
            graph.thread.join(5)
            self.assertFalse(graph.thread.is_alive())

    def test_local_configuration_and_incomplete_run(self):
        nodes = {"avg": NodeSpec("complex_average", SCHEMA, {"repeats": 2})}
        with FilterServer(SCHEMA, nodes, [("$input", "out", "avg")],
                          {"final": ResultSpec("avg", "final")}) as graph:
            graph.push(batch(0))
            graph.end_window(0)
            with self.assertRaisesRegex(PipelineError, "between groups"):
                graph.configure("avg", {"repeats": 3})
            graph.push(batch(1))
            graph.end_window(1)
            self.assertEqual(graph.configure("avg", {"repeats": 3}), 1)
            self.assertEqual(graph.receive("final").revision, 0)
            # Acquisition revision remains independent of this filter's revision.
            for window in range(2, 5):
                graph.push(batch(window))
                graph.end_window(window)
            self.assertEqual(graph.receive("final").revision, 1)
            graph.push(batch(5))
            graph.end_window(5)
            graph.finish()
            with self.assertRaisesRegex(PipelineError, "incomplete"):
                graph.wait_finished()

    def test_internal_capacity_policy(self):
        """Required edges bound both array slots and boundary notifications."""
        context = mp.get_context("spawn")
        stop = context.Event()
        with GraphEdge(context=context, stop=stop, backpressure=False, label="a.out->b",
                       slots=1, slot_bytes=128) as edge:
            edge.send_batch(batch(0))
            with self.assertRaisesRegex(PipelineError, "a.out->b data slots"):
                edge.send_batch(batch(1))
            edge.send_event("window_end", None, 0)
            edge.send_event("window_end", None, 1)
            with self.assertRaisesRegex(PipelineError, "a.out->b notifications"):
                edge.send_event("window_end", None, 2)

    def test_worker_death_unblocks_producer(self):
        """Failure notification and cleanup remain live during admission backpressure."""
        with simple(result_capacity=1, max_windows=1) as graph:
            graph.push(batch(0))
            graph.end_window(0)
            graph.wait_window(0)
            graph.push(batch(1))
            graph.end_window(1)
            started = threading.Event()
            errors = []

            def submit():
                started.set()
                try:
                    graph.push(batch(2))
                except PipelineError as exc:
                    errors.append(str(exc))

            producer = threading.Thread(target=submit)
            producer.start()
            self.assertTrue(started.wait(2))
            graph.processes["pass"].kill()
            self.assertTrue(graph.failed.wait(5))
            producer.join(5)
            self.assertFalse(producer.is_alive())
            self.assertTrue(errors)

    def test_update_does_not_revise_another_branch(self):
        nodes = {"a": NodeSpec("complex_average", SCHEMA, {"repeats": 2}),
                 "b": NodeSpec("complex_average", SCHEMA, {"repeats": 3})}
        with FilterServer(SCHEMA, nodes, [("$input", "out", "a"), ("$input", "out", "b")],
                          {"a": ResultSpec("a", "final"), "b": ResultSpec("b", "final")}) as graph:
            for window in range(2):
                graph.push(batch(window))
                graph.end_window(window)
            graph.configure("a", {"repeats": 1})
            graph.push(batch(2))
            graph.end_window(2)
            graph.finish()
            graph.wait_finished()
            self.assertEqual(graph.receive("a").revision, 0)
            self.assertEqual(graph.receive("a").revision, 1)
            other = graph.receive("b")
            self.assertEqual(other.revision, 0)
            self.assertEqual(other.data[0], 1)

    def test_simulated_vna_smoke(self):
        """Five sweeps pass through two spawned processors with an undrained preview."""
        schema = PortSchema("<c8", 1, ("frequency",), ("Hz",), 64_000)
        nodes = {"pass": NodeSpec("passthrough", schema),
                 "avg": NodeSpec("complex_average", schema, {"repeats": 5})}
        vna = SimulatedVNA("SIM::VNA")
        vna.connect()
        try:
            with FilterServer(schema, nodes,
                    [("$input", "out", "pass"), ("pass", "out", "avg")],
                    {"mean": ResultSpec("avg", "final"),
                     "plot": ResultSpec("avg", "partial", True)}, result_capacity=1) as graph:
                for window in range(5):
                    frequencies, trace = vna.get_trace()
                    expected = trace.copy()
                    graph.push(Batch(None, window, 0, 0, trace,
                                     (Axis("frequency", frequencies, "Hz"),)))
                    trace[:] = 0  # Submission owns its copy before returning.
                    graph.end_window(window)
                graph.finish()
                graph.wait_finished()
                np.testing.assert_array_equal(graph.receive("mean").data, expected)
        finally:
            vna.disconnect()

    def test_invalid_graph_and_startup_failure(self):
        with self.assertRaisesRegex(PipelineError, "memory budget"):
            simple(memory_budget_bytes=1)
        with self.assertRaisesRegex(PipelineError, "exactly one input"):
            FilterServer(SCHEMA, {"pass": NodeSpec("passthrough", SCHEMA)}, [],
                         {"raw": ResultSpec("pass", "out")})
        with self.assertRaisesRegex(PipelineError, "positive repeat count"):
            FilterServer(SCHEMA, {"avg": NodeSpec("complex_average", SCHEMA, {"repeats": 0})},
                         [("$input", "out", "avg")], {"final": ResultSpec("avg", "final")})
