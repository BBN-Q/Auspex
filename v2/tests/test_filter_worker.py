"""Spawned worker and bounded shared-memory edge smoke tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from auspex_v2.data import Axis, AuspexStore
from auspex_v2.filters import Batch, FilterWorker, PipelineError, PortSchema, SharedEdge
from auspex_v2.instruments.drivers.simulated import SimulatedVNA


SCHEMA = PortSchema("<c8", 1, ("frequency",), ("Hz",), 64_000)


class WorkerTest(unittest.TestCase):
    """Exercise actual spawn boundaries without pickling filter instances or arrays."""

    def test_full_required_edge_fails_clearly_and_reuses_released_slot(self):
        """A full edge cannot overwrite unread data or silently discard it."""
        batch = Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                      (Axis("frequency", [1.], "Hz"),))
        with SharedEdge(slots=1, slot_bytes=128) as edge:
            edge.send_batch(batch)
            with self.assertRaisesRegex(PipelineError, "free edge slot"):
                edge.send_batch(batch, timeout=0.01)
            kind, received, slot = edge.receive()
            self.assertEqual(kind, "batch")
            self.assertEqual(received.data[0], 1j)
            edge.release(slot)
            edge.send_batch(batch)
            _, _, slot = edge.receive()
            edge.release(slot)

    def test_averaging_across_two_windows(self):
        """Confirm ordered preview and final messages across a real process boundary."""
        with FilterWorker("complex_average", SCHEMA, config={"repeats": 2}, slot_bytes=8_000) as worker:
            for index in range(2):
                worker.push(Batch(None, index, 0, 0,
                        np.array([complex(index + 1)], dtype=np.complex64),
                        (Axis("frequency", np.array([6.1e9]), "Hz"),)))
                worker.end_window(index)
                event, batch = worker.receive("partial")
                self.assertEqual(event, "batch")
                self.assertEqual(batch.group, "average-0")
                self.assertEqual(batch.count, index + 1)
                self.assertEqual(batch.data[0], (index + 2) / 2)
                if index == 1:
                    event, result = worker.receive("final")
                    self.assertEqual(event, "batch")
                    self.assertEqual(result.data[0], 1.5)
                    self.assertEqual(worker.receive("final")[0], "group_end")
                self.assertEqual(worker.receive("final")[0], "window_end")
            worker.finish()
            self.assertEqual(worker.receive("final")[0], "run_end")
            worker.check()

    def test_five_sweep_vna_result_from_spawned_filter(self):
        """Capture every simulated sweep and its final mean without mixing previews."""
        with tempfile.TemporaryDirectory() as directory:
            vna = SimulatedVNA("SIM::VNA")
            vna.connect()
            with AuspexStore(Path(directory) / "vna.auspex", mode="w") as store, \
                 FilterWorker("complex_average", SCHEMA, config={"repeats": 5}) as worker:
                for index in range(5):
                    frequencies, trace = vna.get_trace()
                    axes = (Axis("frequency", frequencies, "Hz"),)
                    store.write_dataset("sweeps", f"trace_{index}", trace, axes)
                    worker.push(Batch(None, index, 0, 0, trace, axes))
                    worker.end_window(index)
                    event, partial = worker.receive("partial")
                    self.assertEqual(event, "batch")
                    self.assertEqual((partial.count, partial.expected_count), (index + 1, 5))
                    if index == 4:
                        event, final = worker.receive("final")
                        self.assertEqual(event, "batch")
                        np.testing.assert_array_equal(final.data, trace)
                        np.testing.assert_array_equal(final.axes[0].points, frequencies)
                        store.write_dataset("results", "average", final.data, final.axes)
                        self.assertEqual(worker.receive("final")[0], "group_end")
                    self.assertEqual(worker.receive("final")[0], "window_end")
                worker.finish()
                self.assertEqual(worker.receive("final")[0], "run_end")
            vna.disconnect()
            with AuspexStore(Path(directory) / "vna.auspex") as store:
                self.assertEqual(len(store.list_datasets()), 6)
                np.testing.assert_array_equal(store.read_dataset("results", "average").data, trace)

    def test_undrained_preview_does_not_block_required_final(self):
        """An absent plot client cannot block an averaging group or its result."""
        with FilterWorker("complex_average", SCHEMA, config={"repeats": 5}, slots=1) as worker:
            for index in range(5):
                worker.push(Batch(None, index, 0, 0,
                    np.array([index + 1j], dtype=np.complex64),
                    (Axis("frequency", [1.], "Hz"),)))
                worker.end_window(index)
                if index == 4:
                    event, result = worker.receive("final")
                    self.assertEqual(event, "batch")
                    self.assertEqual(result.data[0], 2 + 1j)
                    self.assertEqual(worker.receive("final")[0], "group_end")
                self.assertEqual(worker.receive("final")[0], "window_end")
                # Deliberately do not drain the preview edge.
            _, preview = worker.receive("partial")
            self.assertEqual(preview.count, 1)
            worker.check()

    def test_process_rejects_premature_completion(self):
        """Even a caller bypassing the parent API cannot fabricate WindowEnd."""
        with FilterWorker("passthrough", SCHEMA) as worker:
            worker.input.send_event("window_end", "g", 0)
            with self.assertRaisesRegex(PipelineError, "Wrong window completion"):
                worker.receive("out")

    def test_passthrough_retains_input_until_output_copy(self):
        with FilterWorker("passthrough", SCHEMA, slots=1) as worker:
            for sequence, value in enumerate((1j, 2j, 3j)):
                worker.push(Batch(None, 0, sequence, 0,
                    np.array([value], dtype=np.complex64), (Axis("frequency", [1.], "Hz"),)))
                event, result = worker.receive("out")
                self.assertEqual(event, "batch")
                self.assertEqual(result.data[0], value)

    def test_killed_worker_fails_instead_of_faking_completion(self):
        with FilterWorker("passthrough", SCHEMA) as worker:
            worker.process.kill()
            worker.process.join(timeout=5)
            with self.assertRaises(PipelineError):
                worker.receive("out", timeout=1)

    def test_revision_change_after_completed_group(self):
        """Reject mid-group changes but acknowledge a new revision afterwards."""
        with FilterWorker("complex_average", SCHEMA, config={"repeats": 1}) as worker:
            worker.push(Batch(None, 0, 0, 0,
                np.array([1j], dtype=np.complex64), (Axis("frequency", [1.], "Hz"),)))
            worker.end_window(0)
            with self.assertRaisesRegex(PipelineError, "Wait for required group completion"):
                worker.configure({"repeats": 2})
            self.assertEqual(worker.receive("partial")[0], "batch")
            self.assertEqual(worker.receive("final")[0], "batch")
            self.assertEqual(worker.receive("final")[0], "group_end")
            with self.assertRaisesRegex(PipelineError, "Wait for required group completion"):
                worker.configure({"repeats": 2})
            self.assertEqual(worker.receive("final")[0], "window_end")
            self.assertEqual(worker.configure({"repeats": 2}), 1)

    def test_incomplete_group_fails_on_run_end(self):
        """The runner ends the run; it cannot declare an unfinished average complete."""
        with FilterWorker("complex_average", SCHEMA, config={"repeats": 2}) as worker:
            worker.push(Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                              (Axis("frequency", [1.], "Hz"),)))
            worker.end_window(0)
            self.assertEqual(worker.receive("partial")[0], "batch")
            self.assertEqual(worker.receive("final")[0], "window_end")
            worker.finish()
            with self.assertRaisesRegex(PipelineError, "incomplete"):
                worker.receive("final")

    def test_mismatch_reports_error_without_result(self):
        """Report a coordinate mismatch through the worker status channel."""
        with FilterWorker("complex_average", SCHEMA, config={"repeats": 2}) as worker:
            for index in range(2):
                worker.push(Batch(None, index, 0, 0,
                        np.array([1j], dtype=np.complex64),
                        (Axis("frequency", [6.1e9 + index], "Hz"),)))
                if index == 0:
                    worker.end_window(0)
                    self.assertEqual(worker.receive("partial")[0], "batch")
                    self.assertEqual(worker.receive("final")[0], "window_end")
            self.assertTrue(worker.status_parent.poll(5))
            with self.assertRaisesRegex(PipelineError, "grids differ"):
                worker.check()
