"""Run-folder and volatile buffer ownership, without starting ExperimentServer."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from auspex_v2.data import AuspexStore
from auspex_v2.experiment import Experiment, OutputConnector
from auspex_v2.experiment.resources import RunFolder, RetainedRunBuffers, preflight_run
from auspex_v2.experiment import resources
from auspex_v2.filters import Batch, PipelineError
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, DataBuffer, WriteToFile


class Buffered(Experiment):
    """Minimal saved unit with server-retained temporary results."""
    output = OutputConnector(dtype="float64", max_bytes=1024)

    def setup_pipeline(self):
        source = PipelineInput("input")
        memory = DataBuffer(name="memory", capacity=2)
        PipelineDefinition(inputs=[source], edges=[(source, memory.sink)])
        self.connect(self.output, source)

    def sequence(self, seq):
        seq.publish(self.output, data=1.0)


class RunResourceTest(unittest.TestCase):
    """Test preallocated directories and bounded terminal buffer retention."""
    def test_browser_compatibility_import_remains_available(self):
        """Browser compatibility is independent of the removed execution-test shims."""
        from auspex_v2.browser import inspect_folder
        from auspex_v2.data.browser import inspect_folder as current_browser
        self.assertIs(inspect_folder, current_browser)

    def test_preflight_is_read_only_and_rejects_unsupported_budgets(self):
        """Admission can reject deterministic recipe errors before claiming a folder."""
        plan = Buffered().compile()
        with tempfile.TemporaryDirectory() as directory:
            with patch("auspex_v2.filters.server.FilterServer.__init__", side_effect=AssertionError("spawned")):
                preflight_run(plan)
                with self.assertRaisesRegex(PipelineError, "memory budget"):
                    preflight_run(plan, memory_budget_bytes=1)
                with self.assertRaisesRegex(PipelineError, "positive integers"):
                    preflight_run(plan, slots=0)
            self.assertEqual(list(Path(directory).iterdir()), [])
        class TooLarge(Buffered):
            output = OutputConnector(dtype="float64", max_bytes=8 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(PipelineError, "inline transfer limit"):
            preflight_run(TooLarge().compile())
        class WithWriter(Buffered):
            def setup_pipeline(self):
                source = PipelineInput("input")
                disk = WriteToFile(name="disk", max_scan_bytes=1024)
                PipelineDefinition(inputs=[source], edges=[(source, disk.sink)])
                self.connect(self.output, source)
        with self.assertRaisesRegex(PipelineError, "Writer staging exceeds"):
            preflight_run(WithWriter().compile(), store_scan_budget_bytes=100)
        doc = plan.document
        doc["pipelines"].append(doc["pipelines"][0])
        from auspex_v2.experiment import CompiledPlan
        combined = preflight_run(CompiledPlan(doc))
        self.assertEqual(set(combined.runtime()[4]), {"pipeline0.memory", "pipeline1.memory"})
        with self.assertRaisesRegex(PipelineError, "memory budget"):
            preflight_run(CompiledPlan(doc), memory_budget_bytes=10000)

    def test_default_writer_limits_preserve_four_writer_admission(self):
        """Authoring/runtime/preflight defaults agree without allocating scan capacity."""
        class FourWriters(Experiment):
            """Four default-sized writers exactly fill the default aggregate allowance."""
            output = OutputConnector(dtype="float64")
            writer_count = 4

            def setup_pipeline(self):
                source = PipelineInput("input")
                writers = [WriteToFile(name=f"disk_{i}", dataset=f"value_{i}")
                           for i in range(self.writer_count)]
                PipelineDefinition(inputs=[source], edges=[(source, w.sink) for w in writers])
                self.connect(self.output, source)

            def sequence(self, seq):
                seq.publish(self.output, data=1.0)

        compiled = preflight_run(FourWriters().compile())
        schema, _, _, writers, _, _ = compiled.runtime()
        self.assertEqual(schema.max_bytes, 64 * 1024)  # Transport defaults are unchanged.
        self.assertEqual(len(writers), 4)
        self.assertEqual(sum(w.max_scan_bytes for w in writers.values()), 2 * 1024**3)
        self.assertTrue(all(w.max_scan_bytes == 512 * 1024**2
                            and w.max_scan_windows == 16384 for w in writers.values()))

        class FiveWriters(FourWriters):
            """A fifth default writer must still be rejected before execution."""
            writer_count = 5

        with self.assertRaisesRegex(PipelineError, "Writer staging exceeds scan budget"):
            preflight_run(FiveWriters().compile())

    def test_folder_allocation_metadata_and_numbering(self):
        """Every accepted run owns a distinct initialized store, even without writers."""
        plan = Buffered().compile()
        with tempfile.TemporaryDirectory() as directory:
            first = RunFolder.allocate(plan, name="test", directory=directory)
            self.assertEqual(first.path.name, "test_000.auspex")
            self.assertEqual(first.status()["state"], "ACCEPTED")
            self.assertEqual(first.status()["plan_identity"], plan.identity)
            with AuspexStore(first.path) as store:
                self.assertEqual(store.read_run_metadata()["plan_identity"], plan.identity)
                self.assertEqual(store.list_datasets(), [])
            self.assertEqual((first.path / "plan.json").exists(), True)
            (Path(directory) / "test_008.auspex").mkdir()
            second = RunFolder.allocate(plan, name="test", directory=directory)
            self.assertEqual(second.path.name, "test_009.auspex")
            self.assertEqual(first.status()["state"], "ACCEPTED")
            with self.assertRaises(ValueError):
                RunFolder.allocate(plan, name="../bad", directory=directory)

    def test_initialization_failure_keeps_failed_folder_and_next_number(self):
        """A claimed number and failure stub survive a post-creation write error."""
        plan = Buffered().compile()
        original = resources._atomic_json
        def fail_plan(path, value):
            if Path(path).name == "plan.json":
                raise OSError("disk stopped")
            return original(path, value)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(resources, "_atomic_json", fail_plan):
                with self.assertRaisesRegex(OSError, "disk stopped"):
                    RunFolder.allocate(plan, name="test", directory=directory)
            failed = RunFolder(Path(directory) / "test_000.auspex")
            self.assertEqual(failed.status()["state"], "FAILED")
            self.assertFalse((failed.path / "plan.json").exists())
            next_run = RunFolder.allocate(plan, name="test", directory=directory)
            self.assertEqual(next_run.path.name, "test_001.auspex")

    def test_buffer_reservation_transfer_release_cleanup(self):
        """Results survive caller array mutations but never a server catalog cleanup."""
        plan = Buffered().compile()
        registry = RetainedRunBuffers(2048)
        self.assertEqual(registry.reserve("one", plan), 2048)
        with self.assertRaisesRegex(PipelineError, "Server-wide"):
            registry.reserve("two", plan)
        with self.assertRaises(RuntimeError):
            registry.release_buffers("one")
        data = np.array(2.0)
        batch = Batch(None, 0, 0, 0, data, ())
        registry.finish("one", {"memory": [batch]})
        data[...] = 7.0
        self.assertEqual(registry.view("one", "memory")[0].data.item(), 2.0)
        registry.view("one", "memory")[0].data[...] = 8.0
        self.assertEqual(registry.view("one", "memory")[0].data.item(), 2.0)
        registry.release_buffers("one")
        registry.release_buffers("one")
        self.assertEqual(registry.reserved_bytes, 0)
        with self.assertRaises(KeyError):
            registry.view("one", "memory")
        registry.reserve("two", plan)
        registry.finish("two", {})
        registry.cleanup("one")
        with self.assertRaises(KeyError):
            registry.release_buffers("one")
        registry.cleanup("two")
        self.assertEqual(registry.reserved_bytes, 0)

    def test_over_capacity_transfer_does_not_mark_run_terminal(self):
        """Reject missing reservations and excess batches without adopting partial data."""
        registry = RetainedRunBuffers(2048)
        registry.reserve("one", Buffered().compile())
        batch = Batch(None, 0, 0, 0, np.array(1.0), ())
        with self.assertRaisesRegex(PipelineError, "capacity exceeded"):
            registry.finish("one", {"memory": [batch] * 3})
        with self.assertRaises(RuntimeError):
            registry.view("one", "memory")
        registry.finish("one", {"memory": [batch, batch]})
        self.assertEqual(len(registry.view("one", "memory")), 2)
