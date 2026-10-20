"""Compile whole execution units, then run them without their authoring classes."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from auspex_v2.data import AuspexStore
from auspex_v2.experiment import (
    Experiment, OutputConnector, InstrumentRef, FloatParameter, IntParameter,
    CompileError, CompiledPlan, PipelineOverrideWarning,
)
from auspex_v2.experiment.fake import FakeExecutor
from auspex_v2.experiment.pipeline import PipelineExecutor
from auspex_v2.filters import PipelineError, PortSchema, ManualPipeline, FilterServer
from auspex_v2.filters.definitions import (
    PipelineInput, PipelineDefinition, Passthrough, DataBuffer, WriteToFile,
    EmptySource, ComplexAverage,
)


class Measurement(Experiment):
    """A driver-return tuple is explicitly split into coordinates and values."""
    vna = InstrumentRef(kind="VNA")
    trace = OutputConnector(dtype="complex128", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"}, max_bytes=4096)
    bias = FloatParameter(default=0., unit="V")

    def setup_pipeline(self):
        source = PipelineInput("trace_input")
        forward = Passthrough(name="forward")
        memory = DataBuffer(name="memory", capacity=8)
        disk = WriteToFile(name="disk", group="measurement", dataset="response")
        PipelineDefinition(inputs=[source], edges=[
            (source, forward.sink), (forward.source, memory.sink), (forward.source, disk.sink)])
        self.connect(self.trace, source)

    def sequence(self, seq):
        result = seq.call(self.vna, "get_trace")
        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

    def teardown(self, seq):
        seq.call(self.vna, "safe")


def configured(cls=Measurement):
    experiment = cls()
    experiment.bind_instrument("vna", "lab_vna")
    return experiment


def command(op, instrument, member, args, kwargs):
    if member == "get_trace":
        return np.array([1., 2.]), np.array([2 + 3j, 4 + 5j])


def buffer_input(name="input", schema=None):
    source = PipelineInput(name, schema)
    memory = DataBuffer(name="memory", capacity=8)
    PipelineDefinition(inputs=[source], edges=[(source, memory.sink)])
    return source


class PublicationCompilerTest(unittest.TestCase):
    """Check syntactic wiring, detached setup and plain saved-plan validation."""
    def test_pipeline_required_and_none_setup_ignored(self):
        with self.assertRaisesRegex(CompileError, "requires an attached pipeline"):
            Experiment().compile()
        class External(Measurement):
            setup_pipeline = None
        experiment = configured(External)
        experiment.connect(experiment.trace, buffer_input())
        plan = experiment.compile()
        self.assertEqual(plan.document["version"], 2)
        self.assertEqual(plan.document["pipelines"][0]["connections"], {"input": "trace"})

    def test_internal_setup_uses_config_and_runs_only_on_detached_copy(self):
        """Recompilation sees current configuration without mutating older plans."""
        class Internal(Measurement):
            capacity = IntParameter(default=4)
            def setup_pipeline(self):
                self.constructed = True
                source = PipelineInput("input")
                memory = DataBuffer(name="memory", capacity=self.capacity.value)
                PipelineDefinition(inputs=[source], edges=[(source, memory.sink)])
                self.connect(self.trace, source)
        experiment = configured(Internal)
        old = experiment.compile()
        self.assertFalse(hasattr(experiment, "constructed"))
        self.assertEqual(experiment._connections, [])
        experiment.capacity.value = 7
        new = experiment.compile()
        self.assertEqual(old.document["pipelines"][0]["pipeline"]["definition"]["nodes"][0]["config"]["capacity"], 4)
        self.assertEqual(new.document["pipelines"][0]["pipeline"]["definition"]["nodes"][0]["config"]["capacity"], 7)

    def test_external_wiring_skips_internal_setup_and_warns(self):
        class Internal(Measurement):
            def setup_pipeline(self):
                raise AssertionError("Internal setup must not run")
        experiment = configured(Internal)
        experiment.connect(experiment.trace, buffer_input())
        with self.assertWarnsRegex(PipelineOverrideWarning, "replace"):
            plan = experiment.compile()
        self.assertEqual(len(plan.document["pipelines"]), 1)

    def test_all_inputs_require_sources_and_empty_is_explicit(self):
        """A schema is not a source; empty bindings survive but do not yet execute."""
        class External(Measurement):
            setup_pipeline = None
        experiment = configured(External)
        a = PipelineInput("a")
        b = PipelineInput("b", PortSchema("float64", 0, (), (), 1024))
        x, y = DataBuffer(name="x"), DataBuffer(name="y")
        PipelineDefinition(inputs=[a, b], edges=[(a, x.sink), (b, y.sink)])
        experiment.connect(experiment.trace, a)
        with self.assertRaisesRegex(CompileError, "Every declared"):
            experiment.compile()
        experiment.connect(EmptySource(), b)
        plan = experiment.compile()
        self.assertEqual(plan.document["pipelines"][0]["connections"], {"a": "trace", "b": None})
        with patch("auspex_v2.experiment.pipeline.ManualPipeline") as runtime:
            with self.assertRaisesRegex(PipelineError, "API-only"):
                PipelineExecutor(command).run(plan)
            runtime.assert_not_called()

    def test_unconnected_publication_rejected_even_in_false_branch(self):
        class Unconnected(Measurement):
            other = OutputConnector(dtype="float64")
            def sequence(self, seq):
                with seq.if_(False):
                    seq.publish(self.other, data=1.)
        with self.assertRaisesRegex(CompileError, "no pipeline connection"):
            configured(Unconnected).compile()

    def test_bad_axes_ownership_and_unavailable_values(self):
        """Publication uses owned outputs and definitely available symbolic results."""
        foreign = configured().trace
        for variant in ("axes", "foreign", "scope"):
            class Invalid(Measurement):
                def sequence(self, seq):
                    if variant == "axes":
                        seq.publish(self.trace, data=[], axes={})
                    elif variant == "foreign":
                        seq.publish(foreign, data=[], axes={"frequency": []})
                    else:
                        with seq.if_(False):
                            result = seq.call(self.vna, "get_trace")
                        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})
            with self.subTest(variant=variant), self.assertRaises(CompileError):
                configured(Invalid).compile()

    def test_publish_disallowed_in_setup_and_teardown(self):
        for phase in ("setup", "teardown"):
            class Invalid(Measurement):
                pass
            def publish(self, seq):
                seq.publish(self.trace, data=[1.], axes={"frequency": [1.]})
            setattr(Invalid, phase, publish)
            with self.subTest(phase=phase), self.assertRaisesRegex(CompileError, "main scan"):
                configured(Invalid).compile()

    def test_connector_and_wiring_mutations_rejected_during_recording(self):
        class Mutation(Measurement):
            def sequence(self, seq):
                self.trace.unit = "new unit"
        with self.assertRaisesRegex(CompileError, "frozen"):
            configured(Mutation).compile()
        experiment = configured()
        with self.assertRaises(CompileError):
            experiment.trace = 1
        with self.assertRaises(CompileError):
            del experiment.trace

    def test_loaded_plan_revalidates_connections_and_publications(self):
        """Plain JSON cannot bypass the required connection/schema checks."""
        plan = configured().compile()
        for change in (
            lambda d: d.update(pipelines=[]),
            lambda d: d["pipelines"][0]["connections"].update(trace_input="missing"),
            lambda d: d["outputs"]["trace"].update(dtype="float64"),
            lambda d: d["main"][2].update(output="missing"),
        ):
            document = plan.document
            change(document)
            with self.assertRaises(CompileError):
                CompiledPlan(document)

    def test_saved_definition_is_independent_of_later_edits(self):
        class External(Measurement):
            setup_pipeline = None
        experiment = configured(External)
        source = buffer_input()
        experiment.connect(experiment.trace, source)
        plan = experiment.compile()
        source._pipeline.edges[0][1].owner.capacity = 5
        experiment.trace.unit = "V"
        self.assertIsNone(plan.document["outputs"]["trace"]["value_unit"])
        self.assertEqual(plan.document["pipelines"][0]["pipeline"]["definition"]["nodes"][0]["config"]["capacity"], 8)

    def test_fake_publications_validate_and_preserve_coordinates(self):
        """One explicit publication is one copied window, with requested sweep coordinates."""
        experiment = configured()
        experiment.add_sweep(experiment.bias, [0., 1.])
        run = FakeExecutor(command).run(experiment.compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual([w["coordinates"] for w in run.windows], [{"bias": 0.}, {"bias": 1.}])
        np.testing.assert_array_equal(run.windows[0]["data"], [2 + 3j, 4 + 5j])
        self.assertNotIn("2.+3.j", "\n".join(run.trace))
        self.assertIn("PUBLISH trace", "\n".join(run.trace))


class PublicationExecutionTest(unittest.TestCase):
    """A saved execution unit feeds real filter workers with injected offline results."""
    def test_fresh_process_roundtrip_and_dataset(self):
        """No experiment class, notebook or driver reference is needed after saving."""
        with tempfile.TemporaryDirectory() as directory:
            plan_path, store_path = Path(directory) / "plan.json", Path(directory) / "run.auspex"
            experiment = configured()
            experiment.add_sweep(experiment.bias, [0., 1.])
            plan = experiment.compile()
            plan.save(plan_path)
            code = f'''import numpy as np
from auspex_v2.experiment import CompiledPlan
from auspex_v2.experiment.pipeline import PipelineExecutor

def command(op, instrument, member, args, kwargs):
    if member == "get_trace":
        return np.array([1., 2.]), np.array([2+3j, 4+5j])

if __name__ == "__main__":
    plan = CompiledPlan.load({str(plan_path)!r})
    assert plan.identity == {plan.identity!r}
    run = PipelineExecutor(command).run(plan, store_path={str(store_path)!r})
    assert run.outcome == "SUCCEEDED", run.reason
    assert len(run.buffers["memory"].drain()) == 2
'''
            script = Path(directory) / "run.py"
            script.write_text(code)
            subprocess.run([sys.executable, "-B", str(script)], check=True, capture_output=True, timeout=30)
            with AuspexStore(store_path) as store:
                self.assertEqual(store.list_datasets(), [("measurement", "response")])
                self.assertEqual(store.read_dataset("measurement", "response").data.shape, (2, 2))

    def test_scan_is_sealed_before_teardown_and_awaited_afterward(self):
        """Storage may progress concurrently with teardown, but never delays its start."""
        events = []
        original_seal, original_wait = ManualPipeline._seal_scan, FilterServer.wait_scan
        def seal(pipeline):
            if pipeline._scan is not None:
                events.append("seal")
            return original_seal(pipeline)
        def wait(graph, scan, **kwargs):
            events.append("wait")
            return original_wait(graph, scan, **kwargs)
        def record(*args):
            if args[2] == "safe":
                events.append("teardown")
            return command(*args)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ManualPipeline, "_seal_scan", seal), patch.object(FilterServer, "wait_scan", wait):
                run = PipelineExecutor(record).run(configured().compile(),
                                                   store_path=Path(directory) / "run.auspex")
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual(events, ["seal", "teardown", "wait"])

    def test_bad_data_fails_and_attempts_teardown_without_publishing(self):
        calls = []
        def bad(op, instrument, member, args, kwargs):
            calls.append(member)
            if member == "get_trace":
                return np.array([1., 2.]), np.array([1., 2.])  # Not complex128.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            run = PipelineExecutor(bad).run(configured().compile(), store_path=path)
            self.assertEqual(run.outcome, "FAILED")
            self.assertEqual(calls, ["get_trace", "safe"])
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [])

    def test_teardown_failure_stops_cleanup_but_retains_complete_scan(self):
        """Publication and teardown outcome are distinct, and cleanup is stop-on-error."""
        class Cleanup(Measurement):
            def teardown(self, seq):
                seq.call(self.vna, "safe")
                seq.call(self.vna, "later")
        calls = []
        def fail(op, instrument, member, args, kwargs):
            calls.append(member)
            if member == "safe":
                raise RuntimeError("safe state not established")
            return command(op, instrument, member, args, kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            run = PipelineExecutor(fail).run(configured(Cleanup).compile(), store_path=path)
            self.assertEqual(run.outcome, "FAILED")
            self.assertIn("teardown", run.reason)
            self.assertNotIn("later", calls)
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [("measurement", "response")])

    def test_teardown_precedes_wait_for_required_group_completion(self):
        """An incomplete required average fails finalization after cleanup was attempted."""
        class Average(Measurement):
            def setup_pipeline(self):
                source = PipelineInput("input")
                avg = ComplexAverage(name="average", repeats=2)
                memory = DataBuffer(name="memory")
                PipelineDefinition(inputs=[source], edges=[
                    (source, avg.sink), (avg.final_average, memory.sink)])
                self.connect(self.trace, source)
        calls = []
        def record(*args):
            calls.append(args[2])
            return command(*args)
        run = PipelineExecutor(record).run(configured(Average).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertIn("safe", calls)
        self.assertIn("pipeline finalization", run.reason)

    def test_symbolic_tuple_results_cannot_be_implicitly_iterated(self):
        class Unpack(Measurement):
            def sequence(self, seq):
                frequency, response = seq.call(self.vna, "get_trace")
        with self.assertRaisesRegex(CompileError, "unpacked"):
            configured(Unpack).compile()
