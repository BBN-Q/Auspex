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
from tests.experiment.fake_executor import FakeExecutor
from tests.experiment.pipeline_executor import PipelineExecutor
from auspex_v2.filters import PipelineError, PortSchema
from auspex_v2.filters.definitions import (
    PipelineInput, PipelineDefinition, Passthrough, DataBuffer, WriteToFile,
    EmptySource,
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
    def test_legacy_graph_is_not_silently_saved_as_executable(self):
        """Legacy-looking graph configuration must not imply a running graph."""
        experiment = configured()
        with self.assertRaisesRegex(CompileError, "PipelineDefinition"):
            experiment.set_graph({"writer": "disk"})
        with self.assertRaisesRegex(CompileError, "OutputConnector"):
            experiment.outputs = {"trace": {"dtype": "float64"}}
        with self.assertRaisesRegex(CompileError, "OutputConnector"):
            experiment.graph = {"writer": "disk"}

    def test_pipeline_required_and_none_setup_ignored(self):
        with self.assertRaisesRegex(CompileError, "requires an attached pipeline"):
            Experiment().compile()
        class External(Measurement):
            setup_pipeline = None
        experiment = configured(External)
        experiment.connect(experiment.trace, buffer_input())
        plan = experiment.compile()
        self.assertEqual(plan.document["version"], 6)
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
        with patch("tests.experiment.pipeline_executor.ManualPipeline") as runtime:
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
            lambda d: next(node for node in d["main"] if node["op"] == "PUBLISH").update(output="missing"),
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
            code = f'''import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
import numpy as np
from auspex_v2.experiment import CompiledPlan
from tests.experiment.pipeline_executor import PipelineExecutor

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

    def test_unattended_buffer_overflow_fails_without_a_consumer(self):
        """A buffer plus a disk writer must fail clearly, not hang on scan closure."""
        class Small(Measurement):
            def setup_pipeline(self):
                source = PipelineInput("input")
                buffer = DataBuffer(name="memory", capacity=1)
                disk = WriteToFile(name="disk")
                PipelineDefinition(inputs=[source], edges=[
                    (source, buffer.sink), (source, disk.sink)])
                self.connect(self.trace, source)
            def sequence(self, seq):
                super().sequence(seq)
                super().sequence(seq)
        with tempfile.TemporaryDirectory() as directory:
            run = PipelineExecutor(command).run(configured(Small).compile(),
                                                store_path=Path(directory) / "run.auspex", timeout=1)
        self.assertEqual(run.outcome, "FAILED")
        self.assertIn("Buffer 'memory' capacity exceeded", run.reason)
        self.assertNotIn("Timed out", run.reason)

    def test_symbolic_tuple_results_cannot_be_implicitly_iterated(self):
        class Unpack(Measurement):
            def sequence(self, seq):
                frequency, response = seq.call(self.vna, "get_trace")
        with self.assertRaisesRegex(CompileError, "unpacked"):
            configured(Unpack).compile()
