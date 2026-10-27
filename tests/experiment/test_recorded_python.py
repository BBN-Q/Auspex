"""Real numerical calls and JSON round trips, without hardware or mocked numerical libraries."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

from auspex_v2.experiment import (
    CompileError, CompiledPlan, FloatParameter, InstrumentRef, IntParameter,
    RecordedPythonWarning, RuntimePythonWarning,
)
from auspex_v2.experiment.resources import preflight_run
from .experiment_test_support import Experiment
from tests.experiment.fake_executor import FakeExecutor
MODULE = "tests.experiment.computation_functions"
# unittest discovery also imports this suite as experiment.*. Use the exact
# recorded module path so state probes observe the same fixture module.
functions = importlib.import_module(MODULE)


def compile_quiet(experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordedPythonWarning)
        warnings.simplefilter("ignore", RuntimePythonWarning)
        return experiment.compile()


class RecordedPythonTest(unittest.TestCase):
    """Verify ordered execution, detached data, narrow expressions and trusted dependencies."""

    def test_whole_plan_warning_survives_review_markers_and_saved_round_trips(self):
        """Warn on every compile for nested COMPUTE in any phase, even with only reviewed functions."""
        message = "This compiled experiment executes runtime Python code. Auspex cannot guarantee repeatability"
        def recorded(self, seq):
            with seq.if_(False):
                seq.module(MODULE).ordered_keys({})  # Reviewed, but still runtime Python.
        for phase in ("setup", "sequence", "teardown"):
            with self.subTest(phase=phase):
                class Analysis(Experiment):
                    """Install a reviewed computation in one recording phase."""
                setattr(Analysis, phase, recorded)
                experiment = Analysis()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    plan = experiment.compile()
                    experiment.compile()
                self.assertEqual([(w.category, str(w.message)) for w in caught],
                                 [(RuntimePythonWarning, message)] * 2)
                self.assertEqual(plan.document["warnings"], [message])
                self.assertEqual(plan.explain().splitlines()[1], "WARNING: " + message)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "plan.json"
                    plan.save(path)
                    self.assertEqual(json.loads(path.read_text())["warnings"], [message])
                    loaded = CompiledPlan.load(path)
                self.assertEqual(loaded.identity, plan.identity)
                self.assertIn(message, loaded.explain())
                document = plan.document
                document.pop("warnings")
                self.assertEqual(CompiledPlan(document).document["warnings"], [message])

    def test_no_whole_plan_warning_without_compute_instructions(self):
        """Importing/looking up a numerical function alone is not runtime Python execution."""
        class Analysis(Experiment):
            """Resolve a callable without recording its execution."""

            def sequence(self, seq):
                seq.module(MODULE).ordered_keys  # No call recorded.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = Analysis().compile()
        self.assertEqual(caught, [])
        self.assertEqual(plan.document["warnings"], [])
        self.assertNotIn("cannot guarantee repeatability", plan.explain())

    def test_concrete_and_zero_argument_calls_always_record_and_warn_once(self):
        """No eager execution is hidden behind concrete inputs; aliases share warning deduplication."""
        class Analysis(Experiment):
            answer = FloatParameter(default=0.)
            def sequence(self, seq):
                module = seq.module(MODULE)
                first = module.counted()
                second = module.counted(2.)
                seq.set_parameter(self.answer, first + second + second)
                numpy = seq.module("numpy")
                numpy.abs(-1.)
                numpy.absolute(-2.)
        functions.CALLS.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RecordedPythonWarning)
            plan = Analysis().compile()
        self.assertEqual(functions.CALLS, [])
        notices = [w for w in caught if issubclass(w.category, RecordedPythonWarning)]
        self.assertEqual(len(notices), 2)  # counted and the aliased NumPy absolute ufunc.
        self.assertIn("external side effects", str(notices[0].message))
        self.assertEqual(len(plan.document["operations"]), 3)
        run = FakeExecutor().run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertEqual(functions.CALLS, [1., 2.])
        self.assertEqual(run.params["answer"], 5.)  # Reusing a reference never repeats the call.

    def test_fft_dynamic_index_updates_a_scalar_parameter(self):
        """Compare a known cosine coefficient with the independent analytical value 0.5."""
        from examples.recorded_fft import RecordedFFT
        plan = compile_quiet(RecordedFFT())
        run = FakeExecutor().run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertAlmostEqual(run.params["component"], 0.5)
        self.assertEqual(type(run.params["component"]), float)
        self.assertEqual(plan.document["version"], 6)
        spec = next(iter(plan.document["operations"].values()))
        self.assertEqual((spec["module"], spec["attribute"]), ("numpy.fft", "fft"))
        self.assertEqual(spec["version"], np.__version__)
        self.assertFalse(spec["safe_for_recording"])
        self.assertIn("numpy.fft.fft", plan.explain())
        self.assertIn(".real", plan.explain())

    def test_nested_symbolic_arguments_and_reviewed_wrapper(self):
        """Lists, tuples and keyword dictionaries retain nested runtime references and result types."""
        class Analysis(Experiment):
            offset = FloatParameter(default=4.)
            answer = FloatParameter(default=0.)
            def sequence(self, seq):
                values = seq.module("numpy").arange(4)
                nested = seq.module(MODULE).nested([values], options={"offset": self.offset})
                seq.set_parameter(self.answer, nested["pair"][0] + nested["pair"][1] + nested["z"].imag)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RecordedPythonWarning)
            plan = Analysis().compile()
        notices = [w for w in caught if issubclass(w.category, RecordedPythonWarning)]
        self.assertEqual(len(notices), 1)  # arange warns; the reviewed wrapper does not.
        reviewed = [spec for spec in plan.document["operations"].values() if spec["attribute"] == "nested"][0]
        self.assertTrue(reviewed["safe_for_recording"])
        run = FakeExecutor().run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertEqual(run.params["answer"], 13.)
        self.assertTrue(any(isinstance(value, dict) and isinstance(value["pair"], tuple)
                            for value in run.values.values()))

    def test_canonical_json_preserves_dictionary_and_keyword_order(self):
        """Canonical key sorting must not silently reorder numerical inputs or Python keywords."""
        class Analysis(Experiment):
            def sequence(self, seq):
                seq.module(MODULE).ordered_keys({"z": 1., "a": 2.}, z=3., a=4.)
        plan = compile_quiet(Analysis())
        loaded = CompiledPlan(json.loads(json.dumps(plan.document, sort_keys=True)))
        run = FakeExecutor().run(loaded)
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertEqual(next(iter(run.values.values())), (("z", "a"), ("z", "a")))

    def test_slices_and_tuple_indices_preserve_array_dtype_and_complex_values(self):
        """Symbolic slice bounds and multi-axis indices agree with ordinary NumPy indexing."""
        values = np.arange(12, dtype=np.float32).reshape(3, 4).astype(np.complex64) + 2j
        class Analysis(Experiment):
            start = IntParameter(default=1)
            def sequence(self, seq):
                copied = seq.module("numpy").copy(values)
                seq.module(MODULE).counted(copied[self.start:, ::-2].imag)
        functions.CALLS.clear()
        plan = compile_quiet(Analysis())
        run = FakeExecutor().run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        observed = functions.CALLS[0]
        np.testing.assert_array_equal(observed, values[1:, ::-2].imag)
        self.assertEqual(observed.dtype, np.dtype("float32"))
        saved = CompiledPlan(json.loads(json.dumps(plan.document)))
        self.assertEqual(saved.identity, plan.identity)

    def test_numpy_linear_algebra_tuple_results_need_no_adapter(self):
        """Normalize NumPy's named tuple result into ordinary numerical containers."""
        class Analysis(Experiment):
            largest = FloatParameter(default=0.)
            def sequence(self, seq):
                result = seq.module("numpy").linalg.eigh([[2., 0.], [0., 3.]])
                seq.set_parameter(self.largest, result[0][-1])
        run = FakeExecutor().run(compile_quiet(Analysis()))
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertEqual(run.params["largest"], 3.)
        self.assertIsInstance(next(iter(run.values.values())), tuple)

    def test_calls_detach_inputs_and_results_without_rerunning_references(self):
        """An impure callable cannot mutate earlier local arrays, plan literals, or retained result snapshots."""
        class Analysis(Experiment):
            answer = FloatParameter(default=0.)
            def sequence(self, seq):
                module = seq.module(MODULE)
                original = module.shared_array()
                changed = module.mutate({"arrays": [original]})
                seq.set_parameter(self.answer, original[0] + changed["arrays"][0][0])
        functions.SHARED[:] = [1., 2.]
        plan = compile_quiet(Analysis())
        run = FakeExecutor().run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertEqual(run.params["answer"], 1000.)
        functions.SHARED[:] = [7., 8.]
        original = next(value for value in run.values.values() if isinstance(value, np.ndarray))
        np.testing.assert_array_equal(original, [1., 2.])
        functions.SHARED[:] = [1., 2.]

    def test_array_locals_cannot_be_assigned_to_scalar_parameters(self):
        class Analysis(Experiment):
            target = FloatParameter(default=0.)
            def sequence(self, seq):
                seq.set_parameter(self.target, seq.module("numpy").arange(3))
        run = FakeExecutor().run(compile_quiet(Analysis()))
        self.assertEqual(run.outcome, "FAILED")
        self.assertIn("finite float", run.reason)

    def test_exceptions_and_unsupported_results_attempt_teardown(self):
        """Both function failures and unsupported object results skip later instructions and attempt cleanup."""
        for name, reason in (("fail", "analysis failed"), ("unsupported", "Unsupported recorded Python value")):
            class Analysis(Experiment):
                source = InstrumentRef(kind="MICROWAVE_SOURCE")
                def sequence(self, seq):
                    getattr(seq.module(MODULE), name)()
                    seq.call(self.source, "unreachable")
                def teardown(self, seq):
                    seq.call(self.source, "cleanup")
            experiment = Analysis()
            experiment.bind_instrument("source", "lab_source")
            calls = []
            run = FakeExecutor(lambda *args: calls.append(args)).run(compile_quiet(experiment))
            self.assertEqual(run.outcome, "FAILED")
            self.assertIn(reason, run.reason)
            self.assertEqual([c[2] for c in calls], ["cleanup"])

    def test_manifest_and_expression_validation_do_not_execute_dependencies(self):
        """Malformed operation references, unavailable values and arbitrary attributes fail validation."""
        class Analysis(Experiment):
            def sequence(self, seq):
                seq.module("numpy").sum([1, 2])
        plan = compile_quiet(Analysis())
        document = plan.document
        document["operations"]["python_1"]["module"] = "missing_analysis_dependency"
        loaded = CompiledPlan(document)  # Loading validates structure, not imports.
        with self.assertRaises(ModuleNotFoundError):
            preflight_run(loaded)
        for change in (
            lambda d: d["main"][2].update(operation="missing"),
            lambda d: d["main"][2].update(args=[{"op": "ref", "name": "missing"}]),
            lambda d: d["main"][2].update(args=[{"op": "component", "name": "__class__", "value": {"op": "literal", "value": 1}}]),
            lambda d: d["operations"]["python_1"].update(safe_for_recording="yes"),
            lambda d: d["main"][2].update(args=[{"op": "array", "dtype": "O", "shape": [1], "value": {"op": "literal", "value": [1]}}]),
        ):
            document = plan.document
            change(document)
            with self.assertRaises(CompileError):
                CompiledPlan(document)
        document = plan.document
        document["version"] = 5
        with self.assertRaisesRegex(CompileError, "recompile"):
            CompiledPlan(document)

    def test_unsupported_arguments_and_object_construction_error_early(self):
        """Reject callbacks and constructors rather than serializing code or arbitrary objects."""
        for argument in (object(), lambda x: x, {1: 2}, np.asarray([object()])):
            class Analysis(Experiment):
                def sequence(self, seq):
                    seq.module("numpy").array(argument)
            with self.assertRaises((CompileError, TypeError)):
                compile_quiet(Analysis())
        class Constructor(Experiment):
            def sequence(self, seq):
                seq.module("numpy").ndarray((2,))
        with self.assertRaisesRegex(CompileError, "constructors"):
            Constructor().compile()

    def test_saved_fft_executes_in_a_fresh_process_without_authoring_class(self):
        """The JSON manifest resolves NumPy and reproduces the analytical result independently."""
        from examples.recorded_fft import RecordedFFT
        plan = compile_quiet(RecordedFFT())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan.save(path)
            script = '''import json, sys
from auspex_v2.experiment import CompiledPlan
from tests.experiment.fake_executor import FakeExecutor
plan = CompiledPlan.load(sys.argv[1])
run = FakeExecutor().run(plan)
print(json.dumps([plan.identity, run.outcome, run.params['component']]))
'''
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
            result = subprocess.check_output([sys.executable, "-B", "-c", script, str(path)], env=env, text=True)
        identity, outcome, component = json.loads(result)
        self.assertEqual(identity, plan.identity)
        self.assertEqual(outcome, "SUCCEEDED")
        self.assertAlmostEqual(component, 0.5)
