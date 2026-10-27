"""Parameter associations record detached instructions, never runtime callbacks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from auspex_v2.experiment import CompileError, CompiledPlan, FloatParameter, InstrumentRef
from tests.experiment.fake_executor import FakeExecutor
from .experiment_test_support import Experiment


class RecordedSweep(Experiment):
    """Use an explicit multi-step setter before, during and after a sweep."""

    source = InstrumentRef(kind="MICROWAVE_SOURCE")
    frequency = FloatParameter(default=1., bounds=(0., 10.))

    @frequency.setter
    def set_frequency(self, seq, value):
        seq.set(self.source, "output", False)
        seq.set(self.source, "frequency", value * 2)
        seq.wait(0.01)
        seq.set(self.source, "output", True)

    def setup(self, seq):
        seq.set_parameter(self.frequency, self.frequency)

    def sequence(self, seq):
        seq.call(self.source, "observe", self.frequency)

    def teardown(self, seq):
        seq.set_parameter(self.frequency, 0.)


def configured(cls=RecordedSweep):
    experiment = cls()
    experiment.bind_instrument("source", "lab_source")
    return experiment


class ParameterSetterTest(unittest.TestCase):
    """Check association ownership, expansion, phase rules and saved-plan independence."""

    def test_multistep_setters_apply_in_setup_each_point_and_teardown(self):
        """Repeated points run the full operation in order, without rewriting acquisition coordinates."""
        experiment = configured()
        experiment.add_sweep(experiment.frequency, [2., 2., 3.])
        calls = []
        plan = experiment.compile()
        run = FakeExecutor(lambda *args: calls.append(args)).run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        expected = []
        for value, observe in ((1., False), (2., True), (2., True), (3., True), (0., False)):
            expected.extend([("output", [False]), ("frequency", [value * 2]), ("output", [True])])
            if observe:
                expected.append(("observe", [value]))
        self.assertEqual([(c[2], c[3]) for c in calls], expected)
        self.assertAlmostEqual(run.time, 0.05)
        self.assertEqual(run.params["frequency"], 0.)
        self.assertEqual(run.coordinates, {"frequency": 3.})
        self.assertEqual(experiment.frequency.value, 1.)
        self.assertIn("WAIT", plan.explain())

    def test_explicit_main_updates_reapply_unswept_setters(self):
        """Every legal explicit update expands its recorder, including equal consecutive values."""
        class Updates(RecordedSweep):
            def setup(self, seq):
                pass
            def sequence(self, seq):
                seq.set_parameter(self.frequency, 2.)
                seq.set_parameter(self.frequency, 2.)
            def teardown(self, seq):
                pass
        calls = []
        run = FakeExecutor(lambda *args: calls.append(args)).run(configured(Updates).compile())
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertEqual([(c[2], c[3]) for c in calls],
                         [("output", [False]), ("frequency", [4.]), ("output", [True])] * 2)
        self.assertAlmostEqual(run.time, 0.02)

    def test_class_bindings_and_parameter_references_are_instance_local(self):
        """Copy inherited declarations together, including symbolic wait/keyword parameter references."""
        class Simple(Experiment):
            source = InstrumentRef(kind="MICROWAVE_SOURCE")
            delay = FloatParameter(default=0.25)
            channel = FloatParameter(default=2.)
            frequency = FloatParameter(default=1.).bind(source, "frequency", wait=delay,
                                                       kwargs={"channel": channel})
            def setup(self, seq):
                seq.set_parameter(self.frequency, self.frequency)
        class Child(Simple):
            """Inherit the association without sharing mutable authoring configuration."""
        first, second = configured(Child), configured(Child)
        self.assertIs(first.frequency.binding.instrument, first.source)
        self.assertIs(first.frequency.binding.wait, first.delay)
        self.assertIs(first.frequency.binding.kwargs["channel"], first.channel)
        first.delay.value = 0.5
        first.frequency.binding.kwargs["extra"] = 3
        self.assertNotIn("extra", second.frequency.binding.kwargs)
        self.assertNotIn("extra", Simple.frequency.binding.kwargs)
        calls = []
        run = FakeExecutor(lambda *args: calls.append(args)).run(second.compile())
        self.assertEqual(run.outcome, "SUCCEEDED", run.reason)
        self.assertEqual([(c[2], c[3], c[4]) for c in calls], [("frequency", [1.], {"channel": 2.})])
        self.assertEqual(run.time, 0.25)

    def test_instance_replacement_in_both_directions_and_inheritance(self):
        """An external recorder replaces the whole association on only the configured instance."""
        class Child(RecordedSweep):
            """Inherit the declared recorder."""
        first, second = configured(Child), configured(Child)
        first.frequency.bind(first.source, "simple")
        calls = []
        FakeExecutor(lambda *args: calls.append(args)).run(first.compile())
        self.assertEqual([c[2] for c in calls], ["simple", "observe", "simple"])
        def external(exp, seq, value):
            seq.call(exp.source, "external", value)
        first.frequency.setter(external)
        calls.clear()
        FakeExecutor(lambda *args: calls.append(args)).run(first.compile())
        self.assertEqual([c[2] for c in calls], ["external", "observe", "external"])
        self.assertIs(second.frequency.binding, RecordedSweep.frequency.binding)
        second_calls = []
        FakeExecutor(lambda *args: second_calls.append(args)).run(second.compile())
        self.assertIn("output", [c[2] for c in second_calls])

    def test_defaults_do_not_invoke_any_association(self):
        """A configured initial value is requested state, not implicit hardware initialization."""
        class Defaults(RecordedSweep):
            def setup(self, seq):
                pass
            def teardown(self, seq):
                pass
        for simple in (False, True):
            experiment = configured(Defaults)
            if simple:
                experiment.frequency.bind(experiment.source, "frequency")
            calls = []
            run = FakeExecutor(lambda *args: calls.append(args)).run(experiment.compile())
            self.assertEqual([c[2] for c in calls], ["observe"])
            self.assertEqual(run.time, 0.)
            self.assertEqual(run.params["frequency"], 1.)

    def test_recursion_reports_direct_and_indirect_cycles(self):
        """Reject recursive recording even when it appears inside a runtime conditional."""
        for indirect in (False, True):
            experiment = configured()
            def first(exp, seq, value):
                with seq.if_(value > 0):
                    seq.set_parameter(exp.other if indirect else exp.frequency, value)
            if indirect:
                class Pair(RecordedSweep):
                    other = FloatParameter(default=1.)
                    @other.setter
                    def set_other(self, seq, value):
                        seq.set_parameter(self.frequency, value)
                experiment = configured(Pair)
            experiment.frequency.setter(first)
            expected = "frequency → other → frequency" if indirect else "frequency → frequency"
            with self.assertRaisesRegex(CompileError, expected):
                experiment.compile()
            experiment.frequency.bind(experiment.source, "frequency")
            self.assertEqual(FakeExecutor().run(experiment.compile()).outcome, "SUCCEEDED")

    def test_setter_configuration_mutations_rejected_on_detached_copy(self):
        """Setter bodies obey the same attribute and nested configuration freeze as hooks."""
        changes = [lambda exp: exp.frequency.bind(exp.source, "changed"),
                   lambda exp: exp.frequency.setter(lambda exp, seq, value: None),
                   lambda exp: exp.metadata.update(changed=True),
                   lambda exp: exp.frequency.declaration.bounds.__setitem__(1, 20.)]
        for change in changes:
            experiment = configured()
            experiment.frequency.declaration.bounds = [0., 10.]
            def mutate(exp, seq, value):
                change(exp)
            experiment.frequency.setter(mutate)
            with self.assertRaisesRegex(CompileError, "frozen"):
                experiment.compile()
            self.assertEqual(experiment.metadata, {})
            self.assertEqual(experiment.frequency.declaration.bounds, [0., 10.])
            experiment.frequency.bind(experiment.source, "frequency")

    def test_dynamic_validation_precedes_every_setter_operation(self):
        """An invalid runtime value must not even execute the setter's first output command."""
        class Dynamic(RecordedSweep):
            def setup(self, seq):
                value = seq.get(self.source, "requested")
                seq.set_parameter(self.frequency, value)
            def teardown(self, seq):
                pass
        calls = []
        def command(*args):
            calls.append(args)
            return 99.
        run = FakeExecutor(command).run(configured(Dynamic).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertIn("outside bounds", run.reason)
        self.assertEqual([c[2] for c in calls], ["requested"])

    def test_teardown_setter_runs_after_setup_failure_without_an_initial_value(self):
        """Cleanup may initialize a swept parameter even when setup never reached its update."""
        class Cleanup(RecordedSweep):
            frequency = FloatParameter(bounds=(0., 10.))
            @frequency.setter
            def set_frequency(self, seq, value):
                seq.set(self.source, "frequency", value)
            def setup(self, seq):
                seq.call(self.source, "fail")
        experiment = configured(Cleanup)
        experiment.add_sweep(experiment.frequency, [1., 2.])
        calls = []
        def command(*args):
            calls.append(args)
            if args[2] == "fail":
                raise RuntimeError("setup failed")
        run = FakeExecutor(command).run(experiment.compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertEqual([(c[2], c[3]) for c in calls], [("fail", []), ("frequency", [0.])])
        self.assertEqual(run.params["frequency"], 0.)

    def test_setters_are_not_needed_to_load_and_execute_saved_plans(self):
        """A fresh process executes only JSON instructions, with identical behavior and identity."""
        experiment = configured()
        experiment.add_sweep(experiment.frequency, [2., 2.])
        plan = experiment.compile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan.save(path)
            script = '''import json, sys
from auspex_v2.experiment import CompiledPlan
from tests.experiment.fake_executor import FakeExecutor
plan = CompiledPlan.load(sys.argv[1])
calls = []
run = FakeExecutor(lambda *args: calls.append(args)).run(plan)
print(json.dumps([plan.identity, run.outcome, run.time, [c[2] for c in calls]]))
'''
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
            output = subprocess.check_output([sys.executable, "-B", "-c", script, str(path)],
                                             env=env, text=True)
        identity, outcome, elapsed, members = json.loads(output)
        self.assertEqual(identity, plan.identity)
        self.assertEqual(outcome, "SUCCEEDED")
        self.assertAlmostEqual(elapsed, 0.04)
        self.assertEqual(members.count("frequency"), 4)
        self.assertNotIn("set_frequency", json.dumps(plan.document))

    def test_recorder_registration_rejects_nonrecording_functions(self):
        """Reject coroutine/generator recorders rather than silently recording no operations."""
        async def asynchronous(exp, seq, value):
            pass
        def generator(exp, seq, value):
            yield value
        for function in (asynchronous, generator, lambda value: None, object()):
            with self.assertRaises(CompileError):
                configured().frequency.setter(function)
