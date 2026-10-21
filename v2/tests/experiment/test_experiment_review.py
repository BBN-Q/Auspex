"""Regression tests for compiler state lifetimes and ordinary numerical authoring."""
import copy
import unittest

import numpy as np

from auspex_v2.experiment import (
    CompileError, CompiledPlan, FloatParameter, InstrumentRef,
    IntParameter, Parameter,
)
from auspex_v2.experiment.fake import FakeExecutor
from .experiment_test_support import Experiment


class SweepExperiment(Experiment):
    device = InstrumentRef(kind="SENSOR")
    x = FloatParameter()
    y = FloatParameter()

    def sequence(self, seq):
        seq.set(self.device, "x", self.x)


def swept(experiment_type=SweepExperiment):
    experiment = experiment_type()
    experiment.bind_instrument("device", "test")
    experiment.add_sweep(experiment.x, [1., 2.])
    experiment.add_sweep(experiment.y, [3., 4.])
    return experiment


class CompilerReviewTest(unittest.TestCase):
    def test_uninitialized_sweep_rejected_before_first_point(self):
        class SetupRead(SweepExperiment):
            def setup(self, seq):
                seq.set(self.device, "x", self.x)

        class BeginRead(SweepExperiment):
            def sweep_begin(self, seq, axis):
                if axis.name == "x":
                    seq.set(self.device, "x", self.x)

        class TeardownRead(SweepExperiment):
            def teardown(self, seq):
                seq.set(self.device, "x", self.x)

        for experiment_type in (SetupRead, BeginRead, TeardownRead):
            with self.subTest(experiment_type=experiment_type.__name__):
                with self.assertRaisesRegex(CompileError, "not initialized"):
                    swept(experiment_type).compile()

    def test_sweep_values_available_in_body_end_and_outer_value_in_inner_hook(self):
        class Hooks(SweepExperiment):
            def sweep_begin(self, seq, axis):
                if axis.name == "x":
                    seq.set(self.device, "outer", self.y)

            def sweep_end(self, seq, axis):
                seq.set(self.device, axis.name, axis.parameters[0])

        commands = []
        run = FakeExecutor(lambda *args: commands.append(args)).run(swept(Hooks).compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertTrue(all(command[3][0] is not None for command in commands))
        self.assertEqual([command[3][0] for command in commands if command[2] == "outer"], [3., 4.])

    def test_defaulted_sweep_can_be_read_in_setup_and_teardown(self):
        class Defaulted(SweepExperiment):
            x = FloatParameter(default=0.)

            def setup(self, seq):
                seq.set(self.device, "x", self.x)

            def teardown(self, seq):
                seq.set(self.device, "x", self.x)

        self.assertEqual(FakeExecutor().run(swept(Defaulted).compile()).outcome, "SUCCEEDED")

    def test_parameter_declaration_is_independent_per_instance(self):
        class Settings(Experiment):
            x = FloatParameter(default=1., bounds=[0., 2.])

        first, second = Settings(), Settings()
        first.x.declaration.bounds[1] = 99.
        first.x.declaration.default = 7.
        self.assertEqual(second.x.declaration.bounds, [0., 2.])
        self.assertEqual(Settings.x.bounds, [0., 2.])
        self.assertEqual(second.x.declaration.default, 1.)
        second.x.value = 3.
        with self.assertRaises(CompileError):
            second.compile()

    def test_numpy_values_normalized_before_validation(self):
        class Numerical(Experiment):
            delay = FloatParameter(default=np.float64(0.5), bounds=np.array([0., 1.]))
            repetitions = IntParameter(default=np.int64(2))
            enabled = Parameter(default=np.bool_(True), type=bool)

            def sequence(self, seq):
                with seq.if_(self.enabled):
                    with seq.repeat(self.repetitions):
                        seq.wait(self.delay)

        experiment = Numerical()
        experiment.delay.value = np.float32(0.25)
        experiment.metadata = {"nested": [np.int64(4), np.array([1., 2.])]}
        plan = experiment.compile()
        self.assertEqual(FakeExecutor().run(plan).time, 0.5)
        self.assertEqual(plan.document["metadata"], {"nested": [4, [1., 2.]]})
        experiment.delay.value = np.float64(-0.1)
        with self.assertRaises(CompileError):
            experiment.compile()
        experiment.repetitions.value = np.float64(2.)
        experiment.delay.value = 0.25
        with self.assertRaises(CompileError):
            experiment.compile()

    def test_incompatible_known_comparisons_rejected(self):
        class Comparison(Experiment):
            text = Parameter(default="abc", type=str)

            def sequence(self, seq):
                with seq.if_(self.text > 1):
                    pass

        with self.assertRaisesRegex(CompileError, "comparison"):
            Comparison().compile()

    def test_numeric_string_and_unknown_comparisons(self):
        class Comparison(Experiment):
            device = InstrumentRef(kind="SENSOR")
            integer = IntParameter(default=1)
            text = Parameter(default="abc", type=str)

            def sequence(self, seq):
                with seq.if_(seq.all(self.integer < 2.5, self.text < "xyz")):
                    seq.wait(0.25)
                value = seq.get(self.device, "value")
                with seq.if_(value > 0):
                    seq.wait(0.25)

        experiment = Comparison()
        experiment.bind_instrument("device", "test")
        plan = experiment.compile()
        self.assertEqual(FakeExecutor(lambda *args: 1).run(plan).time, 0.5)
        self.assertEqual(FakeExecutor(lambda *args: "wrong type").run(plan).outcome, "FAILED")

    def test_command_arguments_are_detached_each_invocation(self):
        class MutableArguments(Experiment):
            device = InstrumentRef(kind="SENSOR")

            def sequence(self, seq):
                result = seq.call(self.device, "read")
                with seq.repeat(2):
                    seq.call(self.device, "mutate", [1], result, kwargs={"nested": {"items": [2]}})

        experiment = MutableArguments()
        experiment.bind_instrument("device", "test")
        plan = experiment.compile()
        original = plan.document
        seen = []

        def command(op, device, member, args, kwargs):
            if member == "read":
                return [3]
            seen.append(copy.deepcopy((args, kwargs)))
            args[0].append(99)
            args[1].append(99)
            kwargs["nested"]["items"].append(99)

        run = FakeExecutor(command).run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(run.doc, original)
        self.assertEqual(plan.document, original)

    def test_validation_is_read_only_and_ids_are_a_separate_pass(self):
        from auspex_v2.experiment._experiment_plan import (
            PlanValidator, assign_instruction_ids, normalize_plan, walk_instructions,
        )

        document = swept().compile().document
        for phase in ("setup", "main", "teardown"):
            for node in walk_instructions(document[phase]):
                node.pop("id", None)
        normalize_plan(document)
        before_validation = copy.deepcopy(document)
        PlanValidator(document).validate()
        self.assertEqual(document, before_validation)
        assign_instruction_ids(document)
        self.assertEqual(document["main"][0]["id"], "main.000")
        self.assertEqual(document["main"][1]["body"][0]["id"], "main.001.body.000")

    def test_current_plan_normalization_and_roundtrip(self):
        class Counts(Experiment):
            count = IntParameter(default=2)

            def sequence(self, seq):
                with seq.repeat(self.count, max_iterations=self.count):
                    seq.wait(0.5)

        plan = Counts().compile()
        loop = plan.document["main"][1]
        self.assertEqual(loop["count"], {"op": "literal", "value": 2})
        self.assertEqual(loop["max_iterations"], 2)
        self.assertEqual(loop["body"][0]["id"], "main.001.body.000")
        self.assertEqual(FakeExecutor().run(plan).time, 1.)
        self.assertEqual(CompiledPlan(plan.document).identity, plan.identity)

    def test_explain_uses_readable_symbolic_expressions(self):
        class Explanation(Experiment):
            limit = IntParameter(default=2)

            def sequence(self, seq):
                counter = seq.variable("counter", type=int, initial=0)
                with seq.if_(counter < self.limit):
                    seq.assign(counter, counter + 1)

        text = Explanation().compile().explain()
        # Python gives the BoundParameter subclass's reflected comparison priority.
        self.assertIn("condition=($limit > %counter)", text)
        self.assertIn("value=(%counter + 1)", text)
        self.assertNotIn('"op": "lt"', text)

    def test_loaded_plans_check_parameter_availability(self):
        plan = swept().compile()
        document = plan.document
        document["setup"].append({"op": "SET", "instrument": "device", "member": "x",
                                  "value": {"op": "parameter", "name": "x"},
                                  "kwargs": {}, "result": "setup_result"})
        with self.assertRaisesRegex(CompileError, "not initialized"):
            CompiledPlan(document)


if __name__ == "__main__":
    unittest.main()
