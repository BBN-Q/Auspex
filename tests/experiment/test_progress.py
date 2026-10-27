"""Point progress counts sequence completions, not acquisitions or loop iterations."""
import json
import unittest
from unittest.mock import Mock

from auspex_v2.experiment import FloatParameter, CompiledPlan, CompileError
from auspex_v2.experiment.progress import point_total
from auspex_v2.experiment.runner import LiveInterpreter
from auspex_v2.experiment.interpreter import ProgramTermination
from .experiment_test_support import Experiment


class Points(Experiment):
    x = FloatParameter(default=0.)
    y = FloatParameter(default=0.)
    z = FloatParameter(default=0.)

    def sequence(self, seq):
        with seq.repeat(3):
            with seq.if_(True):
                seq.wait(0.)


class ProgressTest(unittest.TestCase):
    def execute(self, experiment):
        """Exercise live point observation directly, without a fake whole-run lifecycle."""
        plan = experiment.compile()
        channel = Mock()
        channel.poll.return_value = False
        runner = LiveInterpreter(channel, Mock(), Mock(), plan.document)
        try:
            runner.execute(plan.document["main"])
        except ProgramTermination:
            pass
        messages = [json.loads(call.args[0]) for call in channel.send_bytes.call_args_list]
        return runner, [m["progress"] for m in messages if m["type"] == "progress"]

    def test_unswept_sequence_is_one_point_despite_internal_loops(self):
        runner, updates = self.execute(Points())
        self.assertEqual(runner.total, 1)
        self.assertEqual([p["completed"] for p in updates], [0, 1])

    def test_nested_and_joint_sweep_totals_and_coordinates(self):
        experiment = Points()
        experiment.add_sweep((experiment.x, experiment.y), [(1., 2.), (3., 4.)])
        experiment.add_sweep(experiment.z, [5., 6., 7.])
        runner, updates = self.execute(experiment)
        self.assertEqual(runner.total, 6)
        self.assertEqual(runner.completed, 6)
        self.assertEqual(updates[-1]["coordinates"], {"x": 3., "y": 4., "z": 7.})

    def test_early_termination_does_not_complete_point(self):
        class Early(Points):
            def sequence(self, seq):
                seq.terminate("FAILED", reason="stop")
        runner, updates = self.execute(Early())
        self.assertEqual(runner.completed, 0)
        self.assertEqual(updates[-1]["total"], 1)

    def test_unbalanced_boundaries_and_previous_version_rejected(self):
        document = Points().compile().document
        document["main"] = [n for n in document["main"] if n["op"] != "POINT_END"]
        with self.assertRaisesRegex(CompileError, "Unclosed point"):
            CompiledPlan(document)
        document = Points().compile().document
        document["version"] = 4
        with self.assertRaisesRegex(CompileError, "recompile"):
            CompiledPlan(document)

    def test_conditional_point_total_is_unknown(self):
        document = Points().compile().document
        document["main"] = [{"op": "IF", "then": document["main"], "otherwise": []}]
        self.assertIsNone(point_total(document))
