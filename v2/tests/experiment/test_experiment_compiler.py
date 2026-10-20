"""Offline compiler contracts; no hardware, Qt, workers, or service processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from auspex_v2.experiment import (CompileError, CompiledPlan, FloatParameter,
                                  InstrumentRef, IntParameter)
from auspex_v2.experiment.fake import FakeExecutor
from .experiment_test_support import Experiment


class Measurement(Experiment):
    source = InstrumentRef(kind="SOURCE")
    x = FloatParameter(default=1, bounds=(0, 10), unit="V")
    y = FloatParameter(default=2)
    delay = FloatParameter(default=0.5)

    def setup(self, seq):
        seq.call(self.source, "setup")

    def sweep_begin(self, seq, axis):
        seq.call(self.source, "begin", axis.name)

    def sequence(self, seq):
        seq.acquire(self.source, "measure", output="signal")

    def sweep_end(self, seq, axis):
        seq.call(self.source, "end", axis.name)

    def teardown(self, seq):
        seq.call(self.source, "cleanup")


def configured(cls=Measurement):
    exp = cls()
    exp.bind_instrument("source", "lab_source")
    exp.x.bind(exp.source, "x", wait=exp.delay)
    exp.y.bind(exp.source, "y")
    exp.outputs = {"signal": {"dtype": "float64", "rank": 0}}
    return exp


class CompilerTest(unittest.TestCase):
    def test_sweep_hooks_order_waits_and_coordinates(self):
        exp = configured()
        exp.add_sweep(exp.x, [1, 2])
        exp.add_sweep(exp.y, [3, 4])
        calls = []
        run = FakeExecutor(lambda op, name, member, args, kwargs: calls.append((member, args))).run(exp.compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual(calls, [
            ("setup", []), ("begin", ["y"]), ("y", [3]), ("begin", ["x"]),
            ("x", [1]), ("measure", []), ("x", [2]), ("measure", []), ("end", ["x"]),
            ("y", [4]), ("begin", ["x"]), ("x", [1]), ("measure", []),
            ("x", [2]), ("measure", []), ("end", ["x"]), ("end", ["y"]), ("cleanup", [])])
        self.assertEqual([w["coordinates"] for w in run.windows],
                         [{"y": y, "x": x} for y in [3, 4] for x in [1, 2]])
        self.assertEqual(run.time, 2)
        self.assertEqual(len([s for s in run.trace if "WAIT " in s]), 4)

    def test_no_sweeps_static_parameters_and_one_window(self):
        calls = []
        run = FakeExecutor(lambda op, name, member, args, kwargs: calls.append(member)).run(configured().compile())
        self.assertEqual(calls, ["setup", "x", "y", "measure", "cleanup"])
        self.assertEqual(len(run.windows), 1)
        self.assertEqual(run.time, 0.5)

    def test_joint_path_and_repeated_settings(self):
        exp = configured()
        exp.add_sweep((exp.x, exp.y), [[1, 3], [1, 3], [2, 4]], name="path")
        run = FakeExecutor().run(exp.compile())
        self.assertEqual(len(run.windows), 3)
        self.assertEqual(run.time, 1.5)
        self.assertEqual(run.windows[1]["coordinates"], {"x": 1, "y": 3})

    def test_saved_plan_detached_and_fresh_process(self):
        exp = configured()
        points = [1, 2]
        exp.add_sweep(exp.x, points)
        plan = exp.compile()
        before = plan.identity
        exp.delay.value = 99
        exp.sweeps[0].points[0] = 9
        exp.metadata["changed"] = True
        doc = plan.document
        doc["parameters"]["delay"]["value"] = 123
        self.assertEqual(plan.identity, before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan.save(path)
            loaded = CompiledPlan.load(path)
            self.assertEqual(loaded.identity, before)
            script = '''import json, sys
from auspex_v2.experiment import CompiledPlan
from auspex_v2.experiment.fake import FakeExecutor
p = CompiledPlan.load(sys.argv[1])
r = FakeExecutor().run(p)
print(json.dumps([p.identity, r.time, r.windows, r.outcome]))
'''
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            result = subprocess.check_output([sys.executable, "-B", "-c", script, str(path)], env=env, text=True)
            identity, seconds, windows, outcome = json.loads(result)
            self.assertEqual((identity, seconds, outcome), (before, 1, "SUCCEEDED"))
            self.assertEqual([w["coordinates"]["x"] for w in windows], [1, 2])
        self.assertIn("WAIT", plan.explain())
        self.assertIn("SWEEP", plan.explain())

    def test_conditions_state_and_repeat(self):
        class Branches(Measurement):
            def sequence(self, seq):
                counter = seq.variable("counter", type=int, initial=0)
                with seq.repeat(2):
                    seq.assign(counter, counter + 1)
                with seq.while_(counter < 4, max_iterations=3):
                    seq.assign(counter, counter + 1)
                with seq.if_(seq.all(counter == 4, seq.not_(False))):
                    seq.call(self.source, "yes", kwargs={"wait": True}, wait=0.25)
                with seq.else_():
                    seq.call(self.source, "no")
        calls = []
        run = FakeExecutor(lambda op, name, member, args, kwargs: calls.append((member, kwargs))).run(configured(Branches).compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertIn(("yes", {"wait": True}), calls)
        self.assertNotIn(("no", {}), calls)
        self.assertEqual(run.time, 0.75)

    def test_runtime_reads_and_no_payload_in_trace(self):
        class Read(Measurement):
            def sequence(self, seq):
                temperature = seq.get(self.source, "temperature")
                with seq.if_(temperature > 1):
                    seq.terminate("FAILED", reason="too hot")
                seq.call(self.source, "unreachable")
        run = FakeExecutor(lambda *args: 987654321).run(configured(Read).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertNotIn("987654321", "\n".join(run.trace))
        self.assertIn("cleanup", "\n".join(run.trace))
        self.assertNotIn("unreachable", "\n".join(run.trace))

    def test_failure_skips_wait_and_runs_cleanup(self):
        class Bad(Measurement):
            def sequence(self, seq):
                seq.call(self.source, "bad", wait=10)
        def command(op, name, member, args, kwargs):
            if member == "bad": raise RuntimeError("failed")
        run = FakeExecutor(command).run(configured(Bad).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertEqual(run.time, 0.5)
        self.assertIn("cleanup", "\n".join(run.trace))

    def test_bounded_while_and_early_success(self):
        class Endless(Measurement):
            def sequence(self, seq):
                with seq.while_(True, max_iterations=2):
                    seq.wait(0.1)
        run = FakeExecutor().run(configured(Endless).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertIn("iteration limit", run.reason)
        class Early(Measurement):
            def sequence(self, seq):
                seq.terminate(reason="done")
                seq.call(self.source, "unreachable")
        run = FakeExecutor().run(configured(Early).compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertNotIn("unreachable", "\n".join(run.trace))

    def test_frozen_loop_bounds_and_dynamic_repeat(self):
        class Loops(Measurement):
            count = IntParameter(default=2)

            def sequence(self, seq):
                with seq.repeat(self.count):
                    seq.wait(0.1)
                counter = seq.variable("i", type=int, initial=0)
                with seq.while_(counter < 2, max_iterations=self.count):
                    seq.assign(counter, counter + 1)
                count = seq.get(self.source, "count")
                with seq.repeat(count, max_iterations=3):
                    seq.wait(0.1)
        run = FakeExecutor(lambda *args: 2).run(configured(Loops).compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertAlmostEqual(run.time, 0.9)
        run = FakeExecutor(lambda *args: 4).run(configured(Loops).compile())
        self.assertEqual(run.outcome, "FAILED")

    def test_reject_bad_sweeps(self):
        for points in ([], [-1], [11]):
            exp = configured()
            exp.add_sweep(exp.x, points)
            with self.assertRaises(CompileError): exp.compile()
        exp = configured()
        exp.add_sweep((exp.x, exp.y), [[1]])
        with self.assertRaises(CompileError): exp.compile()
        exp = configured()
        exp.add_sweep(exp.x, [1])
        exp.add_sweep(exp.x, [2], name="again")
        with self.assertRaises(CompileError): exp.compile()

    def test_reject_undefined_values_python_bool_and_waits(self):
        class Undefined(Measurement):
            def sequence(self, seq):
                with seq.if_(True): value = seq.get(self.source, "x")
                seq.set(self.source, "x", value)
        with self.assertRaisesRegex(CompileError, "every path"):
            configured(Undefined).compile()
        class PythonIf(Measurement):
            def sequence(self, seq):
                if seq.get(self.source, "x") > 0: pass
        with self.assertRaises(TypeError): configured(PythonIf).compile()
        class Negative(Measurement):
            def sequence(self, seq): seq.wait(-0.5)
        with self.assertRaises(CompileError): configured(Negative).compile()
        class BadCondition(Measurement):
            def sequence(self, seq):
                with seq.if_(4): pass
        with self.assertRaises(CompileError): configured(BadCondition).compile()

    def test_reject_unbound_unknown_instructions_and_version(self):
        with self.assertRaises(CompileError): Measurement().compile()
        doc = configured().compile().document
        doc["version"] = 100
        with self.assertRaises(CompileError): CompiledPlan(doc)
        doc["version"] = 1
        doc["main"][0]["op"] = "EXEC_PYTHON"
        with self.assertRaises(CompileError): CompiledPlan(doc)


if __name__ == "__main__":
    unittest.main()
