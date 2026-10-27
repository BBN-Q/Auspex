"""Production execution must neither import nor inherit test-only infrastructure."""
import subprocess
import sys
import unittest

from auspex_v2.experiment.interpreter import Interpreter
from auspex_v2.experiment.runner import LiveInterpreter
from tests.experiment.fake_executor import FakeExecutor
from tests.experiment.pipeline_executor import PipelineExecutor
from .experiment_test_support import Experiment


class InterpreterBoundaryTest(unittest.TestCase):
    """Guard dependency direction and shared semantics without duplicating a runtime."""

    def test_production_imports_succeed_when_test_infrastructure_is_forbidden(self):
        """Fresh imports exercise transitive dependencies, including the package exports."""
        script = '''import importlib.abc
import importlib.util
import sys

removed = {
    "auspex_v2.experiment.fake", "auspex_v2.experiment.pipeline",
    "auspex_v2.experiment_fake", "auspex_v2.experiment_pipeline",
}
class RejectTestInfrastructure(importlib.abc.MetaPathFinder):
    """Fail rather than permit a production dependency on test executors."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tests" or fullname.startswith("tests.") or fullname in removed:
            raise AssertionError("Production imported test infrastructure: " + fullname)

finder = RejectTestInfrastructure()
sys.meta_path.insert(0, finder)
import auspex_v2.experiment as experiment
from auspex_v2.experiment import service, runner
from auspex_v2.experiment.interpreter import Interpreter
assert runner.LiveInterpreter.__bases__ == (Interpreter,)
assert not hasattr(runner.LiveInterpreter, "run")
assert not hasattr(experiment, "FakeExecutor")
assert not hasattr(experiment, "PipelineExecutor")
sys.meta_path.remove(finder)
for name in removed:
    assert importlib.util.find_spec(name) is None, name
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_backends_share_production_expression_and_control_flow(self):
        """Fixtures supply virtual I/O only, not their own interpretation of the plan."""
        for backend in (LiveInterpreter, FakeExecutor, PipelineExecutor):
            for method in ("execute", "_eval", "_set_parameter", "_if", "_repeat", "_while", "_sweep"):
                self.assertIs(getattr(backend, method), getattr(Interpreter, method))
        live = LiveInterpreter(None, None, None, Experiment().compile().document)
        for field in ("command", "publisher", "windows", "trace", "time", "retain_windows"):
            self.assertFalse(hasattr(live, field), field)

    def test_unsupplied_backend_actions_raise_instead_of_succeeding_silently(self):
        """The production core cannot substitute virtual waits, fake commands or captured publication."""
        runtime = Interpreter(Experiment().compile().document)
        with self.assertRaisesRegex(NotImplementedError, "waits"):
            runtime.execute([{"op": "WAIT", "duration": {"op": "literal", "value": 1.}, "id": "test"}])
        with self.assertRaisesRegex(NotImplementedError, "publication"):
            runtime.execute([{"op": "PUBLISH", "id": "test"}])
        with self.assertRaisesRegex(NotImplementedError, "command dispatch"):
            runtime._dispatch_command("GET", "source", "frequency", [], {})
