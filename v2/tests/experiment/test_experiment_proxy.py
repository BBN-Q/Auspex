"""The live driver-shaped proxy also records offline compiler instructions."""
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from auspex_v2.experiment import CompileError, CompiledPlan, InstrumentRef
from auspex_v2.experiment.fake import FakeExecutor
from .experiment_test_support import Experiment
from auspex_v2.instruments.drivers import Instrument, managed_property, operation
from auspex_v2.instruments.drivers.simulated import SimulatedSource, SimulatedVNA


class ProxyExperiment(Experiment):
    source = InstrumentRef(kind="MICROWAVE_SOURCE", driver=SimulatedSource)
    vna = InstrumentRef(kind="VNA", driver=SimulatedVNA)

    def setup(self, seq):
        self.source.output = True

    def sequence(self, seq):
        self.source.frequency = 5e9
        frequency = self.source.frequency
        self.source.set_channel_frequency(frequency, channel=2)
        frequency = self.source.get_channel_frequency(channel=2)
        with seq.if_(frequency > 0):
            self.source.set_frequency(frequency)
        self.vna.get_trace(measurement="S21")  # CALL, not GET or automatic acquisition
        seq.wait(0.5)
        seq.set(self.source, "output", False, wait=0.25)
        seq.call(self.vna, "get_trace", kwargs={"measurement": "S21"})
        seq.publish(self.signal, data=1.0)

    def teardown(self, seq):
        self.source.output = False


def configured(cls=ProxyExperiment):
    exp = cls()
    exp.bind_instrument("source", "lab_source")
    exp.bind_instrument("vna", "lab_vna")
    return exp


class ProxyCompilerTest(TestCase):
    def test_records_all_forms_without_driver_construction(self):
        with mock.patch.object(SimulatedSource, "__init__", side_effect=AssertionError("No driver construction")):
            plan = configured().compile()
        commands = []
        def command(op, name, member, args, kwargs):
            commands.append((op, name, member, args, kwargs))
            return 5e9
        run = FakeExecutor(command).run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual([c[0] for c in commands],
                         ["SET", "SET", "GET", "SET", "GET", "SET", "CALL", "SET", "CALL", "SET"])
        self.assertEqual(commands[3], ("SET", "lab_source", "channel_frequency", [5e9], {"channel": 2}))
        self.assertEqual(commands[6], ("CALL", "lab_vna", "get_trace", [], {"measurement": "S21"}))
        self.assertEqual(len(run.windows), 1)
        self.assertEqual(run.time, 0.75)

    def test_proxy_and_explicit_forms_compile_identically(self):
        class Explicit(ProxyExperiment):
            def setup(self, seq): seq.set(self.source, "output", True)
            def sequence(self, seq):
                seq.set(self.source, "frequency", 5e9)
                frequency = seq.get(self.source, "frequency")
                seq.set(self.source, "channel_frequency", frequency, kwargs={"channel": 2})
                frequency = seq.get(self.source, "channel_frequency", kwargs={"channel": 2})
                with seq.if_(frequency > 0): seq.set(self.source, "frequency", frequency)
                seq.call(self.vna, "get_trace", kwargs={"measurement": "S21"})
                seq.wait(0.5)
                seq.set(self.source, "output", False, wait=0.25)
                seq.call(self.vna, "get_trace", kwargs={"measurement": "S21"})
                seq.publish(self.signal, data=1.0)
            def teardown(self, seq): seq.set(self.source, "output", False)
        self.assertEqual(configured().compile().identity, configured(Explicit).compile().identity)

    def test_access_outside_compile_and_failed_compile_cleanup(self):
        exp = configured()
        with self.assertRaises(RuntimeError): exp.source.frequency = 1
        with self.assertRaises(RuntimeError): exp.source.get_frequency()
        captured = []
        class Bad(ProxyExperiment):
            def sequence(self, seq):
                captured.append(self.source.get_frequency)
                raise ValueError("author error")
        with self.assertRaises(ValueError): configured(Bad).compile()
        with self.assertRaises(RuntimeError): captured[0]()
        exp.compile()
        with self.assertRaises(RuntimeError): exp.source.frequency

    def test_member_permissions_and_parameterized_args(self):
        for action, error in [
            (lambda e, s: setattr(e.source, "typo", 1), AttributeError),
            (lambda e, s: e.source.get_channel_frequency(), TypeError),
            (lambda e, s: e.source.channel_frequency, AttributeError),
            (lambda e, s: s.call(e.source, "get_frequency"), CompileError),
            (lambda e, s: s.set(e.source, "typo", 1), CompileError),
        ]:
            class Bad(ProxyExperiment):
                def sequence(self, seq): action(self, seq)
            with self.subTest(action=action), self.assertRaises(error): configured(Bad).compile()

    def test_driver_wait_keyword_is_not_runner_wait(self):
        class Device(Instrument):
            from auspex_v2.instruments import InstrumentKind
            KIND = InstrumentKind.SENSOR

            @operation
            def perform(self, *, wait=False): pass
            @managed_property
            def readonly(self): return 1
        class Job(Experiment):
            device = InstrumentRef(driver=Device)
            def sequence(self, seq): self.device.perform(wait=True)
        exp = Job()
        exp.bind_instrument("device", "test")
        plan = exp.compile()
        call = next(node for node in plan.document["main"] if node["op"] == "CALL")
        self.assertEqual(call["kwargs"]["wait"], {"op": "literal", "value": True})
        self.assertNotIn("WAIT", plan.explain())
        class Bad(Job):
            def sequence(self, seq): self.device.readonly = 2
        bad = Bad()
        bad.bind_instrument("device", "test")
        with self.assertRaises(AttributeError): bad.compile()

    def test_roundtrip_no_driver_class_or_server_dependency(self):
        plan = configured().compile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan.save(path)
            self.assertNotIn("SimulatedVNA", path.read_text())
            self.assertEqual(plan.identity, CompiledPlan.load(path).identity)
            script = '''import sys
from auspex_v2.experiment import CompiledPlan
from auspex_v2.experiment.fake import FakeExecutor
assert "auspex_v2.instruments.server.service" not in sys.modules
p = CompiledPlan.load(sys.argv[1])
r = FakeExecutor(lambda *args: 5e9).run(p)
assert r.outcome == "SUCCEEDED", r.reason
assert "auspex_v2.instruments.server.service" not in sys.modules
'''
            subprocess.run([sys.executable, "-B", "-c", script, str(path)], check=True)
