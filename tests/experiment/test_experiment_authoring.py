"""Minimize-surprise contracts for configuration, runtime state and proxies."""
import tempfile
from pathlib import Path
import unittest

from auspex_v2.experiment import (
    CompileError, CompiledPlan, FloatParameter, InstrumentRef, IntParameter,
)
from tests.experiment.fake_executor import FakeExecutor
from .experiment_test_support import Experiment
from auspex_v2.instruments import InstrumentKind
from auspex_v2.instruments.drivers import Instrument, operation
from auspex_v2.instruments.drivers.simulated import SimulatedVNA
from auspex_v2.instruments.drivers.proxy import InstrumentProxy
from auspex_v2.instruments.server.client import Session
from auspex_v2.filters.buffer import BufferSink


class Settings(Experiment):
    device = InstrumentRef(kind="SENSOR")
    bias = FloatParameter(default=1., bounds=[0., 3.], unit="V")
    count = IntParameter(default=1, bounds=(0, 4))


def configured(cls=Settings):
    experiment = cls()
    experiment.bind_instrument("device", "lab_device")
    experiment.bias.bind(experiment.device, "voltage", wait=0.5)
    return experiment


class AuthoringTest(unittest.TestCase):
    def test_fixed_kinds_and_driver_binding_forms(self):
        direct = InstrumentRef(driver=SimulatedVNA)
        self.assertEqual(direct.kind, InstrumentKind.VNA)
        separate = InstrumentRef()
        self.assertIs(separate.bind(SimulatedVNA), separate)
        self.assertEqual(separate.kind, InstrumentKind.VNA)
        chained = InstrumentRef().bind(SimulatedVNA)
        self.assertEqual(chained.kind, InstrumentKind.VNA)
        with self.assertRaisesRegex(CompileError, "disagrees"):
            InstrumentRef(kind=InstrumentKind.DC_SOURCE, driver=SimulatedVNA)
        with self.assertRaisesRegex(CompileError, "Unknown instrument KIND"):
            InstrumentRef(kind="OTHER")
        with self.assertRaisesRegex(CompileError, "read-only"):
            direct.kind = InstrumentKind.DC_SOURCE
        with self.assertRaisesRegex(CompileError, "read-only"):
            direct.driver = None

    def test_declarations_cannot_be_shadowed_or_deleted(self):
        experiment = configured()
        experiment.add_sweep(experiment.bias, [1., 2.])
        with self.assertRaisesRegex(CompileError, r"bias.value"):
            experiment.bias = 9.
        with self.assertRaisesRegex(CompileError, "bind_instrument"):
            experiment.device = InstrumentRef(kind="VNA")
        with self.assertRaises(CompileError):
            del experiment.bias
        with self.assertRaises(CompileError):
            del experiment.device
        experiment.bias.value = 2.
        self.assertEqual(experiment.compile().document["parameters"]["bias"]["value"], 2.)

    def test_recording_rejects_configuration_mutation(self):
        changes = [
            lambda e: setattr(e.bias, "value", 2.),
            lambda e: e.bias.bind(e.device, "other"),
            lambda e: setattr(e.bias.declaration, "unit", "A"),
            lambda e: e.bias.declaration.bounds.__setitem__(1, 5.),
            lambda e: e.bias.binding.kwargs.update(channel=2),
            lambda e: setattr(e.device, "kind", "OTHER"),
            lambda e: e.metadata.update(changed=True),
        ]
        for change in changes:
            class Changed(Settings):
                def setup(self, seq):
                    change(self)
            with self.subTest(change=change):
                experiment = configured(Changed)
                with self.assertRaisesRegex(CompileError, "frozen"):
                    experiment.compile()
                # The failed detached recording must not freeze/mutate its author.
                experiment.bias.value = 2.
                self.assertEqual(experiment.bias.declaration.bounds, [0., 3.])
                self.assertEqual(experiment.metadata, {})

    def test_recording_rejects_nested_sweep_mutation(self):
        class Changed(Settings):
            def sweep_begin(self, seq, axis):
                axis.points[0] = 3.
        experiment = configured(Changed)
        experiment.add_sweep(experiment.bias, [1., 2.])
        with self.assertRaisesRegex(CompileError, "frozen"):
            experiment.compile()
        self.assertEqual(experiment.sweeps[0].points, [1., 2.])

    def test_python_locals_and_symbolic_temporaries_are_allowed(self):
        class Locals(Settings):
            def setup(self, seq):
                self.readback = seq.get(self.device, "temperature")
            def sequence(self, seq):
                constants = [1., 2.]
                for value in constants:
                    seq.call(self.device, "observe", value, self.readback)
        plan = configured(Locals).compile()
        self.assertEqual(FakeExecutor(lambda *args: 12.).run(plan).outcome, "SUCCEEDED")

    def test_set_parameter_applies_binding_wait_and_runtime_reference(self):
        class Runtime(Settings):
            def sequence(self, seq):
                seq.set_parameter(self.bias, 2.)
                seq.call(self.device, "observe", self.bias)
            def teardown(self, seq):
                seq.set_parameter(self.bias, 0.)
        calls = []
        experiment = configured(Runtime)
        plan = experiment.compile()
        run = FakeExecutor(lambda *args: calls.append(args)).run(plan)
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual([(c[2], c[3]) for c in calls],
                         [("voltage", [2.]), ("observe", [2.]), ("voltage", [0.])])
        self.assertEqual(run.time, 1.)
        self.assertEqual(run.params["bias"], 0.)
        self.assertEqual(experiment.bias.value, 1.)
        self.assertIn("SET_PARAMETER bias requested=2.0", "\n".join(run.trace))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan.save(path)
            loaded = CompiledPlan.load(path)
            self.assertEqual(loaded.identity, plan.identity)
            self.assertEqual(FakeExecutor().run(loaded).params["bias"], 0.)

    def test_setup_update_is_not_implicitly_reapplied(self):
        class SetupUpdate(Settings):
            def setup(self, seq):
                seq.set_parameter(self.bias, 2.)
        calls = []
        run = FakeExecutor(lambda *args: calls.append(args)).run(configured(SetupUpdate).compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual([c[3] for c in calls], [[2.]])
        self.assertEqual(run.time, 0.5)

    def test_dynamic_wait_rejects_negative_duration(self):
        class Delay(Settings):
            delay = FloatParameter(default=0.)
            def sequence(self, seq):
                seq.set_parameter(self.delay, -1.)
                seq.wait(self.delay)
        run = FakeExecutor().run(configured(Delay).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertIn("Negative wait", run.reason)
        self.assertEqual(run.time, 0.)

    def test_direct_setting_does_not_change_parameter_state(self):
        class Direct(Settings):
            def sequence(self, seq):
                seq.set(self.device, "voltage", 0.)
                seq.call(self.device, "observe", self.bias)
        calls = []
        run = FakeExecutor(lambda *args: calls.append(args)).run(configured(Direct).compile())
        self.assertEqual(run.params["bias"], 1.)
        self.assertEqual(calls[-1][3], [1.])

    def test_runtime_values_checked_before_bound_command(self):
        class Dynamic(Settings):
            def sequence(self, seq):
                value = seq.get(self.device, "suggested_bias")
                seq.set_parameter(self.bias, value)
        calls = []
        def command(*args):
            calls.append(args)
            return 99.
        run = FakeExecutor(command).run(configured(Dynamic).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertIn("outside bounds", run.reason)
        self.assertEqual([c[3] for c in calls if c[0] == "SET"], [])
        self.assertEqual(run.params["bias"], 1.)
        self.assertEqual(run.time, 0.)

    def test_invalid_literals_and_foreign_parameters_rejected(self):
        foreign = Settings().bias
        for value in (4., "bad", True):
            class Invalid(Settings):
                def sequence(self, seq):
                    seq.set_parameter(self.bias, value)
            with self.subTest(value=value), self.assertRaises(CompileError):
                configured(Invalid).compile()
        class Foreign(Settings):
            def sequence(self, seq):
                seq.set_parameter(foreign, 1.)
        with self.assertRaisesRegex(CompileError, "belong"):
            configured(Foreign).compile()

    def test_teardown_initializes_unset_parameter_even_after_setup_failure(self):
        class Cleanup(Settings):
            bias = FloatParameter(bounds=(0., 3.))
            def setup(self, seq):
                seq.call(self.device, "fail")
            def teardown(self, seq):
                seq.set_parameter(self.bias, 0.)
                seq.call(self.device, "observe", self.bias)
        calls = []
        def command(*args):
            calls.append(args)
            if args[2] == "fail":
                raise RuntimeError("setup failure")
        run = FakeExecutor(command).run(configured(Cleanup).compile())
        self.assertEqual(run.outcome, "FAILED")
        self.assertEqual(run.params["bias"], 0.)
        self.assertEqual(calls[-1][3], [0.])

    def test_conditional_initialization_does_not_leak(self):
        class Conditional(Settings):
            bias = FloatParameter()
            def sequence(self, seq):
                with seq.if_(True):
                    seq.set_parameter(self.bias, 1.)
                seq.call(self.device, "observe", self.bias)
        with self.assertRaisesRegex(CompileError, "not initialized"):
            configured(Conditional).compile()

    def test_main_swept_runtime_updates_rejected_including_loaded_plans(self):
        class Swept(Settings):
            def sequence(self, seq):
                seq.set_parameter(self.bias, 0.)
        experiment = configured(Swept)
        experiment.add_sweep(experiment.bias, [1., 2.])
        with self.assertRaisesRegex(CompileError, "swept"):
            experiment.compile()
        experiment = configured()
        experiment.add_sweep(experiment.bias, [1., 2.])
        document = experiment.compile().document
        document["main"].insert(1, {"op": "SET_PARAMETER", "name": "bias", "value": {"op": "literal", "value": 0.}})
        with self.assertRaisesRegex(CompileError, "swept"):
            CompiledPlan(document)

    def test_updated_repeat_count_is_not_frozen_to_initial_value(self):
        class Repetitions(Settings):
            def sequence(self, seq):
                seq.set_parameter(self.count, 3)
                with seq.repeat(self.count, max_iterations=4):
                    seq.wait(0.25)
        run = FakeExecutor().run(configured(Repetitions).compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual(run.time, 0.75)  # Three repeats; defaults have no settling effect.
        self.assertEqual(run.params["count"], 3)

    def test_changed_parameter_waits_use_current_value_but_limits_stay_frozen(self):
        class Delay(Settings):
            def sequence(self, seq):
                seq.set_parameter(self.bias, 2.)
                seq.wait(self.bias)
        run = FakeExecutor().run(configured(Delay).compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual(run.time, 2.5)  # Explicit binding wait plus the updated duration.
        class Limit(Settings):
            def sequence(self, seq):
                seq.set_parameter(self.count, 2)
                with seq.while_(False, max_iterations=self.count):
                    pass
        with self.assertRaises(CompileError):
            configured(Limit).compile()

    def test_reflected_arithmetic_and_negation(self):
        class Arithmetic(Settings):
            def sequence(self, seq):
                seq.call(self.device, "values", 2 + self.bias, 8 - self.bias,
                         2 * self.bias, 12 / self.bias, -self.bias,
                         self.bias - 8, self.bias / 12)
        experiment = configured(Arithmetic)
        experiment.bias.value = 3.
        calls = []
        run = FakeExecutor(lambda *args: calls.append(args)).run(experiment.compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual(calls[-1][3], [5., 5., 6., 4., -3., -5., 0.25])

    def test_explanation_includes_bindings_configuration_and_nesting(self):
        experiment = configured()
        experiment.add_sweep(experiment.bias, [1., 2.])
        experiment.add_sweep(experiment.count, [1, 2])
        explanation = experiment.compile().explain()
        self.assertIn("device -> lab_device (KIND=SENSOR)", explanation)
        self.assertIn("$bias: float = 1.0 unit='V'", explanation)
        self.assertIn("binding=device.voltage", explanation)
        self.assertIn("wait=0.5", explanation)
        self.assertLess(explanation.index("count: parameters="), explanation.index("bias: parameters="))
        self.assertIn("runtime schema validation deferred", explanation)
        self.assertIn("test_memory: buffer", explanation)

    def test_buffer_public_api_is_consumer_only(self):
        sink = BufferSink("test", 1)
        for name in ("receive", "drain", "view", "clear"):
            self.assertTrue(callable(getattr(sink, name)))
        self.assertFalse(hasattr(sink, "put_nowait"))
        self.assertFalse(hasattr(sink, "get_nowait"))


class KeywordDevice(Instrument):
    from auspex_v2.instruments import InstrumentKind
    KIND = InstrumentKind.SENSOR

    @operation
    def save(self, **kwargs):
        pass


class ProxyKeywordTest(unittest.TestCase):
    def test_recording_proxy_keeps_routing_keywords_in_driver_mapping(self):
        class Save(Experiment):
            device = InstrumentRef(driver=KeywordDevice)
            def sequence(self, seq):
                self.device.save(name="trace", member="channel", args=[1], kwargs={"nested": 2},
                                 value=3, wait=True)
        experiment = Save()
        experiment.bind_instrument("device", "registered")
        calls = []
        run = FakeExecutor(lambda *args: calls.append(args)).run(experiment.compile())
        self.assertEqual(run.outcome, "SUCCEEDED")
        self.assertEqual(run.time, 0.)
        self.assertEqual(calls[0][1:3], ("registered", "save"))
        self.assertEqual(calls[0][4], {"name": "trace", "member": "channel", "args": [1],
                                      "kwargs": {"nested": 2}, "value": 3, "wait": True})

    def test_live_session_envelope_preserves_driver_keywords(self):
        requests = []
        class Client:
            def request(self, operation, **fields):
                requests.append((operation, fields))
        session = Session(Client(), None, "token", run=True)
        capabilities = {"settings": {"setting": {"read": True, "write": True,
                                                  "args": ["name", "member", "value"]}},
                        "operations": ["save"]}
        proxy = InstrumentProxy(session, "registered", capabilities)
        proxy.save(name="trace", member="x", value=3, wait=True)
        proxy.set_setting(5, name="trace", member="x", value=3)
        proxy.get_setting(name="trace", member="x", value=3)
        self.assertEqual([op for op, _ in requests], ["CALL", "SET", "GET"])
        for op, envelope in requests:
            self.assertEqual(envelope["name"], "registered")
            self.assertEqual(envelope["kwargs"]["name"], "trace")
            self.assertEqual(envelope["kwargs"]["member"], "x")
            self.assertEqual(envelope["kwargs"]["value"], 3)
        self.assertEqual(requests[1][1]["value"], 5)
        session.closed = True
        with self.assertRaises(RuntimeError):
            proxy.save(name="trace")
