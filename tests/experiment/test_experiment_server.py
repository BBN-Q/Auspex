"""Real local-service/runner tests with simulated instruments, never hardware."""
from datetime import datetime
import contextlib
import io
import json
import runpy
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings

import numpy as np
from unittest.mock import Mock, patch

from auspex_v2.data import AuspexStore
from auspex_v2.experiment import (Experiment, OutputConnector, InstrumentRef, ExperimentServerClient,
                                  RecordedPythonWarning, FloatParameter)
from auspex_v2.experiment.service import ExperimentServer
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, WriteToFile, ComplexAverage
from auspex_v2.instruments.drivers.simulated import SimulatedSource
from auspex_v2.instruments.server import InstrumentServerClient


class Measurement(Experiment):
    source = InstrumentRef(kind="MICROWAVE_SOURCE", driver=SimulatedSource)
    signal = OutputConnector(dtype="float64", max_bytes=1024)

    def setup_pipeline(self):
        source = PipelineInput("input")
        PipelineDefinition(inputs=[source], edges=[(source, WriteToFile(name="disk", dataset="signal").sink)])
        self.connect(self.signal, source)

    def sequence(self, seq):
        self.source.frequency = 5e9
        seq.wait(0.3)
        seq.publish(self.signal, data=1.0)

    def teardown(self, seq):
        self.source.output = False


def plan(cls=Measurement):
    experiment = cls()
    experiment.bind_instrument("source", "source")
    return experiment.compile()


class RunnerLifecycleTest(unittest.TestCase):
    """Observe production lifecycle ordering, not another executor's implementation."""

    def test_seal_teardown_and_lease_release_precede_pipeline_drain(self):
        from auspex_v2.experiment import runner
        from auspex_v2.filters import PipelineError

        class Immediate(Measurement):
            def sequence(self, seq):
                seq.publish(self.signal, data=1.0)

        document = plan(Immediate).document
        for drain_fails in (False, True):
            with self.subTest(drain_fails=drain_fails):
                events = []
                channel, client, session, pipeline = Mock(), Mock(), Mock(), Mock()
                channel.poll.return_value = False
                channel.recv_bytes.return_value = b'{"type": "ack"}'
                client.begin_run.return_value = session
                client.status.return_value = {
                    "run": {"run_id": "ordering", "orphaned": False},
                    "instruments": {"source": {"driver": "SimulatedSource",
                        "state": "CONNECTED", "live_sessions": 0}},
                }
                pipeline._seal_scan.side_effect = lambda: events.append("seal")
                session._dispatch.side_effect = lambda *args, **kwargs: events.append("teardown")
                session.close.side_effect = lambda: events.append("release")

                def drain():
                    events.append("drain")
                    if drain_fails:
                        raise PipelineError("incomplete required average")

                def observe_message(payload):
                    if json.loads(payload)["type"] == "terminal":
                        events.append("terminal")

                pipeline.close.side_effect = drain
                channel.send_bytes.side_effect = observe_message
                # Only I/O boundaries are replaced; compilation, admission, interpretation
                # and phase/exception handling all execute their production implementations.
                with patch.object(runner, "InstrumentServerClient", return_value=client), \
                     patch.object(runner, "ManualPipeline", return_value=pipeline):
                    runner.run(channel, document, "unused.auspex", "ordering", "unused.sock")

                self.assertEqual(events, ["seal", "teardown", "release", "drain", "terminal"])
                session._dispatch.assert_called_once_with(
                    "SET", "source", "output", args=[False], value=False, kwargs={})
                session.close.assert_called_once_with()
                pipeline.cancel.assert_called_once_with()
                channel.close.assert_called_once_with()
                terminal = json.loads(channel.send_bytes.call_args.args[0])
                self.assertEqual(terminal["state"], "FAILED" if drain_fails else "SUCCEEDED")
                self.assertTrue(terminal["lease_released"])
                if drain_fails:
                    self.assertIn("incomplete required average", terminal["reason"])
                else:
                    self.assertEqual(terminal["reason"], "")


class ExperimentServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.instrument_socket = self.root / "instruments.sock"
        self.socket = self.root / "experiments.sock"
        self.instrument_process = subprocess.Popen(
            [sys.executable, "-B", "-m", "auspex_v2.instruments.server", "--socket", str(self.instrument_socket), "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 10
        while True:
            try:
                with InstrumentServerClient(self.instrument_socket) as client:
                    client.register("source", "SimulatedSource", "SIM::SOURCE")
                break
            except (OSError, ConnectionError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)
        self.server = ExperimentServer(self.socket, instrument_socket=self.instrument_socket)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.client = ExperimentServerClient(self.socket)
        self.data = self.root / "data"
        self.data.mkdir()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.instrument_process.terminate()
        try:
            _, errors = self.instrument_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.instrument_process.kill()
            _, errors = self.instrument_process.communicate()
        self.tmp.cleanup()

    def test_short_run_identity_summary_and_actions_over_real_service(self):
        """Short IDs route real actions; human text omits paths without changing persisted data."""
        class Slow(Measurement):
            """Keep acquisition active long enough to inspect and cancel its real lease."""

            def sequence(self, seq):
                seq.wait(5)
                seq.publish(self.signal, data=1.0)

        compiled = plan(Slow)
        run = self.client.submit(compiled, name="vna_sweep", dir=self.data)
        short = run.run_id[:7]
        label = f"vna_sweep_000 [{short}]"
        self.assertEqual(str(run), label)
        self.assertEqual(repr(run), f"<RunHandle {label}>")
        self.assertNotIn(run.run_id, repr(run))
        self.assertEqual(self.client.run(short).run_id, run.run_id)
        self.assertEqual(self.client.request("plan", run_id=short), compiled.document)
        deadline = time.monotonic() + 10
        while run.status()["progress"]["phase"] != "acquisition":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.02)
        with InstrumentServerClient(self.instrument_socket) as instruments:
            lease = instruments.status()["run"]
            self.assertEqual(lease["run_id"], run.run_id)
            self.assertEqual(lease["name"], "vna_sweep_000")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.client.summary()
        self.assertIn("ExperimentServer: busy", output.getvalue())
        self.assertIn("Active run: " + label, output.getvalue())
        self.assertIn("acquisition:", output.getvalue())
        self.assertNotIn(str(self.data), output.getvalue())
        self.assertNotIn(run.run_id, output.getvalue())
        from utils.src.operator_commands import OperatorCommands
        commands = OperatorCommands(self.instrument_socket, self.socket)
        commands.execute(["experiment"])
        self.assertIn(label, commands.execute(["runs"]).text)
        self.assertIn(label, commands.execute(["cancel", short]).text)
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "CANCELLED")
        self.assertEqual(result["run_id"], run.run_id)
        self.assertEqual(result["path"], str(run.path))
        self.assertIn("RUN_FINISHED CANCELLED", self.client.request("log", run_id=short))
        self.assertEqual(json.loads((run.path / "status.json").read_text())["run_id"], run.run_id)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.client.summary()
        self.assertIn("ExperimentServer: ready", output.getvalue())
        self.assertIn(label + " — CANCELLED", output.getvalue())
        self.client.request("cleanup", run_id=short)
        with self.assertRaisesRegex(RuntimeError, "Unknown run ID"):
            self.client.run(short)
        self.assertTrue(run.path.is_dir())
        self.assertEqual(run.wait()["state"], "CANCELLED")  # Known-folder fallback still uses the full ID.

    def test_independent_trace_and_temperature_in_one_store(self):
        """Saved plans route unequal streams through shared or separate definitions.

        Separate definitions intentionally reuse local names; an output also feeds
        a third definition to verify explicit fan-out across attachments.
        """
        class Multi(Measurement):
            """Averaged traces and occasional scalar readings share sweep coordinates."""
            signal = OutputConnector(dtype="complex128", dimensions=("frequency",),
                                     axis_units={"frequency": "Hz"}, max_bytes=1024)
            temperature = OutputConnector(dtype="float64", unit="K", max_bytes=8)
            bias = FloatParameter(default=0., unit="V")
            separate = False

            def setup_pipeline(self):
                """Keep independent definition-local names valid in a combined runtime."""
                trace = PipelineInput("input")
                temperature = PipelineInput("input" if self.separate else "temperature")
                average = ComplexAverage(name="average", repeats=2)
                disk = WriteToFile(name="disk", group="vna", dataset="s21")
                readings = WriteToFile(name="disk" if self.separate else "readings",
                                       group="environment", dataset="temperature")
                trace_edges = [(trace, average.sink), (average.final_average, disk.sink)]
                if self.separate:
                    PipelineDefinition(inputs=[trace], edges=trace_edges)
                    PipelineDefinition(inputs=[temperature], edges=[(temperature, readings.sink)])
                    copy = PipelineInput("input")
                    PipelineDefinition(inputs=[copy], edges=[
                        (copy, WriteToFile(name="disk", group="vna", dataset="raw").sink)])
                    self.connect(self.signal, copy)
                else:
                    PipelineDefinition(inputs=[trace, temperature],
                                       edges=trace_edges + [(temperature, readings.sink)])
                self.connect(self.signal, trace)
                self.connect(self.temperature, temperature)

            def sequence(self, seq):
                trace = seq.module("numpy").array(np.array([1 + 2j, 3 + 4j]))
                for _ in range(2):
                    seq.publish(self.signal, data=trace, axes={"frequency": np.array([1., 2.])})
                with seq.if_(self.bias < 2.):
                    seq.publish(self.temperature, data=273. + self.bias)

        for separate in (False, True):
            with self.subTest(separate=separate):
                experiment = Multi()
                experiment.separate = separate
                experiment.bind_instrument("source", "source")
                experiment.add_sweep(experiment.bias, [0., 1., 2., 3.])
                from auspex_v2.experiment import CompiledPlan
                # Exercise the detached JSON route rather than live authoring objects.
                with self.assertWarns(RecordedPythonWarning):
                    compiled = CompiledPlan(experiment.compile().document)
                run = self.client.submit(compiled, name=f"multi_{separate}", dir=self.data)
                result = run.wait(timeout=20)
                self.assertEqual(result["state"], "SUCCEEDED", result)
                with AuspexStore(run.path) as store:
                    self.assertEqual(len(store.list_datasets()), 3 if separate else 2)
                    trace = store.read_dataset("vna", "s21")
                    temperature = store.read_dataset("environment", "temperature")
                    if separate:
                        raw = store.read_dataset("vna", "raw")
                        np.testing.assert_array_equal(raw.data, np.tile([1 + 2j, 3 + 4j], (8, 1)))
                np.testing.assert_array_equal(trace.data, np.tile([1 + 2j, 3 + 4j], (4, 1)))
                np.testing.assert_array_equal(temperature.data, [273., 274.])
                np.testing.assert_array_equal(temperature.coordinates["bias"].values, [0., 1.])
                self.assertEqual(temperature.value_unit, "K")
                self.assertEqual(temperature.dimensions, ("bias",))
                with InstrumentServerClient(self.instrument_socket) as client:
                    self.assertIsNone(client.status()["run"])

    def test_saved_plan_submit_disconnect_reconnect_and_date_folder(self):
        saved = self.root / "plan.json"
        original = plan()
        original.save(saved)
        # This authoring-independent process exits immediately after submission.
        code = """
import json, sys
from auspex_v2.experiment import CompiledPlan, ExperimentServerClient
run = ExperimentServerClient(sys.argv[1]).submit(CompiledPlan.load(sys.argv[2]), name='demo', dir=sys.argv[3], date=True)
print(json.dumps({'id':run.run_id, 'path':str(run.path)}))
"""
        response = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code,
                                                      str(self.socket), str(saved), str(self.data)], text=True))
        run = ExperimentServerClient(self.socket).run(response["id"])
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "SUCCEEDED", result)
        self.assertTrue(result["lease_released"])
        self.assertEqual(result["progress"]["completed"], 1)
        self.assertEqual(result["progress"]["total"], 1)
        self.assertEqual(result["progress"]["windows_submitted"], 1)
        self.assertEqual(result["progress"]["phase"], "processing drain")
        self.assertEqual(json.loads((run.path / "status.json").read_text())["progress"], result["progress"])
        self.assertEqual(run.path.parent.name, datetime.now().strftime("%Y-%m-%d"))
        self.assertEqual(run.path.name, "demo_000.auspex")
        self.assertEqual(run.plan().identity, original.identity)
        self.assertIn("RUN_FINISHED SUCCEEDED", run.log())
        with AuspexStore(run.path) as store:
            self.assertEqual(store.list_datasets(), [("data", "signal")])
            self.assertEqual(store.read_run_metadata()["plan_identity"], original.identity)
        with InstrumentServerClient(self.instrument_socket) as client:
            self.assertIsNone(client.status()["run"])
        run.cleanup()
        self.assertEqual(run.wait()["state"], "SUCCEEDED")
        self.assertTrue(run.path.exists())
        self.assertEqual(self.client.list_runs(), [])

    def test_busy_and_invalid_preflight_do_not_allocate(self):
        run = self.client.submit(plan(), name="one", dir=self.data)
        with self.assertRaisesRegex(RuntimeError, "busy"):
            self.client.submit(plan(), name="two", dir=self.data)
        self.assertEqual([p.name for p in self.data.iterdir()], ["one_000.auspex"])
        run.wait(timeout=20)
        with self.assertRaisesRegex(RuntimeError, "absolute"):
            self.client.submit(plan(), name="bad", dir="relative")
        self.assertEqual(len(list(self.data.iterdir())), 1)

    def test_instrument_domain_blockers_reject_before_folder_allocation(self):
        with InstrumentServerClient(self.instrument_socket) as instruments:
            instruments.connect("source")
            live = instruments.live("source")
            with self.assertRaisesRegex(RuntimeError, "live sessions"):
                self.client.submit(plan(), name="blocked", dir=self.data)
            live.session.close()
            with instruments.begin_run("other"):
                with self.assertRaisesRegex(RuntimeError, r"owned by run \[other\]"):
                    self.client.submit(plan(), name="blocked", dir=self.data)
            instruments.register("spare", "SimulatedSource", "SIM::SPARE")
            instruments.terminate("spare")
            with self.assertRaisesRegex(RuntimeError, "explicit recovery"):
                self.client.submit(plan(), name="blocked", dir=self.data)
        self.assertEqual(list(self.data.iterdir()), [])

    def test_buffer_submission_rejected_before_date_directory_creation(self):
        from auspex_v2.filters.definitions import DataBuffer
        class Buffered(Measurement):
            def setup_pipeline(self):
                source = PipelineInput("input")
                PipelineDefinition(inputs=[source], edges=[(source, DataBuffer(name="memory").sink)])
                self.connect(self.signal, source)
        with self.assertRaisesRegex(RuntimeError, "writer-only"):
            self.client.submit(plan(Buffered), name="buffer", dir=self.data, date=True)
        self.assertEqual(list(self.data.iterdir()), [])

    def test_cooperative_cancellation_discards_partial_scan(self):
        class Slow(Measurement):
            def sequence(self, seq):
                seq.publish(self.signal, data=1.0)
                seq.wait(5)
        run = self.client.submit(plan(Slow), name="cancel", dir=self.data)
        deadline = time.monotonic() + 10
        while "WAIT 5" not in run.log():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.03)
        run.cancel()
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "CANCELLED", result)
        self.assertEqual(result["progress"]["completed"], 0)
        self.assertEqual(result["progress"]["total"], 1)
        self.assertTrue(result["lease_released"])
        self.assertIn("SET source.output", run.log())
        with AuspexStore(run.path) as store:
            self.assertEqual(store.list_datasets(), [])

    def test_documented_vna_example_publishes_named_coordinates_and_computation(self):
        """Exercise the manual's canonical example through real services and simulated workers."""
        namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "documented_vna_sweep.py"))
        experiment = namespace["SourceVnaSweep"]()
        experiment.bind_instrument("source", "source")
        experiment.bind_instrument("vna", "vna")
        frequencies = [5e9, 5.1e9, 5.2e9]
        experiment.add_sweep(experiment.source_frequency, frequencies)
        with InstrumentServerClient(self.instrument_socket) as instruments:
            instruments.register("vna", "SimulatedVNA", "SIM::VNA")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            compiled = experiment.compile()
        run = self.client.submit(compiled, name="documented_vna", dir=self.data)
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "SUCCEEDED", result)
        self.assertTrue(result["lease_released"])
        self.assertIn("COMPUTE numpy.min completed", run.log())
        self.assertIn("SET_PARAMETER minimum_magnitude requested=", run.log())
        with AuspexStore(run.path) as store:
            self.assertEqual(store.list_datasets(), [("measurement", "s21")])
            dataset = store.read_dataset("measurement", "s21")
        self.assertEqual(dataset.data.shape, (3, 201))
        self.assertEqual(dataset.data.dtype, np.dtype("complex64"))
        self.assertEqual(dataset.axes["source_frequency"].points, frequencies)
        self.assertEqual(dataset.axes["source_frequency"].unit, "Hz")
        np.testing.assert_allclose(dataset.coordinates["frequency"].values, np.linspace(6.09e9, 6.11e9, 201))
        np.testing.assert_array_equal(dataset.data[0], dataset.data[1])
        self.assertAlmostEqual(float(abs(dataset.data[0, 100])), 0.3, places=6)
        with InstrumentServerClient(self.instrument_socket) as instruments:
            with instruments.live("source") as source:
                self.assertFalse(source.output)

    def test_live_plot_plan_runs_and_remains_discoverable_after_completion(self):
        """A real service/runner publishes optional trace/map snapshots alongside saved S21."""
        from auspex_v2.filters.definitions import ComplexAverage, Plot
        from auspex_v2.filters.preview_client import poll_run
        namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "documented_vna_sweep.py"))

        class Live(namespace["SourceVnaSweep"]):
            """Exercise ordinary experiment authoring, not an injected test interpreter."""

            def setup_pipeline(self):
                trace = PipelineInput("trace")
                average = ComplexAverage(name="average", repeats=2)
                writer = WriteToFile(name="disk", group="measurement", dataset="s21")
                current = Plot(name="current")
                image = Plot(name="map", mode="map")
                PipelineDefinition(inputs=[trace], edges=[(trace, average.sink),
                    (average.final_average, writer.sink), (average.partial_average, current.sink),
                    (average.final_average, image.sink)])
                self.connect(self.trace, trace)

            def sequence(self, seq):
                for _ in range(2):
                    result = self.vna.get_trace(measurement="S21")
                    seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

        experiment = Live()
        experiment.minimum_magnitude.value = 0.0
        experiment.bind_instrument("source", "source")
        experiment.bind_instrument("vna", "vna")
        experiment.add_sweep(experiment.source_frequency, [5e9, 5.1e9, 5.2e9])
        with InstrumentServerClient(self.instrument_socket) as instruments:
            instruments.register("vna", "SimulatedVNA", "SIM::VNA")
        run = self.client.submit(experiment.compile(), name="live", dir=self.data)
        self.assertEqual(run.wait(timeout=20)["state"], "SUCCEEDED")
        discovered = self.client.list_runs()[0]
        preview = poll_run(discovered["path"], wanted=["map", "current"])
        self.assertEqual(preview["state"], "SUCCEEDED")
        self.assertEqual(preview["id"], run.run_id)
        image = preview["frames"]["map"][0]
        self.assertEqual(image.data.shape, (3, 201))
        np.testing.assert_array_equal(image.axes[0].points, [5e9, 5.1e9, 5.2e9])
        with AuspexStore(run.path) as store:
            np.testing.assert_array_equal(image.data, store.read_dataset("measurement", "s21").data)
        experiment.sweeps.clear()
        experiment.add_sweep(experiment.source_frequency, [5e9, 5e9])
        with self.assertRaisesRegex(RuntimeError, "strictly monotonic"):
            self.client.submit(experiment.compile(), name="invalid_map", dir=self.data)
        self.assertEqual(len(list(self.data.iterdir())), 1)

    def test_recorded_fft_executes_in_runner_and_publishes_analytical_result(self):
        """A real runner resolves the saved NumPy dependency and publishes its scalar update."""
        namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "recorded_fft.py"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RecordedPythonWarning)
            compiled = namespace["RecordedFFT"]().compile()
        run = self.client.submit(compiled, name="fft", dir=self.data)
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "SUCCEEDED", result)
        self.assertTrue(result["lease_released"])
        self.assertIn("COMPUTE numpy.fft.fft completed", run.log())
        with AuspexStore(run.path) as store:
            self.assertAlmostEqual(float(store.read_dataset("data", "fft_component").data.item()), 0.5)
        self.assertEqual(run.plan().identity, compiled.identity)

    def test_recorded_python_cancellation_waits_for_call_then_attempts_teardown(self):
        """A gated real function cannot be interrupted; cancellation prevents its following hardware command."""
        entered, release = self.root / "entered", self.root / "release"
        class Computing(Measurement):
            def sequence(self, seq):
                seq.module("tests.experiment.computation_functions").wait_for_release(str(entered), str(release))
                self.source.frequency = -1.0  # Must not execute after cancellation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RecordedPythonWarning)
            compiled = plan(Computing)
        run = self.client.submit(compiled, name="compute_cancel", dir=self.data)
        try:
            deadline = time.monotonic() + 10
            while not entered.exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            run.cancel()
            self.assertEqual(run.status()["state"], "RUNNING")
            self.assertNotIn("COMPUTE tests.experiment.computation_functions.wait_for_release completed", run.log())
        finally:
            release.touch()
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "CANCELLED", result)
        self.assertTrue(result["lease_released"])
        self.assertIn("COMPUTE tests.experiment.computation_functions.wait_for_release completed", run.log())
        self.assertIn("SET source.output", run.log())
        self.assertNotIn("SET source.frequency", run.log())
        with AuspexStore(run.path) as store:
            self.assertEqual(store.list_datasets(), [])

    def test_teardown_failure_orphans_and_preserves_completed_scan(self):
        class BadTeardown(Measurement):
            def teardown(self, seq):
                self.source.frequency = -1.0
                self.source.output = False  # Must not be attempted after the first error.
        run = self.client.submit(plan(BadTeardown), name="failed", dir=self.data)
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "FAILED", result)
        self.assertIn("safe state not established", result["reason"])
        self.assertFalse(result["lease_released"])
        self.assertNotIn("SET source.output", run.log())
        with InstrumentServerClient(self.instrument_socket) as client:
            self.assertTrue(client.status()["run"]["orphaned"])
        with AuspexStore(run.path) as store:
            self.assertEqual(store.list_datasets(), [("data", "signal")])

    def test_measurement_failure_with_successful_teardown_releases_lease(self):
        class BadPublication(Measurement):
            def sequence(self, seq):
                self.source.output = True
                seq.publish(self.signal, data=1)  # Integer, not the declared float64.

        class BadMeasurement(Measurement):
            def sequence(self, seq):
                self.source.frequency = -1.0
            def teardown(self, seq):
                # The instrument server quarantines this device on driver failure;
                # do not claim further commands to that device could succeed.
                seq.wait(0)

        # Run the publication case first: the driver-failure case quarantines source.
        for cls in (BadPublication, BadMeasurement):
            with self.subTest(failure=cls.__name__):
                run = self.client.submit(plan(cls), name=cls.__name__, dir=self.data)
                result = run.wait(timeout=20)
                self.assertEqual(result["state"], "FAILED", result)
                self.assertTrue(result["lease_released"])
                with AuspexStore(run.path) as store:
                    self.assertEqual(store.list_datasets(), [])
                with InstrumentServerClient(self.instrument_socket) as client:
                    self.assertIsNone(client.status()["run"])
                    if cls is BadPublication:
                        self.assertIn("dtype", result["reason"])
                        self.assertIn("SET source.output args=[False]", run.log())
                        with client.live("source") as source:
                            self.assertFalse(source.output)
                    else:
                        self.assertIn("WAIT 0", run.log())

    def test_connection_failure_releases_lease_before_setup(self):
        with InstrumentServerClient(self.instrument_socket) as client:
            client.deregister("source")
            client.register("source", "SimulatedSource", "SIM::FAIL_CONNECT")
        run = self.client.submit(plan(), name="connect", dir=self.data)
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "FAILED")
        self.assertTrue(result["lease_released"])
        self.assertNotIn("SET source.output", run.log())
        with InstrumentServerClient(self.instrument_socket) as client:
            status = client.status()
            self.assertIsNone(status["run"])
            self.assertEqual(status["instruments"]["source"]["state"], "UNKNOWN")

    def test_runner_death_orphans_instruments(self):
        class Slow(Measurement):
            def sequence(self, seq):
                seq.wait(5)
        run = self.client.submit(plan(Slow), name="crash", dir=self.data)
        deadline = time.monotonic() + 10
        while "WAIT 5" not in run.log():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.03)
        process = self.server.registry.runs[run.run_id]["process"]
        process.terminate()
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "FAILED")
        with InstrumentServerClient(self.instrument_socket) as client:
            self.assertTrue(client.status()["run"]["orphaned"])

    def test_vna_saved_sweep_through_separate_service_process(self):
        namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "server_vna_sweep.py"))
        experiment = namespace["Sweep"]()
        experiment.bind_instrument("source", "source")
        experiment.bind_instrument("vna", "vna")
        experiment.add_sweep(experiment.source_frequency, [5e9, 5.1e9])
        with InstrumentServerClient(self.instrument_socket) as client:
            client.register("vna", "SimulatedVNA", "SIM::VNA")
        socket_path = self.root / "separate.sock"
        process = subprocess.Popen([sys.executable, "-B", "-m", "auspex_v2.experiment.service",
                                    "--socket", str(socket_path), "--instrument-socket", str(self.instrument_socket)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            client = ExperimentServerClient(socket_path)
            deadline = time.monotonic() + 10
            while not socket_path.exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            run = client.submit(experiment.compile(), name="vna", dir=self.data)
            del client
            reconnect = ExperimentServerClient(socket_path).run(run.run_id)
            result = reconnect.wait(timeout=20)
            self.assertEqual(result["state"], "SUCCEEDED", result)
            self.assertIn("SET_PARAMETER source_frequency", reconnect.log())
            self.assertIn("WAIT 0.01", reconnect.log())
            with AuspexStore(reconnect.path) as store:
                self.assertEqual(store.list_datasets(), [("data", "response")])
                self.assertEqual(store.read_dataset("data", "response").data.shape, (2, 201))
        finally:
            process.send_signal(signal.SIGINT)
            _, errors = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, errors.decode())

    def test_log_write_failure_is_nonfatal(self):
        """A real log-open failure must expose a diagnostic without failing the run."""
        from auspex_v2.experiment.resources import RunFolder

        allocate = RunFolder.allocate

        def allocate_with_log_directory(*args, **kwargs):
            # Install the collision before spawning, without replacing logging or I/O.
            folder = allocate(*args, **kwargs)
            (folder.path / "execution.log").mkdir()
            return folder

        with patch.object(RunFolder, "allocate", side_effect=allocate_with_log_directory):
            run = self.client.submit(plan(), name="no_log", dir=self.data)
        result = run.wait(timeout=20)
        self.assertEqual(result["state"], "SUCCEEDED", result)
        self.assertTrue(result["lease_released"])
        self.assertIn("Best-effort execution log unavailable:", result["diagnostic"])
        self.assertIn(str(run.path / "execution.log"), result["diagnostic"])
        self.assertTrue((run.path / "execution.log").is_dir())

    def test_running_status_write_failure_prevents_setup(self):
        from auspex_v2.experiment import service
        original = service._atomic_json
        def fail_running(path, value):
            if value["state"] == "RUNNING":
                raise OSError("RUNNING write failed")
            return original(path, value)
        with patch.object(service, "_atomic_json", fail_running):
            run = self.client.submit(plan(), name="no_setup", dir=self.data)
            # The diagnostic can appear before the runner reports terminal failure.
            deadline = time.monotonic() + 15
            while self.client.status()["active"] is not None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.03)
            status = run.status()
            self.assertEqual(status["state"], "FAILED")
            self.assertTrue(status["lease_released"])
            self.assertNotIn("SET source", run.log())

    def test_loss_of_supervision_skips_teardown_and_orphans(self):
        class Slow(Measurement):
            def sequence(self, seq):
                seq.wait(5)
        run = self.client.submit(plan(Slow), name="lost", dir=self.data)
        deadline = time.monotonic() + 10
        while "WAIT 5" not in run.log():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.03)
        # Stop is the service-shutdown equivalent of EOF on the control connection.
        self.server.registry.close()
        self.assertEqual(run.wait(timeout=20)["state"], "FAILED")
        self.assertNotIn("SET source.output", run.log())
        with InstrumentServerClient(self.instrument_socket) as client:
            self.assertTrue(client.status()["run"]["orphaned"])

    def test_terminal_status_write_failure_disables_admission(self):
        from auspex_v2.experiment import service
        original = service._atomic_json
        def fail_terminal(path, value):
            if value["state"] == "SUCCEEDED":
                raise OSError("disk unavailable")
            return original(path, value)
        with patch.object(service, "_atomic_json", fail_terminal):
            run = self.client.submit(plan(), name="status", dir=self.data)
            with self.assertRaisesRegex(RuntimeError, "Durable status is unresolved"):
                run.wait(timeout=20)
            self.assertIn("disk unavailable", self.client.status()["degraded"])
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                self.client.submit(plan(), name="next", dir=self.data)
        self.assertEqual(len(list(self.data.iterdir())), 1)
