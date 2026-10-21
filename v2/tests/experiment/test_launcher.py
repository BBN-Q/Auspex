"""Launcher integration tests use empty services/simulators, never hardware."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from auspex_v2.experiment.client import ExperimentServerClient
from auspex_v2.instruments.server import InstrumentServerClient


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.instrument = root / "instrument.sock"
        self.experiment = root / "experiment.sock"
        v2 = Path(__file__).resolve().parents[2]
        self.command = [sys.executable, "-B", str(v2 / "utils" / "launch_auspex.py"),
                        "--instrument-socket", str(self.instrument),
                        "--experiment-socket", str(self.experiment)]
        self.env = dict(os.environ, PYTHONPATH=str(v2 / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        self.output = (root / "terminal.log").open("w+")
        self.addCleanup(self.output.close)
        self.process = subprocess.Popen(self.command, env=self.env, stdout=self.output,
                                        stderr=subprocess.STDOUT)
        self.addCleanup(self.stop)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.assertIsNone(self.process.poll(), self.text())
            if "[Auspex] Ready." in self.text():
                return
            time.sleep(.05)
        self.fail("Launcher failed to become ready: " + self.text())

    def text(self):
        return Path(self.output.name).read_text()

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=35)

    def test_ready_duplicate_refusal_and_preserved_inventory(self):
        with InstrumentServerClient(self.instrument) as client:
            client.register("source", "SimulatedSource", "SIM::SRC")
        self.assertIsNone(ExperimentServerClient(self.experiment).status()["active"])
        duplicate = subprocess.run(self.command, env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(duplicate.returncode, 1)
        self.assertIn("leaving it untouched", duplicate.stdout)
        self.assertIsNone(self.process.poll())
        self.stop()
        self.assertEqual(self.process.returncode, 0)
        self.assertFalse(self.instrument.exists())
        self.assertFalse(self.experiment.exists())
        saved = json.loads(Path(str(self.instrument) + ".state.json").read_text())
        self.assertIn("source", saved["instruments"])
        self.assertNotIn("EXPERIMENT SERVER", self.text())
        self.assertNotIn("INSTRUMENT SERVER", self.text())
        self.assertIn("EXPERIMENT CONTROL SYSTEM", self.text())
        self.assertNotIn("\033[", self.text())
        self.assertIn("Stopping ExperimentServer", self.text())
        self.assertLess(self.text().index("Stopping ExperimentServer"), self.text().index("Stopping InstrumentServer"))

    def start_run(self, *, blocking=False):
        from .test_experiment_server import Measurement, plan
        class Slow(Measurement):
            def sequence(self, seq):
                if blocking:
                    self.source.wait_for_test(4)
                else:
                    seq.wait(10)
                seq.publish(self.signal, data=1.0)
        with InstrumentServerClient(self.instrument) as client:
            client.register("source", "SimulatedSource", "SIM::SRC")
        run = ExperimentServerClient(self.experiment).submit(plan(Slow), name="slow", dir=self.tmp.name)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            started = ("CALL source.wait_for_test" in run.log() if blocking
                       else run.status()["state"] == "RUNNING")
            if started:
                return run
            time.sleep(.05)
        self.fail("Run did not start")

    def test_first_interrupt_cancels_and_preserves_teardown(self):
        run = self.start_run()
        self.process.send_signal(signal.SIGINT)
        self.process.wait(timeout=25)
        self.assertEqual(self.process.returncode, 0, self.text())
        result = json.loads((run.path / "status.json").read_text())
        self.assertEqual(result["state"], "CANCELLED", self.text())
        self.assertTrue(result["lease_released"])
        self.assertEqual(result["progress"]["completed"], 0)
        self.assertIn("Requesting cooperative cancellation", self.text())

    def test_second_interrupt_stops_supervision(self):
        self.start_run(blocking=True)
        self.process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + 5
        while "Waiting for teardown/finalization" not in self.text():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.02)
        self.process.send_signal(signal.SIGINT)
        self.process.wait(timeout=30)
        self.assertIn("Second interrupt", self.text())
        self.assertIn("leases may need reconciliation", self.text())

    def test_unexpected_service_exit_stops_other_service(self):
        with InstrumentServerClient(self.instrument) as client:
            client.request("SHUTDOWN")
        self.process.wait(timeout=35)
        self.assertEqual(self.process.returncode, 1)
        self.assertIn("exited unexpectedly", self.text())
        self.assertFalse(self.experiment.exists())
