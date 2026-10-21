"""Launcher integration tests use empty services/simulators, never hardware."""
import json
import os
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

    def test_unexpected_service_exit_stops_other_service(self):
        with InstrumentServerClient(self.instrument) as client:
            client.request("SHUTDOWN")
        self.process.wait(timeout=35)
        self.assertEqual(self.process.returncode, 1)
        self.assertIn("exited unexpectedly", self.text())
        self.assertFalse(self.experiment.exists())
