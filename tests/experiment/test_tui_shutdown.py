"""Headless Textual shutdown integration with real services and simulated hardware."""

import importlib.util
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
from .test_experiment_server import Measurement, plan
from utils.src.service_output import ServiceOutput


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tui_shutdown_launcher", ROOT / "utils" / "launch_auspex.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@unittest.skipUnless(importlib.util.find_spec("textual"), "install auspex-v2[tui] to test the console")
class TuiShutdownTest(unittest.IsolatedAsyncioTestCase):
    """Keep the real console visible until cooperative or forced service shutdown finishes."""

    def setUp(self):
        """Start the same independent services and output pipes used by the launcher."""
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.instrument = Path(self.tmp.name) / "instrument.sock"
        self.experiment = Path(self.tmp.name) / "experiment.sock"
        self.children = []
        self.output = ServiceOutput()
        self.addCleanup(self.stop_children)
        env = dict(os.environ, AUSPEX_HOME=self.tmp.name,
                   PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        commands = [
            ("InstrumentServer", [str(ROOT / "utils" / "instrument_server.py"), "--no-splash",
                                  "--socket", str(self.instrument)], self.instrument,
             {"op": "HELLO", "version": launcher.INSTRUMENT_VERSION}),
            ("ExperimentServer", [str(ROOT / "utils" / "experiment_server.py"), "--no-splash",
                                  "--socket", str(self.experiment), "--instrument-socket", str(self.instrument)],
             self.experiment, {"op": "server_status", "version": launcher.EXPERIMENT_VERSION}),
        ]
        for name, args, path, hello in commands:
            process = subprocess.Popen([sys.executable, "-B", "-u", *args], env=env,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.children.append((process, name))
            self.output.attach(process, name)
            launcher.wait_ready(process, path, hello)

        sys.path.insert(0, str(ROOT / "utils"))
        try:
            from src.auspex_tui import OperatorConsole
        finally:
            sys.path.remove(str(ROOT / "utils"))

        class RecordingConsole(OperatorConsole):
            """Retain rendered messages so assertions do not depend on RichLog internals."""

            def _log(self, message):
                self.messages.append(message)
                super()._log(message)

        self.app = RecordingConsole(self.instrument, self.experiment, self.children, output=self.output,
            shutdown=lambda stop_wait, report: launcher.shutdown_tui(
                self.experiment, self.children, stop_wait, report, output=self.output))
        self.app.messages = []

    def stop_children(self):
        for process, name in reversed(self.children):
            launcher.stop(process, name)
        self.output.finish()

    async def until(self, pilot, predicate):
        """Allow both the Textual loop and actual service processes to make progress."""
        deadline = time.monotonic() + 15
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Shutdown did not progress: " + "\n".join(self.app.messages))
            await pilot.pause(.05)

    def start_run(self, *, blocking=False):
        """Use an interruptible wait or an in-flight driver call to test both Ctrl+C paths."""
        class Slow(Measurement):
            """Leave ample time to exercise shutdown before a measurement can finish."""

            def sequence(self, seq):
                if blocking:
                    self.source.wait_for_test(4)
                else:
                    seq.wait(20)
                seq.publish(self.signal, data=1.0)

            def teardown(self, seq):
                seq.wait(.8)
                self.source.output = False

        with InstrumentServerClient(self.instrument) as client:
            client.register("source", "SimulatedSource", "SIM::SRC")
        run = ExperimentServerClient(self.experiment).submit(plan(Slow), name="slow", dir=self.tmp.name)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            started = ("CALL source.wait_for_test" in run.log() if blocking
                       else run.status()["progress"].get("phase") == "acquisition")
            if started:
                return run
            time.sleep(.02)
        self.fail("Run did not start")

    async def test_first_ctrl_c_keeps_console_and_logs_through_teardown_and_shutdown(self):
        """The first key cancels once, preserves teardown, drains logs, and only then exits."""
        from textual.widgets import Input

        run = self.start_run()
        async with self.app.run_test() as pilot:
            self.app.pending_registration = ["register", "a", "AgilentN5230A"]
            await pilot.press("ctrl+c")
            self.assertTrue(self.app.is_running)
            self.assertTrue(self.app.shutting_down)
            self.assertIsNone(self.app.pending_registration)
            self.assertTrue(self.app.query_one(Input).disabled)
            self.app._command_finished()  # A previously dispatched command must not restore input.
            self.assertTrue(self.app.query_one(Input).disabled)
            await self.until(pilot, lambda: not self.app.is_running)
        result = json.loads((run.path / "status.json").read_text())
        self.assertEqual(result["state"], "CANCELLED")
        self.assertTrue(result["lease_released"])
        text = "\n".join(self.app.messages)
        self.assertIn("does not just clear the prompt", text)
        self.assertIn("Waiting for teardown/finalization", text)
        self.assertIn("[ExperimentServer]", text)
        self.assertIn("CANCELLED", text)
        self.assertIn("Services stopped. Exiting console.", text)
        self.assertLess(text.index("Stopping ExperimentServer"), text.index("Stopping InstrumentServer"))
        self.assertTrue(all(process.poll() is not None for process, _ in self.children))
        self.assertFalse(self.instrument.exists())
        self.assertFalse(self.experiment.exists())
        self.assertEqual(self.app.return_value, 0)

    async def test_second_ctrl_c_skips_cooperative_wait_but_keeps_console_during_cleanup(self):
        """A second key requests forced supervision shutdown without closing the UI prematurely."""
        self.start_run(blocking=True)
        async with self.app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await self.until(pilot, lambda: any("Waiting for teardown/finalization" in m for m in self.app.messages))
            self.assertTrue(self.app.is_running)
            await pilot.press("ctrl+c")
            self.assertTrue(self.app.stop_wait.is_set())
            await self.until(pilot, lambda: not self.app.is_running)
        text = "\n".join(self.app.messages)
        self.assertIn("Second interrupt: skipping cooperative wait", text)
        self.assertIn("leases may need reconciliation", text)
        self.assertIn("Services stopped. Exiting console.", text)
        self.assertEqual(text.count("Requesting cooperative cancellation"), 1)
        self.assertTrue(all(process.poll() is not None for process, _ in self.children))
        self.assertEqual(self.app.return_value, 0)

    async def test_inherited_stdout_does_not_hold_console_open_after_services_stop(self):
        """The UI exits with a warning while a surviving descendant still owns a captured pipe."""
        gate = Path(self.tmp.name) / "release-descendant"
        ready = Path(self.tmp.name) / "descendant-ready"
        child = ("from pathlib import Path; import sys,time; "
                 "Path(sys.argv[2]).touch(); gate=Path(sys.argv[1]); "
                 "exec('while not gate.exists(): time.sleep(.01)'); "
                 "print('descendant released', flush=True)")
        parent = ("import subprocess,sys,time; "
                  "subprocess.Popen([sys.executable,'-u','-c',sys.argv[1],sys.argv[2],sys.argv[3]]); "
                  "time.sleep(30)")
        process = subprocess.Popen([sys.executable, "-u", "-c", parent, child, str(gate), str(ready)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.children.append((process, "pipe-holder"))
        self.output.attach(process, "pipe-holder")
        try:
            deadline = time.monotonic() + 5
            while not ready.exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            async with self.app.run_test() as pilot:
                await pilot.press("ctrl+c")
                await self.until(pilot, lambda: not self.app.is_running)
            text = "\n".join(self.app.messages)
            self.assertIn("pipe-holder: service exited, but output remains open", text)
            self.assertIn("Inspect hardware and leases", text)
            self.assertIn("Services stopped. Exiting console.", text)
            self.assertTrue(all(p.poll() is not None for p, _ in self.children))
            self.assertTrue(any(reader.is_alive() for _, reader in self.output.readers))
            self.assertEqual(self.app.return_value, 0)
        finally:
            gate.touch()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            for _, reader in self.output.readers:
                reader.join(5)
        self.assertFalse(any(reader.is_alive() for _, reader in self.output.readers))

    async def test_ctrl_c_without_active_run_still_stops_both_services(self):
        """Idle shutdown uses the same ordered cleanup without issuing a run cancellation."""
        async with self.app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await self.until(pilot, lambda: not self.app.is_running)
        text = "\n".join(self.app.messages)
        self.assertNotIn("Requesting cooperative cancellation", text)
        self.assertIn("Stopping ExperimentServer", text)
        self.assertIn("Stopping InstrumentServer", text)
        self.assertTrue(all(process.poll() is not None for process, _ in self.children))
        self.assertEqual(self.app.return_value, 0)
