"""Regression tests for startup pipe pressure, bounded capture and inherited stdout."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from utils.src.service_output import ServiceOutput


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("output_test_launcher", ROOT / "utils" / "launch_auspex.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class ServiceOutputTest(unittest.TestCase):
    """Use actual subprocess pipes rather than mocking their buffering or EOF behavior."""

    def test_flood_without_ui_keeps_newest_lines_and_reports_exact_overflow(self):
        """A producer can exceed pipe and pending-buffer capacity without needing a UI consumer."""
        process = subprocess.Popen([sys.executable, "-u", "-c",
            "import sys; [print(f'line-{i} ' + 'x'*120) for i in range(7000)]"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = ServiceOutput()
        output.attach(process, "flood")
        try:
            process.wait(timeout=5)  # Would block here if output capture depended on the UI.
            output.finish()
            lines = output.drain()
            self.assertEqual(process.returncode, 0)
            self.assertEqual(len(lines), 5001)  # Overflow notice plus the retained 5,000 lines.
            self.assertIn("2000 service lines dropped", lines[0])
            self.assertTrue(lines[1].startswith("line-2000 "))
            self.assertTrue(lines[-1].startswith("line-6999 "))
            self.assertEqual(output.drain(), [])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            output.finish()

    def test_chatty_real_service_becomes_ready_before_any_ui_exists(self):
        """Restoring enough inventory to fill a pipe must not block the readiness handshake."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            socket = Path(temp) / "instrument.sock"
            inventory = {f"source_{i}": {"driver": "SimulatedSource", "resource": f"SIM::{i}",
                         "physical": f"SIM::{i}", "state": "DISCONNECTED",
                         "command_timeout": None, "io_timeout": None} for i in range(1200)}
            Path(str(socket) + ".state.json").write_text(json.dumps({"instruments": inventory, "run": None}))
            process = subprocess.Popen([sys.executable, "-B", "-u", str(ROOT / "utils" / "instrument_server.py"),
                "--no-splash", "--socket", str(socket)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=dict(os.environ, AUSPEX_HOME=temp), start_new_session=True)
            output = ServiceOutput()
            output.attach(process, "InstrumentServer")
            try:
                launcher.wait_ready(process, socket, {"op": "HELLO", "version": launcher.INSTRUMENT_VERSION}, timeout=5)
                lines = output.drain()
                self.assertGreater(sum(len(line) for line in lines), 65536)
                self.assertTrue(any("on server restart" in line for line in lines))
            finally:
                launcher.stop(process, "InstrumentServer", report=lambda message: None)
                output.finish()
            self.assertFalse(socket.exists())

    def test_inherited_stdout_has_one_total_deadline_and_does_not_kill_descendants(self):
        """Two exited parents with pipe-holding children share one second, not one second each."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            gate = Path(temp) / "release"
            child = ("from pathlib import Path; import sys,time; "
                     "gate=Path(sys.argv[1]); "
                     "exec('while not gate.exists(): time.sleep(.01)'); "
                     "print('descendant released', flush=True)")
            parent = ("import subprocess,sys; "
                      "subprocess.Popen([sys.executable, '-u', '-c', sys.argv[1], sys.argv[2]]); "
                      "print('parent exited', flush=True)")
            output = ServiceOutput()
            processes = []
            try:
                for name in ("first", "second"):
                    process = subprocess.Popen([sys.executable, "-u", "-c", parent, child, str(gate)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    processes.append(process)
                    output.attach(process, name)
                for process in processes:
                    process.wait(timeout=5)
                start = time.monotonic()
                output.finish()
                self.assertLess(time.monotonic() - start, 1.7)
                lines = output.drain()
                self.assertEqual(sum("output remains open" in line for line in lines), 2)
                self.assertTrue(all(reader.is_alive() for _, reader in output.readers))
                self.assertFalse(gate.exists())
                start = time.monotonic()
                output.finish()  # A fallback cleanup must not wait another second or repeat warnings.
                self.assertLess(time.monotonic() - start, .2)
                self.assertEqual(output.drain(), [])
            finally:
                gate.touch()
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                for _, reader in output.readers:
                    reader.join(5)
            self.assertFalse(any(reader.is_alive() for _, reader in output.readers))
            self.assertEqual(output.drain().count("descendant released"), 2)
