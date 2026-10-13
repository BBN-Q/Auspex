"""Local IPC integration tests using only the simulated instrument."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from auspex_v2.instruments.client import CoordinatorClient, RemoteError
from auspex_v2.instruments.coordinator import PROTOCOL_VERSION


class CoordinatorTest(unittest.TestCase):
    """Exercise coordinator ownership and recovery over a real local socket."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.socket = Path(self.tmp.name) / "instrument.sock"
        self.server = None
        self.start_server()

    def start_server(self, *, cli=False, quiet=False, capture=False):
        """Launch the service or operator CLI and wait for its handshake."""
        env = dict(os.environ)
        source_dir = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")
        entrypoint = ([str(Path(__file__).resolve().parents[1] / "utils" / "instrument_coordinator.py")]
                      if cli else ["-m", "auspex_v2.instruments.coordinator"])
        argv = [sys.executable, "-B", *entrypoint, "--socket", str(self.socket)]
        if quiet:
            argv.append("--quiet")
        self.server = subprocess.Popen(argv, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, env=env)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with CoordinatorClient(self.socket) as c:
                    c.status()
                return
            except (OSError, ConnectionError):
                if self.server.poll() is not None:
                    self.fail(f"Coordinator exited: {self.server.communicate()[1].decode()}")
                time.sleep(0.02)
        self.fail("Coordinator did not start")

    def stop_server(self):
        """Terminate the service and assert that it shuts down cleanly."""
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                output, errors = self.server.communicate(timeout=7)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.communicate()
                self.fail("Coordinator did not terminate")
            self.assertEqual(self.server.returncode, 0, errors.decode())
            self.server = None
            return output or b"", errors
        self.server = None
        return b"", b""

    def tearDown(self):
        self.stop_server()
        self.tmp.cleanup()

    def register(self, client, name="src", resource="SIM::SRC"):
        return client.register(name, "simulated_source", resource)

    def test_cli_logs_commands_and_transitions_unless_quiet(self):
        """Show a splash and operator events by default but emit nothing in quiet mode."""
        self.stop_server()
        self.start_server(cli=True, capture=True)
        with CoordinatorClient(self.socket) as c:
            self.register(c)
            with c.live("src") as source:
                source.set("frequency", 5e9)
                self.assertEqual(source.get("frequency"), 5e9)
            with c.begin_run("observed-run"):
                pass
            c.terminate("src", "test reset")
            c.recover("src")
            c.terminate("src", "test reset")
            c.recover("src", "Operator checked source")
        output, _ = self.stop_server()
        text = output.decode()
        self.assertIn("INSTRUMENT COORDINATOR", text)
        self.assertIn("UNREGISTERED -> IDLE", text)
        self.assertIn("SET frequency value=5000000000.0", text)
        self.assertIn("GET frequency -> 5000000000.0", text)
        self.assertIn("UNLOCKED -> LOCKED", text)
        self.assertIn("LOCKED -> UNLOCKED", text)
        self.assertIn("src: TERMINATED -> IDLE\n", text)
        self.assertIn("src: TERMINATED -> IDLE (Operator checked source)", text)

        self.start_server(cli=True, quiet=True, capture=True)
        with CoordinatorClient(self.socket) as c:
            c.recover("src")
            with c.live("src") as source:
                source.set("frequency", 6e9)
        output, _ = self.stop_server()
        self.assertEqual(output, b"")

    def test_vna_inline_arrays_and_size_limit(self):
        """Transfer complex VNA traces without desynchronizing subsequent requests."""
        with CoordinatorClient(self.socket) as c:
            c.register("vna", "simulated_vna", "SIM::VNA")
            with c.live("vna") as vna:
                vna.set("frequency_center", 6.1e9)
                frequencies, values = vna.call("get_trace", measurement="S21")
                self.assertEqual(frequencies.shape, (2000,))
                self.assertEqual(values.shape, (2000,))
                self.assertEqual(frequencies.dtype, np.float64)
                self.assertEqual(values.dtype, np.complex64)
                self.assertAlmostEqual(frequencies[0], 5.6e9)
                self.assertEqual(vna.get("num_points"), 2000)
                second_freqs, second_values = vna.call("get_trace")
                np.testing.assert_array_equal(second_freqs, frequencies)
                np.testing.assert_array_equal(second_values, values)
                vna.set("num_points", 1_000_000)
                with self.assertRaisesRegex(RemoteError, "Command executed but result unavailable"):
                    vna.call("get_trace")
                self.assertEqual(c.status()["instruments"]["vna"]["state"], "IDLE")
                vna.set("num_points", 2000)
                self.assertEqual(vna.call("get_trace")[1].shape, (2000,))

    def test_disconnect_mid_bulk_result_releases_live_session(self):
        """Discard an abandoned array reply without retaining live ownership."""
        with CoordinatorClient(self.socket) as observer:
            observer.register("vna", "simulated_vna", "SIM::VNA")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(str(self.socket))
            reader = sock.makefile("rb")

            def request(op, **kwargs):
                sock.sendall(json.dumps({"op": op, **kwargs}).encode() + b"\n")
                return json.loads(reader.readline())

            try:
                request("HELLO", version=PROTOCOL_VERSION)
                response = request("OPEN_LIVE", name="vna")
                token = response["result"]["items"]["session"]["value"]
                response = request("CALL", name="vna", session=token, member="get_trace", args=[], kwargs={})
                self.assertEqual(len(response["arrays"]), 2)
                # Close before reading the array body; server must release the live lease.
            finally:
                sock.close()
                reader.close()
            for _ in range(100):
                if observer.status()["instruments"]["vna"]["live_sessions"] == 0:
                    break
                time.sleep(0.01)
            self.assertEqual(observer.status()["instruments"]["vna"]["live_sessions"], 0)
            with observer.live("vna") as vna:
                self.assertEqual(vna.get("num_points"), 2000)

    def test_live_run_lockout_and_commands(self):
        """Verify live access blocks atomic run lockout and run access locks all devices."""
        with CoordinatorClient(self.socket) as client, CoordinatorClient(self.socket) as other:
            self.register(client)
            self.register(client, "spare", "SIM::SPARE")
            with client.live("src") as source:
                self.assertEqual(source.get("frequency"), 1e9)
                source.set("frequency", 5e9)
                self.assertEqual(source.get("frequency"), 5e9)
                self.assertEqual(source.call("identify"), "SimulatedSource:SIM::SRC")
                with self.assertRaisesRegex(RemoteError, "already has a live session"):
                    other.live("src")
                with self.assertRaisesRegex(RemoteError, "close live sessions on src"):
                    other.begin_run("experiment-42")
                self.assertIsNone(other.status()["run"])  # no partial lockout
                with self.assertRaisesRegex(RemoteError, "Not a registered operation"):
                    source.call("disconnect")
                self.assertEqual(other.status()["instruments"]["src"]["state"], "IDLE")
            with other.begin_run("experiment-42") as run:
                self.assertEqual(other.status()["run"]["run_id"], "experiment-42")
                self.assertTrue(other.status()["instruments"]["spare"]["locked"])
                with self.assertRaisesRegex(RemoteError, "locks all registered instruments"):
                    client.live("spare")
                run.set("frequency", 6e9, name="src")
                self.assertEqual(run.get("frequency", name="src"), 6e9)
                with self.assertRaises(RemoteError):
                    client.register("new", "simulated_source", "SIM::NEW")
            self.assertIsNone(client.status()["run"])
            with client.live("spare") as spare:
                self.assertEqual(spare.get("frequency"), 1e9)

    def test_disconnect_releases_live_but_orphans_run(self):
        """Distinguish transient live disconnects from orphaned run ownership."""
        with CoordinatorClient(self.socket) as owner, CoordinatorClient(self.socket) as observer:
            self.register(owner)
            live = owner.live("src")
            owner.close()  # simulate notebook disappearance
            for _ in range(100):
                if observer.status()["instruments"]["src"]["live_sessions"] == 0:
                    break
                time.sleep(0.01)
            self.assertEqual(observer.status()["instruments"]["src"]["live_sessions"], 0)
            with observer.live("src") as new_live:
                self.assertEqual(new_live.get("frequency"), 1e9)
            run_owner = CoordinatorClient(self.socket)
            run_owner.begin_run("lost-run")
            run_owner.close()
            for _ in range(100):
                if observer.status()["run"]["orphaned"]:
                    break
                time.sleep(0.01)
            self.assertTrue(observer.status()["run"]["orphaned"])
            self.assertEqual(observer.status()["instruments"]["src"]["state"], "UNKNOWN")
            with self.assertRaises(RemoteError):
                observer.live("src")
            observer.reconcile_run("Operator examined interrupted run")
            with self.assertRaisesRegex(RemoteError, "recover instruments src"):
                observer.begin_run("next")
            observer.recover("src", "Operator checked simulated source")
            with observer.begin_run("next"):
                pass

    def test_termination_is_latched_and_recovery_is_explicit(self):
        """Require a reason for termination but not for explicit recovery."""
        with CoordinatorClient(self.socket) as c:
            self.register(c)
            with c.live("src"):
                pass
            with self.assertRaisesRegex(RemoteError, "reason required"):
                c.terminate("src", "")
            c.terminate("src", "Operator found unresponsive worker")
            self.assertEqual(c.status()["instruments"]["src"]["state"], "TERMINATED")
            with self.assertRaisesRegex(RemoteError, "explicit recovery"):
                c.live("src")
            with self.assertRaisesRegex(RemoteError, "Recovery reason must be a string"):
                c.request("RECOVER", name="src", reason=42)
            self.assertEqual(c.status()["instruments"]["src"]["state"], "TERMINATED")
            c.recover("src")
            self.assertEqual(c.status()["instruments"]["src"]["state"], "IDLE")
            c.terminate("src", "Operator requested a second check")
            c.recover("src", "Operator inspected simulated device")
            self.assertEqual(c.status()["instruments"]["src"]["state"], "IDLE")

    def test_unexpected_worker_death_latches_terminated(self):
        """Verify an unexpected subprocess exit quarantines its instrument."""
        with CoordinatorClient(self.socket) as c:
            self.register(c)
            with c.live("src") as source:
                with self.assertRaisesRegex(RemoteError, "Worker exited"):
                    source.call("crash_for_test")
            self.assertEqual(c.status()["instruments"]["src"]["state"], "TERMINATED")
            c.recover("src", "Operator inspected device after process crash")
            with c.live("src") as source:
                self.assertEqual(source.get("frequency"), 1e9)

    def test_restart_requires_recovery_and_duplicate_physical_device_fails(self):
        """Preserve inventory across restarts without trusting old worker state."""
        with CoordinatorClient(self.socket) as c:
            self.register(c)
            with self.assertRaisesRegex(RemoteError, "Physical device already registered"):
                self.register(c, "duplicate", "SIM::SRC")
        self.stop_server()
        self.start_server()
        with CoordinatorClient(self.socket) as c:
            self.assertEqual(c.status()["instruments"]["src"]["state"], "UNKNOWN")
            with self.assertRaises(RemoteError):
                c.begin_run("new")
            c.recover("src", "Operator inspected device after restart")
            c.deregister("src")
            self.assertEqual(c.status()["instruments"], {})


if __name__ == "__main__":
    unittest.main()
