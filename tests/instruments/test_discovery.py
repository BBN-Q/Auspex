"""Discovery over real local IPC with a small, hardware-free VISA backend."""

import contextlib
import io
import os
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from auspex_v2.instruments.server import InstrumentServerClient, RemoteError
from auspex_v2.instruments.server import discovery
from auspex_v2.instruments.server.service import Server


VNA = "TCPIP0::192.0.2.1::hpib7,16::INSTR"
SERIAL = "ASRL1::INSTR"
UNKNOWN = "USB0::unknown::INSTR"
PROLOGIX = "PRLGX-ASRL/dev/ttyUSB0::1::INSTR"
DENIED = "USB0::denied::INSTR"


class VisaDevice:
    """Record only the probe traffic, optionally blocking it for ownership tests."""

    def __init__(self, manager, resource):
        self.manager = manager
        self.resource = resource
        self.closed = False

    def query(self, command):
        self.manager.queries.append((self.resource, command))
        if self.manager.entered is not None:
            self.manager.entered.set()
            if not self.manager.release.wait(5):
                raise TimeoutError("test probe stalled")
        if self.resource == UNKNOWN:
            raise TimeoutError("no response")
        return "Agilent Technologies,N5230A,serial,firmware\n"

    def close(self):
        self.closed = True


class VisaManager:
    """Share fake resources across short-lived resource-manager sessions."""

    def __init__(self):
        self.devices = []
        self.queries = []
        self.close_count = 0
        self.entered = None
        self.release = threading.Event()
        self.resources = (VNA, SERIAL, UNKNOWN)
        self.denied = set()
        self.opened = []

    def list_resources(self, expression):
        assert expression == "?*"
        return self.resources

    def open_resource(self, resource, **kwargs):
        assert kwargs == {"open_timeout": 1000}
        self.opened.append(resource)
        if resource in self.denied:
            raise PermissionError(13, "Permission denied", resource)
        device = VisaDevice(self, resource)
        self.devices.append(device)
        return device

    def close(self):
        self.close_count += 1


class DiscoveryTest(unittest.TestCase):
    """Verify probe isolation, closure, driver suggestions, and lease exclusion."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.home = patch.dict(os.environ, {"AUSPEX_HOME": self.tmp.name})
        self.home.start()
        self.manager = VisaManager()
        self.backend = patch.object(discovery, "resource_manager", return_value=self.manager)
        self.backend.start()
        self.server = Server(Path(self.tmp.name) / "instrument.sock", quiet=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = InstrumentServerClient(self.server.path)

    def tearDown(self):
        self.manager.release.set()
        self.client.close()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.backend.stop()
        self.home.stop()
        self.tmp.cleanup()

    def test_enumerate_identify_register_and_skip_registered(self):
        """Enumeration is inert; registration stays disconnected and aliases cannot be probed."""
        candidates = self.client.enumerate()
        self.assertEqual(len(candidates), 3)
        self.assertEqual(self.manager.queries, [])
        self.assertTrue(next(c for c in candidates if c["resource"] == SERIAL)["serial"])
        result = self.client.identify(VNA)
        self.assertEqual(result["drivers"], ["AgilentN5230A"])
        self.assertEqual(self.manager.queries, [(VNA, "*IDN?")])
        device = self.manager.devices[-1]
        self.assertTrue(device.closed)
        self.assertEqual(device.timeout, 1000)
        self.assertEqual(device.read_termination, "\n")
        self.client.register("vna", "AgilentN5230A", "192.0.2.1")
        self.assertEqual(self.client.status()["instruments"]["vna"]["state"], "DISCONNECTED")
        candidates = self.client.enumerate()
        self.assertTrue(next(c for c in candidates if c["resource"] == VNA)["registered"])
        with self.assertRaisesRegex(RemoteError, "already registered"):
            self.client.identify(VNA)
        with self.assertRaisesRegex(RemoteError, "already registered"):
            self.client.register("duplicate", "AgilentN5230A", VNA)

    def test_failed_identity_closes_resources_and_allows_retry(self):
        """A nonresponsive candidate is not classified as absent and leaks no connection."""
        result = self.client.identify(UNKNOWN)
        self.assertEqual(result["error"], "no response")
        self.assertEqual(result["drivers"], [])
        self.assertTrue(self.manager.devices[-1].closed)
        self.assertEqual(self.manager.close_count, 1)
        self.assertEqual(self.client.identify(SERIAL)["drivers"], ["AgilentN5230A"])
        with self.assertRaisesRegex(RemoteError, "Only interface visa"):
            self.client.enumerate("other")

    def test_probe_reservation_keeps_status_responsive_and_excludes_leases(self):
        """No client can claim the domain or register while temporary probe I/O is active."""
        self.manager.entered = threading.Event()
        results = []

        def probe():
            with InstrumentServerClient(self.server.path) as client:
                results.append(client.identify(VNA))

        thread = threading.Thread(target=probe)
        thread.start()
        try:
            self.assertTrue(self.manager.entered.wait(3))
            self.assertIsNone(self.client.status()["run"])
            with self.assertRaises(RemoteError):
                self.client.begin_run("blocked")
            with self.assertRaises(RemoteError):
                self.client.register("vna", "AgilentN5230A", VNA)
            with self.assertRaises(RemoteError):
                self.client.enumerate()
        finally:
            self.manager.release.set()
            thread.join(5)
        self.assertEqual(results[0]["drivers"], ["AgilentN5230A"])
        with self.client.begin_run("exclusive"):
            with self.assertRaises(RemoteError):
                self.client.identify(VNA)
            with self.assertRaises(RemoteError):
                self.client.enumerate()

    def test_operator_workflow_keeps_letters_separate_and_skips_serial(self):
        """The shared dispatcher enumerates, probes, and explicitly registers without connecting."""
        from utils.src.operator_commands import OperatorCommands

        commands = OperatorCommands(self.server.path, None)
        with self.assertRaisesRegex(ValueError, "enumerate visa first"):
            commands.execute(["identify", "a"])
        output = commands.execute(commands.parse("enumerate visa")).text
        self.assertNotIn("a. ASRL1::INSTR", output)
        self.assertIn("1 serial", output)
        self.assertIn("unknown", output)
        self.assertTrue(commands.confirmation(commands.parse("identify all")))
        self.assertFalse(commands.confirmation(commands.parse("identify b")))
        output = commands.execute(["identify", "all"]).text
        self.assertEqual(self.manager.queries, [(VNA, "*IDN?"), (UNKNOWN, "*IDN?")])
        self.assertIn("AgilentN5230A", output)
        self.assertIn("no response", output)
        self.assertIn(VNA, commands.registration_prompt(["register", "b", "AgilentE8363C"]))
        commands.execute(["register", "b", "AgilentE8363C", "chosen"])
        self.assertEqual(self.client.status()["instruments"]["chosen"]["driver"], "AgilentE8363C")
        self.assertFalse(self.client.status()["instruments"]["chosen"]["worker_running"])
        with self.assertRaisesRegex(ValueError, "list first"):
            commands.execute(["connect", "1"])
        commands.execute(["identify", "a"])
        self.assertEqual(self.manager.queries[-1], (SERIAL, "*IDN?"))
        with self.assertRaisesRegex(ValueError, "already registered"):
            commands.execute(["register", "b", "AgilentN5230A", "duplicate"])
        commands.execute(["enumerate", "visa"])
        self.assertEqual(list(commands.candidates), ["a", "b", "c"])
        self.assertTrue(commands.candidates["b"]["registered"])
        self.assertIsNone(commands.candidates["a"]["identity"])

    def test_candidate_letters_continue_past_z(self):
        """Large candidate lists never reuse numeric inventory identifiers."""
        from utils.src.operator_commands import OperatorCommands

        self.manager.resources = tuple(f"USB0::{index:02}::INSTR" for index in range(28))
        commands = OperatorCommands(self.server.path, None)
        commands.execute(["enumerate", "visa"])
        self.assertEqual(list(commands.candidates)[-3:], ["z", "aa", "ab"])
        self.assertEqual(commands.candidates["aa"]["resource"], "USB0::26::INSTR")
        self.manager.resources = ()
        self.assertIn("No VISA candidates", commands.execute(["enumerate", "visa"]).text)
        with self.assertRaisesRegex(ValueError, "enumerate visa first"):
            commands.registration_prompt(["register", "aa", "AgilentN5230A"])

    def test_plain_and_standalone_prompts_confirm_and_collect_names(self):
        """Both terminal entry points preserve driver case and survive rejected registrations."""
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "utils"))
        try:
            from utils import launch_auspex, instrument_server
            lines = ["enumerate visa", "identify all", "n", "register b NotADriver", "bad",
                     "identify all", "y", "register b AgilentN5230A", "bench_vna"]
            inputs = iter(lines)

            def read_input(prompt):
                try:
                    return next(inputs)
                except StopIteration:
                    raise EOFError from None

            output = io.StringIO()
            with patch("builtins.input", side_effect=read_input), contextlib.redirect_stdout(output):
                launch_auspex.plain_operator_prompt(self.server.path, None)
            self.assertIn("Command failed:", output.getvalue())
            self.assertIn("Registered bench_vna as DISCONNECTED", output.getvalue())
            self.assertEqual(len(self.manager.queries), 2)
            self.client.deregister("bench_vna")
            self.manager.queries.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                instrument_server.terminal_commands(self.server.path, iter(lines))
            self.assertEqual(len(self.manager.queries), 2)
            self.assertEqual(self.client.status()["instruments"]["bench_vna"]["state"], "DISCONNECTED")
        finally:
            sys.path.remove(str(root / "utils"))

    def test_display_options_and_bulk_probe_exclude_serial_and_prologix(self):
        """Serial/Prologix are independent display categories, never implicit bulk probe targets."""
        from utils.src.operator_commands import OperatorCommands

        self.manager.resources = (SERIAL, PROLOGIX, VNA)
        commands = OperatorCommands(self.server.path, None)
        output = commands.execute(commands.parse("enumerate visa")).text
        self.assertNotIn(SERIAL, output)
        self.assertNotIn(PROLOGIX, output)
        self.assertIn(VNA, output)
        self.assertIn("1 serial, 1 Prologix", output)
        self.assertEqual(self.manager.opened, [])
        output = commands.execute(commands.parse("enumerate visa --serial")).text
        self.assertIn(SERIAL, output)
        self.assertNotIn(PROLOGIX, output)
        output = commands.execute(commands.parse("enumerate visa --prologix")).text
        self.assertIn(PROLOGIX, output)
        self.assertNotIn(SERIAL, output)
        output = commands.execute(commands.parse("enumerate visa --serial --prologix --show-permission-denied")).text
        self.assertIn(PROLOGIX, output)
        self.assertIn(SERIAL, output)
        commands.execute(["identify", "all"])
        self.assertEqual(self.manager.queries, [(VNA, "*IDN?")])
        letter = next(k for k, c in commands.candidates.items() if c["resource"] == PROLOGIX)
        commands.execute(["identify", letter])
        self.assertEqual(self.manager.queries[-1], (PROLOGIX, "*IDN?"))
        for line in ("enumerate visa --other", "enumerate visa serial", "enumerate other --serial"):
            with self.assertRaises(ValueError):
                commands.parse(line)
        self.assertIn("--show-permission-denied", commands.execute(["help"]).text)

    def test_permission_denial_is_retained_until_retry_and_does_not_renumber(self):
        """Denied rows hide after a probe, survive enumeration as known denied, and can be retried."""
        from utils.src.operator_commands import OperatorCommands

        self.manager.resources = (DENIED, VNA)
        self.manager.denied.add(DENIED)
        commands = OperatorCommands(self.server.path, None)
        commands.execute(["enumerate", "visa"])
        letter = next(k for k, c in commands.candidates.items() if c["resource"] == DENIED)
        labels = list(commands.candidates)
        commands.execute(["identify", letter])
        self.assertTrue(commands.candidates[letter]["permission_denied"])
        self.assertEqual(list(commands.candidates), labels)
        opened = list(self.manager.opened)
        output = commands.execute(["enumerate", "visa"]).text
        self.assertNotIn(DENIED, output)
        self.assertIn("1 permission denied", output)
        self.assertEqual(self.manager.opened, opened)
        commands.execute(["identify", "all"])
        self.assertEqual(self.manager.opened, opened + [VNA])
        output = commands.execute(commands.parse("enumerate visa --show-permission-denied")).text
        self.assertIn(DENIED, output)
        self.manager.denied.clear()
        commands.execute(["enumerate", "visa"])
        output = commands.execute(["identify", letter]).text  # Explicit retry works even while hidden.
        self.assertIn(DENIED, output)
        self.assertFalse(commands.candidates[letter]["permission_denied"])
        candidate = next(c for c in self.client.enumerate() if c["resource"] == DENIED)
        self.assertFalse(candidate["permission_denied"])

    def test_server_whitelist_reloads_and_overrides_visibility_not_probe_eligibility(self):
        """A server-local file overrides all filters without adding resources or probing them."""
        from utils.src.operator_commands import OperatorCommands

        self.manager.resources = (SERIAL, PROLOGIX, DENIED, VNA)
        self.manager.denied.add(DENIED)
        self.client.identify(DENIED)
        path = Path(self.tmp.name) / "instrument_whitelist.txt"
        path.write_text(f"# Display overrides\n\n{SERIAL}\n{PROLOGIX}\n{DENIED}\ntcpip::192.0.2.1::hpib7,16::instr\n"
                        "TCPIP0::not-enumerated::INSTR\n", encoding="utf-8")
        commands = OperatorCommands(self.server.path, None)
        output = commands.execute(["enumerate", "visa"]).text
        self.assertIn(SERIAL, output)
        self.assertIn(PROLOGIX, output)
        self.assertIn(DENIED, output)
        self.assertNotIn("not-enumerated", output)
        self.assertIn("4 matching resources shown", output)
        self.assertEqual(self.manager.opened, [DENIED])
        commands.execute(["identify", "all"])
        self.assertEqual(self.manager.opened, [DENIED, VNA, DENIED])
        path.write_text("# now empty\n", encoding="utf-8")
        output = commands.execute(["enumerate", "visa"]).text
        self.assertNotIn(SERIAL, output)
        self.assertNotIn(PROLOGIX, output)
        self.assertNotIn(DENIED, output)
        self.assertIn("0 matching resources shown", output)

    def test_whitelist_missing_unreadable_and_home_fallback(self):
        """Missing whitelist is normal; unreadable content warns clients even without any candidates."""
        from utils.src.operator_commands import OperatorCommands

        snapshot = self.client.enumerate(include_metadata=True)
        self.assertIsNone(snapshot["whitelist_warning"])
        path = Path(self.tmp.name) / "instrument_whitelist.txt"
        path.write_text(SERIAL + "\n", encoding="utf-8")
        with patch.dict(os.environ, {"AUSPEX_HOME": "", "HOME": self.tmp.name}):
            snapshot = self.client.enumerate(include_metadata=True)
        self.assertEqual(snapshot["whitelist_path"], str(path))
        self.assertTrue(next(c for c in snapshot["candidates"] if c["resource"] == SERIAL)["whitelisted"])
        path.unlink()
        path.mkdir()  # Reliable unreadability even when tests run as root.
        self.manager.resources = ()
        output = OperatorCommands(self.server.path, None).execute(["enumerate", "visa"]).text
        self.assertIn("Warning: Cannot read whitelist", output)
        self.assertIn(str(path), output)
        self.assertIn("No VISA candidates", output)

    def test_access_error_classification_does_not_hide_other_failures(self):
        """Only attributable access-denial errors trigger filtering, including wrapped backend errors."""
        wrapped = RuntimeError("serial port open failed")
        wrapped.__cause__ = PermissionError(13, "access failed")
        for error in (wrapped, OSError(1, "operation not permitted"),
                      RuntimeError("could not open port: Permission denied"),
                      RuntimeError("VI_ERROR_NPERMISSION")):
            self.assertTrue(discovery.permission_denied(error))
        for error in (TimeoutError("no response"), OSError("device not found"), RuntimeError("resource locked")):
            self.assertFalse(discovery.permission_denied(error))

    def test_matching_is_literal_and_reports_multiple_suggestions(self):
        """A driver hint is not a choice, and unknown identities remain registerable."""
        from auspex_v2.instruments.drivers.registry import DRIVERS
        driver = DRIVERS["AgilentE8363C"]
        with patch.object(driver, "IDN_STRING", "N5230A"):
            self.assertEqual(set(self.client.identify(VNA)["drivers"]),
                             {"AgilentN5230A", "AgilentE8363C"})
        with patch.object(DRIVERS["AgilentN5230A"], "IDN_STRING", "n5230a"):
            self.assertEqual(self.client.identify(VNA)["drivers"], [])
        self.client.register("explicit", "AgilentE8363C", UNKNOWN)
        self.assertEqual(self.client.status()["instruments"]["explicit"]["state"], "DISCONNECTED")
