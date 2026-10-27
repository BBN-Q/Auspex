"""Canonical endpoint identity and saved-inventory migration, without hardware I/O."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from auspex_v2.instruments.drivers.addresses import normalize_visa_address
from auspex_v2.instruments.drivers.registry import DRIVERS
from auspex_v2.instruments.server import InstrumentServerClient, RemoteError
from auspex_v2.instruments.server.service import Registry, Server
from auspex_v2.instruments.server import discovery
from auspex_v2.status import format_status
from .test_discovery import VisaManager
from .test_instruments import FakeTransport


ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def running_server(path):
    """Exercise the actual local protocol with an isolated, explicitly started server."""
    with Server(path, quiet=True) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with InstrumentServerClient(path) as client:
                yield client
        finally:
            server.shutdown()
            thread.join()


def definition(driver, resource, *, state="DISCONNECTED"):
    return {"driver": driver, "resource": resource, "physical": resource, "state": state,
            "command_timeout": None, "io_timeout": None}


class VisaAddressTest(unittest.TestCase):
    """Verify address equivalence against explicit examples, not a second copy of the parser."""

    def test_common_spellings_and_idempotence(self):
        """Normalize equivalent numeric/default fields while preserving case-sensitive address components."""
        examples = [
            ("tcpip::VNA.EXAMPLE::instr", "TCPIP0::vna.example::inst0::INSTR"),
            ("TCPIP00::vna.example::INST0::INSTR", "TCPIP0::vna.example::inst0::INSTR"),
            ("TCPIP1::HOST::05025::socket", "TCPIP1::host::5025::SOCKET"),
            ("TCPIP::HOST::DeviceA::instr", "TCPIP0::host::DeviceA::INSTR"),
            ("gpib::05::00::instr", "GPIB0::5::INSTR"),
            ("GPIB1::5::02::INSTR", "GPIB1::5::2::INSTR"),
            ("gpib::intfc", "GPIB0::INTFC"),
            ("usb::0x1234::0x00aB::SerialAbC::instr", "USB0::4660::171::SerialAbC::0::INSTR"),
            ("USB0::4660::171::SerialAbC::00::INSTR", "USB0::4660::171::SerialAbC::0::INSTR"),
            ("asrl001::instr", "ASRL1::INSTR"),
            ("asrl/dev/ttyUSB0::instr", "ASRL/dev/ttyUSB0::INSTR"),
            ("prlgx-asrl/dev/ttyUSB0::1::instr", "PRLGX-ASRL/dev/ttyUSB0::1::INSTR"),
            ("SIM::caseSensitive", "SIM::caseSensitive"),
        ]
        with patch.dict("sys.modules", {"pyvisa": None}):
            for address, expected in examples:
                with self.subTest(address=address):
                    self.assertEqual(normalize_visa_address(address), expected)
                    self.assertEqual(normalize_visa_address(expected), expected)
        self.assertNotEqual(normalize_visa_address("ASRL/dev/ttyUSB0::INSTR"),
                            normalize_visa_address("ASRL/dev/ttyusb0::INSTR"))

    def test_bare_ipv4_requires_a_driver_default(self):
        with self.assertRaisesRegex(ValueError, "driver-specific endpoint"):
            normalize_visa_address("192.0.2.5")
        self.assertEqual(normalize_visa_address("192.0.2.5", ipv4_device="5025::SOCKET"),
                         "TCPIP0::192.0.2.5::5025::SOCKET")
        self.assertEqual(normalize_visa_address("192.0.2.5", ipv4_device="hpib7,16::INSTR"),
                         "TCPIP0::192.0.2.5::hpib7,16::INSTR")

    def test_connection_and_ownership_use_exactly_the_same_endpoint(self):
        """Each admitted driver's existing IPv4 shorthand expands once for both I/O and identity."""
        for name, suffix in (("AgilentN5183A", "5025::SOCKET"),
                             ("AgilentE8363C", "hpib7,16::INSTR"),
                             ("AgilentN5230A", "hpib7,16::INSTR")):
            address = f"TCPIP0::192.0.2.5::{suffix}"
            with self.subTest(driver=name), patch("auspex_v2.instruments.drivers.visa.VisaTransport") as transport:
                transport.return_value = FakeTransport()
                driver = DRIVERS[name]("192.0.2.5")
                self.assertEqual(driver.physical_resource, address)
                driver.connect()
                transport.assert_called_once_with(address)
                driver.disconnect()
                self.assertEqual(DRIVERS[name](address).physical_resource, address)

    def test_discovery_rejects_registered_endpoints_and_duplicate_spellings(self):
        """The previously missed socket and hpib endpoints are registered, not fresh probe targets."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch.dict(os.environ, {"AUSPEX_HOME": temp}):
            manager = VisaManager()
            manager.resources = ("TCPIP0::192.0.2.1::5025::SOCKET", "TCPIP0::192.0.2.2::hpib7,16::INSTR")
            with patch.object(discovery, "resource_manager", return_value=manager), \
                 running_server(Path(temp) / "instrument.sock") as client:
                for name, driver, address, endpoint in (
                        ("source", "AgilentN5183A", "192.0.2.1", manager.resources[0]),
                        ("vna", "AgilentE8363C", "192.0.2.2", manager.resources[1])):
                    client.register(name, driver, address)
                    with self.assertRaisesRegex(RemoteError, "already registered"):
                        client.register("duplicate", driver, endpoint)
                    with self.assertRaisesRegex(RemoteError, "already registered"):
                        client.identify(endpoint)
                self.assertTrue(all(c["registered"] for c in client.enumerate()))
                client.register("separate_endpoint", "AgilentE8363C", "TCPIP0::192.0.2.2::inst0::INSTR")
                client.register("gpib", "YokogawaGS200", "GPIB::05::00::instr")
                with self.assertRaisesRegex(RemoteError, "already registered"):
                    client.register("duplicate", "YokogawaGS200", "GPIB0::5::INSTR")
                self.assertEqual(manager.opened, [])

    def test_whitelist_requires_complete_addresses_and_preserves_valid_entries(self):
        """Bare IP whitelist lines warn instead of inventing a LAN endpoint."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch.dict(os.environ, {"AUSPEX_HOME": temp}):
            path = Path(temp) / "instrument_whitelist.txt"
            path.write_text("192.0.2.1\ngpib::05::00::instr\nasrl/dev/ttyUSB0::instr\n", encoding="utf-8")
            addresses, filename, warning = discovery.read_whitelist()
            self.assertEqual(addresses, {"GPIB0::5::INSTR", "ASRL/dev/ttyUSB0::INSTR"})
            self.assertEqual(filename, str(path))
            self.assertIn(f"{path}:1", warning)
            self.assertIn("complete VISA address", warning)

    def test_saved_duplicates_are_all_skipped_and_reported_without_losing_originals(self):
        """Migration preserves rejected records, restores unrelated devices, and exposes errors over IPC."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch.dict(os.environ, {"AUSPEX_HOME": temp}):
            socket = Path(temp) / "instrument.sock"
            state = Path(str(socket) + ".state.json")
            definitions = {
                "source": definition("AgilentN5183A", "192.0.2.1"),
                "source_alias": definition("AgilentN5183A", "TCPIP0::192.0.2.1::5025::SOCKET"),
                "vna": definition("AgilentE8363C", "192.0.2.2"),
                "vna_alias": definition("AgilentN5230A", "TCPIP0::192.0.2.2::hpib7,16::INSTR"),
                "unrelated": definition("SimulatedSource", "SIM::source"),
                "migrated": definition("AgilentE8363C", "192.0.2.3", state="CONNECTED"),
            }
            state.write_text(json.dumps({"instruments": definitions, "run": None}))
            with running_server(socket) as client:
                status = client.status()
                self.assertEqual(set(status["instruments"]), {"unrelated", "migrated"})
                self.assertEqual(status["instruments"]["migrated"]["physical"], "TCPIP0::192.0.2.3::hpib7,16::INSTR")
                self.assertEqual(status["instruments"]["migrated"]["state"], "UNKNOWN")
                self.assertEqual(set(status["registration_errors"]), {"source", "source_alias", "vna", "vna_alias"})
                self.assertTrue(all("None of these entries was restored" in e for e in status["registration_errors"].values()))
                self.assertFalse(any(s["worker_running"] for s in status["instruments"].values()))
                from utils.src.operator_commands import OperatorCommands
                self.assertIn("REGISTRATION ERROR", OperatorCommands(socket, None).execute(["list"]))
                text = format_status({"instruments": {"reachable": True, **status},
                                      "experiments": {"reachable": False, "error": "not started"}})
                self.assertIn("source_alias", text)
                self.assertIn("REGISTRATION ERROR", text)
            saved = json.loads(state.read_text())
            self.assertEqual(saved["rejected_instruments"]["source"]["spec"], definitions["source"])
            with running_server(socket) as client:
                self.assertEqual(len(client.status()["registration_errors"]), 4)
                client.register("source", "AgilentN5183A", "192.0.2.1")  # Explicit operator choice.
                self.assertNotIn("source", client.status()["registration_errors"])
                self.assertIn("source_alias", client.status()["registration_errors"])
                client.recover("migrated")
                client.clear_instruments()
                self.assertEqual(client.status()["registration_errors"], {})
            self.assertEqual(json.loads(state.read_text())["rejected_instruments"], {})

    def test_invalid_saved_driver_is_reported_without_preventing_service_start(self):
        """A bad saved definition cannot silently disappear or prevent unrelated inventory use."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "state.json"
            bad = definition("UnavailableDriver", "GPIB0::5::INSTR")
            path.write_text(json.dumps({"instruments": {"bad": bad}, "run": None}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                registry = Registry(path)
            self.assertEqual(registry.instruments, {})
            self.assertEqual(registry.rejected_instruments["bad"]["spec"], bad)
            self.assertIn("REGISTRATION ERROR", output.getvalue())
            self.assertIn("UnavailableDriver", output.getvalue())
            clean = Registry(path, clean=True, quiet=True)
            self.assertEqual(clean.registration_errors(), {})
