"""Offline instrument contract tests: no VISA backend or hardware required."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

from auspex_v2.instruments import (AgilentE8363C, AgilentN5183A, AgilentN5230A,
    BoolCommand, FloatCommand, Instrument, InstrumentError, IntCommand,
    SCPIInstrument, StringCommand, YokogawaGS200, managed_property, operation)


class FakeTransport:
    """Record SCPI I/O and provide scripted responses without hardware."""

    def __init__(self, replies=None, binary=None):
        self.replies = replies or {}
        self.binary = binary
        self.writes = []
        self.queries = []
        self.binary_queries = []
        self.closed = False
        self.read_termination = None
        self.write_termination = None
        self.timeout = None
        self.chunk_size = None

    def write(self, text):
        self.writes.append(text)

    def query(self, text):
        self.queries.append(text)
        return self.replies[text]

    def query_binary_values(self, text, **kwargs):
        self.binary_queries.append((text, kwargs))
        return self.binary

    def close(self):
        self.closed = True


class Example(SCPIInstrument):
    """Exercise declarative command types, aliases, and extra arguments."""

    frequency = FloatCommand(scpi_string="FREQ", value_range=(1, 10), aliases=("freq",))
    number = IntCommand(get_string="NUMBER?")
    mode = StringCommand(scpi_string="MODE", value_map={"A": "0", "B": "1"})
    enabled = BoolCommand(scpi_string="ENABLED")
    channel = FloatCommand(scpi_string="CHAN{channel}:FREQ", additional_args=("channel",))


class Derived(Example):
    """Exercise an inherited command override and its alias."""

    frequency = FloatCommand(scpi_string="NEWFREQ")


class Multichannel(Instrument):
    """Model shared physical ownership and handwritten driver operations."""

    @property
    def physical_resource(self):
        return self.resource_name.rsplit("-", 1)[0]

    @managed_property
    def frequency(self):
        return self._transport().query("FREQ?")

    @frequency.setter
    def frequency(self, value):
        self._transport().write(f"FREQ {value}")

    @operation
    def status(self):
        return self._transport().query("STATUS?")


class TestCommandDefinitions(unittest.TestCase):
    """Validate command metadata and direct-use behavior with fake SCPI I/O."""

    def setUp(self):
        self.io = FakeTransport({"FREQ?;": "5", "NUMBER?": "3.0", "MODE?;": "1",
                                 "ENABLED?;": "0", "CHAN2:FREQ?;": "2.5"})
        self.device = Example("SIM", interface=self.io)
        self.device.connect()

    def tearDown(self):
        self.device.disconnect()

    def test_metadata_is_inspectable_and_inherited(self):
        self.assertIs(Example.commands["freq"], Example.commands["frequency"])
        self.assertIsNotNone(Example.commands["frequency"].get_string)  # GET
        self.assertIsNotNone(Example.commands["frequency"].set_string)  # SET
        self.assertIsNone(Example.commands["number"].set_string)  # GET only
        self.assertEqual(Derived.commands["frequency"].get_string, "NEWFREQ?;")
        self.assertIs(Derived.commands["freq"], Derived.commands["frequency"])
        self.assertEqual(Derived.commands["number"].get_string, "NUMBER?")
        self.assertEqual(Derived("SIM", interface=FakeTransport({"NEWFREQ?;": "4"})).commands["frequency"].get_string,
                         "NEWFREQ?;")

    def test_read_write_alias_and_inherited_override(self):
        """Keep aliases bound to the overridden command in a subclass."""
        self.assertEqual(self.device.freq, 5)
        self.device.set_freq(6)
        self.assertIn("FREQ 6.000000E+00", self.io.writes)
        self.assertEqual(self.device.number, 3)
        self.assertFalse(self.device.enabled)  # bool("0") would be True
        derived_io = FakeTransport({"NEWFREQ?;": "4"})
        derived = Derived("SIM", interface=derived_io)
        derived.connect()
        derived.set_freq(5)
        self.assertEqual(derived_io.writes, ["NEWFREQ 5.000000E+00"])
        derived.disconnect()

    def test_validation_before_io(self):
        with self.assertRaises(ValueError):
            self.device.frequency = 11
        with self.assertRaises(ValueError):
            self.device.mode = "C"
        self.assertEqual(self.io.writes, [])
        self.device.mode = "A"
        self.assertEqual(self.device.mode, "B")
        self.assertEqual(self.io.writes, ["MODE 0"])
        with self.assertRaises(AttributeError):
            self.device.number = 2

    def test_parameterized_methods(self):
        self.device.set_channel(3.0, channel=2)
        self.assertEqual(self.device.get_channel(channel=2), 2.5)
        self.assertEqual(self.io.writes[-1], "CHAN2:FREQ 3.000000E+00")
        with self.assertRaises(AttributeError):
            _ = self.device.channel
        with self.assertRaises(TypeError):
            self.device.set_channel(3.0)

    def test_connected_lifecycle_and_failure(self):
        self.device.disconnect()
        with self.assertRaises(InstrumentError):
            _ = self.device.frequency
        with self.assertRaises(InstrumentError):
            self.device.get_frequency()

    def test_context_manager(self):
        io = FakeTransport()
        with Example("SIM", interface=io) as device:
            device.mode = "A"
        self.assertTrue(io.closed)

    def test_custom_backend_and_physical_identity(self):
        """Register handwritten properties and share a physical resource key."""
        a = Multichannel("HS9004A-009-1", interface=FakeTransport({"FREQ?": "100", "STATUS?": "OK"}))
        b = Multichannel("HS9004A-009-2", interface=FakeTransport({"FREQ?": "200"}))
        self.assertEqual(a.physical_resource, b.physical_resource)
        self.assertTrue(Multichannel.properties["frequency"].writable)
        self.assertIs(Multichannel.operations["status"], Multichannel.status)
        self.assertNotIn("get_frequency", Example.operations)
        a.connect()
        self.assertEqual(a.frequency, "100")
        self.assertEqual(a.status(), "OK")  # CALL can return data
        a.frequency = 101
        self.assertEqual(a.interface.writes, ["FREQ 101"])
        a.disconnect()


class TestDrivers(unittest.TestCase):
    """Test targeted V2 driver ports using deterministic fake transports."""

    def test_ported_driver_kinds_are_fixed_role_families(self):
        from auspex_v2.instruments import InstrumentKind
        self.assertEqual(AgilentN5183A.KIND, InstrumentKind.MICROWAVE_SOURCE)
        self.assertEqual(AgilentE8363C.KIND, InstrumentKind.VNA)
        self.assertEqual(AgilentN5230A.KIND, InstrumentKind.VNA)
        self.assertEqual(YokogawaGS200.KIND, InstrumentKind.DC_SOURCE)

    def test_visa_transport_always_uses_pyvisa_py(self):
        from auspex_v2.instruments.drivers.visa import VisaTransport
        manager = unittest.mock.Mock()
        manager.open_resource.return_value = unittest.mock.Mock()
        module = unittest.mock.Mock()
        module.ResourceManager.return_value = manager
        with patch.dict(sys.modules, {"pyvisa": module}):
            transport = VisaTransport("TCPIP0::host::INSTR")
        module.ResourceManager.assert_called_once_with("@py")
        manager.open_resource.assert_called_once_with("TCPIP0::host::inst0::INSTR")
        transport.close()

    def test_visa_resource_normalization_and_failure(self):
        """Normalize VISA addresses without masking connection failures."""
        with patch("auspex_v2.instruments.drivers.visa.VisaTransport") as transport:
            io = FakeTransport()
            transport.return_value = io
            device = AgilentN5183A("192.0.2.5")
            device.connect()
            transport.assert_called_once_with("TCPIP0::192.0.2.5::5025::SOCKET")
            self.assertEqual(device.physical_resource, "TCPIP0::192.0.2.5::5025::SOCKET")
            device.disconnect()
        with patch("auspex_v2.instruments.drivers.visa.VisaTransport", side_effect=OSError("offline")):
            device = AgilentN5183A("192.0.2.5")
            with self.assertRaisesRegex(OSError, "offline"):
                device.connect()
            self.assertFalse(device.connected)
            self.assertIsNone(device.interface)  # No MagicMock fallback

    def test_n5183a(self):
        """Exercise microwave-source properties and connection settings."""
        io = FakeTransport({":freq?;": "5e9", ":output?;": "1"})
        device = AgilentN5183A("SIM", interface=io, io_timeout=12)
        device.connect()
        device.frequency = 5e9
        device.output = True
        self.assertEqual(device.frequency, 5e9)
        self.assertIs(device.output, True)
        self.assertEqual(io.writes, [":freq 5.000000E+09", ":output 1"])
        self.assertEqual(io.timeout, 12000)
        device.disconnect()
        self.assertTrue(io.closed)

    def test_base_io_timeout_failure_closes_transport(self):
        class BrokenTimeout(FakeTransport):
            def __setattr__(self, name, value):
                if name == "timeout" and value is not None:
                    raise OSError("timeout configuration failed")
                object.__setattr__(self, name, value)
        io = BrokenTimeout()
        device = SCPIInstrument("SIM", interface=io, io_timeout=1)
        with self.assertRaisesRegex(OSError, "timeout configuration failed"):
            device.connect()
        self.assertFalse(device.connected)
        self.assertTrue(io.closed)
        self.assertIs(device.interface, io)  # Injected ownership/reference is preserved.

    def test_connect_configuration_failure_cleans_up(self):
        """Close a partially initialized connection after a transport error."""
        class BrokenTransport(FakeTransport):
            """Simulate a transport failing during GS200 configuration."""

            def write(self, text):
                raise OSError("configuration failed")
        io = BrokenTransport()
        device = YokogawaGS200("SIM", interface=io)
        with self.assertRaisesRegex(OSError, "configuration failed"):
            device.connect()
        self.assertFalse(device.connected)
        self.assertTrue(io.closed)

    def test_gs200_mode_level_alias_and_output(self):
        """Verify source-mode mapping, level aliasing, and RF output state."""
        io = FakeTransport({":source:function?;": "CURR", ":source:level?;": "0.001",
                            ":output:state?;": "1"})
        device = YokogawaGS200("SIM", interface=io)
        device.connect()
        self.assertEqual(device.mode, "current")
        self.assertEqual(device.value, 0.001)
        self.assertTrue(device.output)
        device.set_value(0.002)
        device.mode = "voltage"
        device.output = False
        self.assertEqual(io.writes[1:], [":source:level 2.000000E-03",
                                         ":source:function VOLT", ":output:state 0"])
        device.disconnect()

    def test_gs200_ramp_and_read_only_sense(self):
        """Bound stepped level writes and keep sense readback read-only."""
        io = FakeTransport({":source:level?;": "0", ":fetch?": "0.004", ":output:state?;": "0"})
        device = YokogawaGS200("SIM", interface=io)
        device.connect()
        self.assertEqual(device.sense_value, 0.004)
        self.assertFalse(device.output)
        with self.assertRaises(AttributeError):
            device.sense_value = 1
        with patch("auspex_v2.instruments.drivers.core.time.sleep"):
            device.ramp = 0.0002
        self.assertEqual(io.writes, [":sense:trigger immediate",
                                      ":source:level 1.000000E-04", ":source:level 2.000000E-04"])
        with self.assertRaises(ValueError):
            device.ramp = 0.2
        device.disconnect()

    def test_vna_ports_and_power_bounds(self):
        """Check per-port limits and output modes before VNA writes."""
        io = FakeTransport({"SOUR:POW1? MIN": "-40", "SOUR:POW1? MAX": "10",
                            "SOUR:POW1?": "-20", "SOUR:POW1:MODE?": "AUTO",
                            "SOUR:POW2:MODE?": "OFF"})
        device = AgilentE8363C("SIM", interface=io)
        device.connect()
        self.assertEqual(device.get_port_power(1), -20)
        self.assertEqual(device.output_enable, {1: True, 2: False})
        with self.assertRaises(ValueError):
            device.set_port_power(1, 20)
        with self.assertRaises(ValueError):
            device.set_port_power(3, -10)
        self.assertEqual(io.writes, [])
        device.set_port_power(1, -25)
        device.output_enable = {"2": True}  # JSON-backed proxies carry object keys as strings.
        self.assertEqual(io.writes, ["SOUR:POW1 -25", "SOUR:POW2:MODE AUTO"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            device.output_enable = {1: True, "1": False}
        device.disconnect()

    def test_vna_complex_trace_and_linear_axis(self):
        """Construct a linear axis from read-back endpoints, not a stimulus vector."""
        for cls in (AgilentE8363C, AgilentN5230A):
            with self.subTest(cls=cls):
                io = FakeTransport({"CALC:PAR:CAT?": '"M_S21,S21"',
                                    ":SENSe:SWEep:POINTS?;": "3",
                                    ":SENSe:FREQuency:STARt?;": "100",
                                    ":SENSe:FREQuency:STOP?;": "300"},
                                   binary=[1, 2, 3, 4, 5, 6])
                device = cls("SIM", interface=io)
                device.connect()
                self.assertEqual(io.writes, [])  # connect must not reconfigure RF outputs
                frequencies, data = device.get_trace(measurement="S21", restart_measurement=False)
                np.testing.assert_array_equal(frequencies, [100, 200, 300])
                np.testing.assert_array_equal(data, np.array([1+2j, 3+4j, 5+6j]))
                self.assertEqual(data.dtype, np.complex64)
                self.assertEqual(io.binary_queries[0][0], ":CALC:DATA? SDATA")
                self.assertEqual(io.writes, [":CALCulate:PARameter:SELect 'M_S21'", "FORM REAL,32"])
                device.disconnect()

    def test_vna_rejects_missing_measurement_and_truncated_binary(self):
        io = FakeTransport({"CALC:PAR:CAT?": '"M_S21,S21"',
                            ":SENSe:SWEep:POINTS?;": "2"}, binary=[1, 2])
        device = AgilentE8363C("SIM", interface=io)
        device.connect()
        with self.assertRaises(ValueError):
            device.get_trace("S11", restart_measurement=False)
        with self.assertRaises(InstrumentError):
            device.get_trace("S21", restart_measurement=False)
        device.disconnect()

    def test_vna_completion_discards_stale_event_before_trigger(self):
        for cls in (AgilentE8363C, AgilentN5230A):
            for averaging in (False, True):
                with self.subTest(driver=cls.__name__, averaging=averaging):
                    events = []
                    esr = iter(["1", "0", "1"])  # stale, not done yet, newly done
                    class ScriptedTransport(FakeTransport):
                        def query(self, text):
                            events.append(("query", text))
                            if text == "*ESR?":
                                return next(esr)
                            return super().query(text)

                        def write(self, text):
                            events.append(("write", text))
                            super().write(text)
                    io = ScriptedTransport({":SENSe1:AVERage:STATe?;": str(int(averaging)),
                                            ":SENSe1:AVERage:COUNt?;": "3"})
                    device = cls("SIM", interface=io)
                    device.connect()
                    with patch("auspex_v2.instruments.drivers.agilent.time.sleep") as sleep:
                        device.reaverage()
                    self.assertEqual(events[0], ("query", "*ESR?"))
                    self.assertEqual(events.count(("query", "*ESR?")), 3)
                    trigger = "ABORT; SENSe1:SWEep:MODE GRO" if averaging else "ABORT; SENS:SWE:MODE SING"
                    self.assertLess(events.index(("write", trigger)), events.index(("write", "*OPC")))
                    sleep.assert_called_once()
                    device.disconnect()

    def test_vna_invalid_timeout_does_not_touch_hardware(self):
        io = FakeTransport()
        device = AgilentE8363C("SIM", interface=io)
        device.connect()
        for timeout in (0, -1, True, "1", float("nan"), float("inf")):
            for acquire in (device.reaverage, device.get_trace):
                with self.subTest(timeout=timeout, method=acquire.__name__), self.assertRaises(ValueError):
                    acquire(timeout=timeout)
        self.assertEqual(io.writes, [])
        self.assertEqual(io.queries, [])
        device.disconnect()

    def test_vna_acquisition_triggers_only_when_requested(self):
        """Trigger a requested sweep and restore continuous acquisition mode."""
        io = FakeTransport({"CALC:PAR:CAT?": '"M_S21,S21"',
                            ":SENSe1:AVERage:STATe?;": "0", "*ESR?": "1",
                            ":SENSe:SWEep:POINTS?;": "1",
                            ":SENSe:FREQuency:STARt?;": "100",
                            ":SENSe:FREQuency:STOP?;": "100"}, binary=[1, -1])
        device = AgilentE8363C("SIM", interface=io)
        device.connect()
        device.get_trace("S21")
        self.assertIn("ABORT; SENS:SWE:MODE SING", io.writes)
        self.assertIn("*OPC", io.writes)
        self.assertEqual(io.writes[-1], "SENS:SWE:MODE CONT")
        device.disconnect()


if __name__ == "__main__":
    unittest.main()
