"""Selected Agilent source and network-analyzer drivers."""

from __future__ import annotations

__all__ = ["AgilentN5183A", "AgilentE8363C", "AgilentN5230A"]

import time

import numpy as np

from .core import (BoolCommand, FloatCommand, InstrumentError, IntCommand,
                   SCPIInstrument, StringCommand, managed_property, operation)
from ..kinds import InstrumentKind


def _validate_sweep_timeout(timeout):
    if timeout is not None and (type(timeout) not in (int, float)
                                or not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite positive seconds or None")


class AgilentN5183A(SCPIInstrument):
    """Agilent N5183A microwave source."""

    IDN_STRING = "N5183A"
    VISA_IPV4_DEVICE = "5025::SOCKET"
    KIND = InstrumentKind.MICROWAVE_SOURCE
    instrument_type = "Microwave Source"
    frequency = FloatCommand(scpi_string=":freq")
    power = FloatCommand(scpi_string=":power")
    phase = FloatCommand(scpi_string=":phase")
    alc = StringCommand(scpi_string=":power:alc", value_map={True: "1", False: "0"})
    mod = StringCommand(scpi_string=":output:mod", value_map={True: "1", False: "0"})
    output = StringCommand(scpi_string=":output", value_map={True: "1", False: "0"})

    def connect(self, resource_name=None):
        """Normalize bare IPv4 addresses and configure VISA I/O on connection."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\n"
            self.interface.write_termination = "\n"
            if self.io_timeout is None:
                self.interface.timeout = 3000
        except Exception:
            self.disconnect()
            raise


class _AgilentNetworkAnalyzer(SCPIInstrument):
    """Common subset of the E8363C/N5230A VNA API.

    ``connect`` does not change output power or measurement configuration.
    ``get_trace`` explicitly triggers a sweep by default. The returned
    frequency axis is linear spacing between read-back endpoints. Linear sweeps
    are assumed; logarithmic and segmented stimulus readback is not implemented.
    """

    VISA_IPV4_DEVICE = "hpib7,16::INSTR"
    KIND = InstrumentKind.VNA
    instrument_type = "Vector Network Analyzer"
    ports = ()
    frequency_center = FloatCommand(scpi_string=":SENSe:FREQuency:CENTer")
    frequency_start = FloatCommand(scpi_string=":SENSe:FREQuency:STARt", formatter="{:.0f}")
    frequency_stop = FloatCommand(scpi_string=":SENSe:FREQuency:STOP", formatter="{:.0f}")
    frequency_span = FloatCommand(scpi_string=":SENSe:FREQuency:SPAN", formatter="{:.0f}")
    if_bandwidth = FloatCommand(scpi_string=":SENSe1:BANDwidth")
    num_points = IntCommand(scpi_string=":SENSe:SWEep:POINTS")
    averaging_enable = BoolCommand(scpi_string=":SENSe1:AVERage:STATe",
                                   value_map={False: "0", True: "1"})
    averaging_factor = IntCommand(scpi_string=":SENSe1:AVERage:COUNt")

    def __init__(self, resource_name=None, name=None, *, interface=None, io_timeout=None):
        super().__init__(resource_name, name, interface=interface, io_timeout=io_timeout)

    def connect(self, resource_name=None):
        """Open VNA I/O without selecting measurements or changing RF output."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\n"
            self.interface.write_termination = "\n"
            if self.io_timeout is None:
                self.interface.timeout = 10000
            self.interface.chunk_size = 2 ** 20
        except Exception:
            self.disconnect()
            raise

    @managed_property
    def measurements(self):
        """Map active S-parameter to trace name (read-only, no changes)."""
        reply = self._transport().query("CALC:PAR:CAT?").strip().strip('"')
        if not reply:
            return {}
        fields = reply.split(",")
        if len(fields) % 2:
            raise InstrumentError(f"Malformed VNA measurement catalog: {reply!r}")
        return {s.upper(): label for label, s in zip(fields[::2], fields[1::2])}

    @operation
    def get_port_power(self, port):
        if port not in self.ports:
            raise ValueError(f"Unknown VNA port {port}")
        return float(self._transport().query(f"SOUR:POW{port}?"))

    @operation
    def set_port_power(self, port, power):
        if port not in self.ports:
            raise ValueError(f"Unknown VNA port {port}")
        io = self._transport()
        lower = float(io.query(f"SOUR:POW{port}? MIN"))
        upper = float(io.query(f"SOUR:POW{port}? MAX"))
        if not lower <= power <= upper:
            raise ValueError(f"Port {port} power {power} outside {lower} to {upper} dBm")
        io.write(f"SOUR:POW{port} {power}")

    @managed_property
    def output_enable(self):
        return {port: self._transport().query(f"SOUR:POW{port}:MODE?").strip().upper() == "AUTO"
                for port in self.ports}

    @output_enable.setter
    def output_enable(self, state):
        if isinstance(state, dict):
            settings = {}
            for port, enabled in state.items():
                if isinstance(port, str) and port.isdecimal():
                    port = int(port)
                if port in settings:
                    raise ValueError("Duplicate VNA output port")
                settings[port] = enabled
        else:
            settings = {port: state for port in self.ports}
        if not settings.keys() <= set(self.ports) or any(type(v) is not bool for v in settings.values()):
            raise ValueError("Expected a boolean or a mapping from valid port numbers to booleans")
        for port, enabled in settings.items():
            self._transport().write(f"SOUR:POW{port}:MODE {'AUTO' if enabled else 'OFF'}")

    @operation
    def set_measurements(self, measurements):
        """Explicitly replace active measurements; this changes VNA state."""
        requested = [s.upper() for s in measurements]
        allowed = {f"S{a}{b}" for a in self.ports for b in self.ports}
        if not requested or any(s not in allowed for s in requested):
            raise ValueError(f"Measurements must be in {sorted(allowed)}")
        self._transport().write("CALC:PAR:DEL:ALL")
        for s in requested:
            self._transport().write(f'CALC:PAR:DEF "M_{s}",{s}')

    @operation
    def reaverage(self, *, timeout=None):
        """Start one sweep/averaging group; optionally impose a total deadline."""
        _validate_sweep_timeout(timeout)
        io = self._transport()
        io.query("*ESR?")  # Clear stale event bits before starting this acquisition.
        if self.averaging_enable:
            io.write(":SENSe1:AVERage:CLEar")
            io.write(f"SENSe1:SWEep:GROups:COUNt {self.averaging_factor}")
            io.write("ABORT; SENSe1:SWEep:MODE GRO")
        else:
            io.write("ABORT; SENS:SWE:MODE SING")
        io.write("*OPC")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if int(io.query("*ESR?")) & 1:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("VNA sweep did not complete")
            time.sleep(0.1)

    @operation
    def get_trace(self, measurement=None, restart_measurement=True, timeout=None):
        """Return (frequency in Hz, complex S-parameter), assuming a linear sweep.

        Coordinates are synthesized from read-back endpoints and point count, not
        queried as a stimulus vector. Configure linear sweep mode before use.
        """
        _validate_sweep_timeout(timeout)
        active = self.measurements
        if measurement is None:
            if not active:
                raise InstrumentError("No active VNA measurements")
            measurement = next(iter(active))
        measurement = measurement.upper()
        if measurement not in active:
            raise ValueError(f"Unknown measurement {measurement!r}; active: {list(active)}")
        io = self._transport()
        io.write(f":CALCulate:PARameter:SELect '{active[measurement]}'")
        try:
            if restart_measurement:
                self.reaverage(timeout=timeout)
            io.write("FORM REAL,32")
            raw = np.asarray(io.query_binary_values(":CALC:DATA? SDATA", datatype="f",
                             is_big_endian=True, expect_termination=True), dtype=np.float32)
        finally:
            if restart_measurement:
                io.write("SENS:SWE:MODE CONT")
        count = self.num_points
        if raw.ndim != 1 or raw.size != 2 * count:
            raise InstrumentError(f"Expected {2 * count} interleaved floats, received {raw.size}")
        values = raw[::2] + 1j * raw[1::2]
        frequencies = np.linspace(self.frequency_start, self.frequency_stop, count)
        return frequencies, values


class AgilentN5230A(_AgilentNetworkAnalyzer):
    """Agilent N5230A four-port VNA."""
    IDN_STRING = "N5230A"
    ports = (1, 2, 3, 4)


class AgilentE8363C(_AgilentNetworkAnalyzer):
    """Agilent E8363C two-port VNA."""
    IDN_STRING = "E8363C"
    ports = (1, 2)


# Deliberately absent from __all__: manual porting and hardware review remain.
class HP33120A(SCPIInstrument):
    """Unavailable HP33120A stub; reference: v1/src/auspex/instruments/agilent.py."""

    KIND = InstrumentKind.FUNCTION_GENERATOR

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HP33120A requires manual V2 porting and hardware validation")


class Agilent33220A(SCPIInstrument):
    """Unavailable 33220A; review V1 agilent.py's inverted output_sync mapping."""

    KIND = InstrumentKind.FUNCTION_GENERATOR

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Agilent33220A requires manual V2 porting and hardware validation")


class Agilent33500B(SCPIInstrument):
    """Unavailable 33500B; V1 agilent.py marks binary upload NOT YET WORKING."""

    KIND = InstrumentKind.AWG

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Agilent33500B requires manual V2 porting and hardware validation")


class Agilent34970A(SCPIInstrument):
    """Unavailable 34970A stub; reference: v1/src/auspex/instruments/agilent.py."""

    KIND = InstrumentKind.DAQ

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Agilent34970A requires manual V2 porting and hardware validation")


class AgilentN9010A(SCPIInstrument):
    """Unavailable N9010A stub; reference: v1/src/auspex/instruments/agilent.py."""

    KIND = InstrumentKind.SPECTRUM_ANALYZER

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AgilentN9010A requires manual V2 porting and hardware validation")
