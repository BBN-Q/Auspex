"""Small simulated source used to exercise the server without hardware."""

import os

import numpy as np

from .core import Command, Instrument, managed_property, operation
from ..kinds import InstrumentKind

__all__ = ["SimulatedSource", "SimulatedVNA"]


class _MemoryBackend:
    """Hold scalar source settings in memory for offline server tests."""

    def __init__(self):
        self.frequency = 1e9
        self.output = False
        self.channels = {1: 1e9, 2: 1e9}

    def close(self):
        if getattr(self, "fail_disconnect", False):
            raise OSError("Simulated disconnect failure")


class _ChannelFrequency(Command):
    """Exercise parameterized server GET/SET without a SCPI transport."""

    def __init__(self):
        super().__init__(get_string=":CH{channel}:FREQ?", set_string=":CH{channel}:FREQ {}",
                         additional_args=("channel",))

    def get(self, instrument, **kwargs):
        self._check_args(kwargs)
        return instrument._transport().channels[kwargs["channel"]]

    def set(self, instrument, value, **kwargs):
        self._check_args(kwargs)
        if not 1e6 <= value <= 20e9:
            raise ValueError("Frequency must be between 1 MHz and 20 GHz")
        instrument._transport().channels[kwargs["channel"]] = float(value)


class SimulatedSource(Instrument):
    """Provide a hardware-free source with managed settings and a test crash."""
    KIND = InstrumentKind.MICROWAVE_SOURCE

    channel_frequency = _ChannelFrequency()

    def connect(self, resource_name=None):
        if self.interface is None:
            self.interface = _MemoryBackend()
        super().connect(resource_name)
        if self.resource_name == "SIM::FAIL_CONNECT":
            raise OSError("Simulated connect failure")
        self.interface.fail_disconnect = self.resource_name == "SIM::FAIL_DISCONNECT"

    @managed_property
    def frequency(self):
        return self._transport().frequency

    @frequency.setter
    def frequency(self, value):
        if not isinstance(value, (float, int)) or not 1e6 <= value <= 20e9:
            raise ValueError("Frequency must be between 1 MHz and 20 GHz")
        self._transport().frequency = float(value)

    @managed_property
    def output(self):
        return self._transport().output

    @output.setter
    def output(self, value):
        if type(value) is not bool:
            raise ValueError("Output requires a boolean")
        self._transport().output = value

    @operation
    def identify(self):
        return f"SimulatedSource:{self.resource_name}"

    @operation
    def wait_for_test(self, seconds):
        import time
        time.sleep(seconds)

    @operation
    def crash_for_test(self):
        """Exercise unexpected worker death; available only on the simulator."""
        os._exit(1)


class _MemoryVNABackend:
    """Hold a simulated VNA's sweep configuration without real VISA calls."""

    def __init__(self):
        self.center = 6e9
        self.span = 1e9
        self.points = 2000

    def close(self):
        pass


class SimulatedVNA(Instrument):
    """Produce two-array complex resonance traces for bulk IPC tests."""
    KIND = InstrumentKind.VNA

    def connect(self, resource_name=None):
        if self.interface is None:
            self.interface = _MemoryVNABackend()
        super().connect(resource_name)

    @managed_property
    def frequency_center(self):
        return self._transport().center

    @frequency_center.setter
    def frequency_center(self, value):
        if not isinstance(value, (int, float)) or not 1e6 <= value <= 20e9:
            raise ValueError("Center must be between 1 MHz and 20 GHz")
        self._transport().center = float(value)

    @managed_property
    def frequency_span(self):
        return self._transport().span

    @frequency_span.setter
    def frequency_span(self, value):
        if not isinstance(value, (int, float)) or not 1e3 <= value <= 2e9:
            raise ValueError("Span must be between 1 kHz and 2 GHz")
        self._transport().span = float(value)

    @managed_property
    def num_points(self):
        return self._transport().points

    @num_points.setter
    def num_points(self, value):
        if type(value) is not int or not 2 <= value <= 1_000_000:
            raise ValueError("Number of points must be between 2 and 1,000,000")
        self._transport().points = value

    @operation
    def get_trace(self, measurement="S21"):
        """Return readback frequencies and a complex resonator transmission."""
        if measurement != "S21":
            raise ValueError("Only S21 is simulated")
        io = self._transport()
        frequencies = np.linspace(io.center - io.span / 2, io.center + io.span / 2, io.points)
        detuning = (frequencies - 6.1e9) / 0.5e6
        response = (1 - 0.7 / (1 + 1j * detuning)).astype(np.complex64)
        return frequencies, response
