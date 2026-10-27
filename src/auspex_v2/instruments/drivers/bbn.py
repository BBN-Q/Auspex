# Copyright 2016 Raytheon BBN Technologies
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""BBN VISA serial relay and attenuation hardware, without native APS backends."""

import time

from .core import (SCPIInstrument, InstrumentError, experimental,
                   managed_property, operation)
from ..kinds import InstrumentKind

__all__ = ["RelayDriver", "DigitalAttenuator"]


@experimental
class RelayDriver(SCPIInstrument):
    """Two six-pole Radiall relays controlled through a virtual COM port."""

    KIND = InstrumentKind.SWITCH
    pulse_width = 125  # milliseconds, as in V1

    def connect(self, resource_name=None):
        """Configure serial framing without parking or switching the relays."""
        super().connect(resource_name)
        try:
            self.interface.baud_rate = 9600
            self.interface.read_termination = "\n"
            self.interface.write_termination = "\n"
            if self.io_timeout is None:
                self.interface.timeout = 1000
        except Exception:
            self.disconnect()
            raise

    @operation
    def park(self):
        """Open control relays and lower CTRL/COM; this changes relay-control state."""
        io = self._transport()
        for pole in range(1, 7):
            io.write(f"W{pole:02d}F")
            io.write(f"W{8 + pole:02d}F")
        io.write("W07F")
        io.write("W08F")

    @staticmethod
    def _check_pole(relay, pole):
        if relay not in (0, 1) or pole not in range(1, 7):
            raise ValueError("relay must be 0 or 1 and pole must be 1 through 6")

    @operation
    def open_switch(self, relay, pole):
        self._check_pole(relay, pole)
        io = self._transport()
        io.write(f"W{relay * 8 + pole:02d}L")
        io.write("W07F")
        io.write(f"P08{self.pulse_width:03d}")
        self.park()

    @operation
    def close_switch(self, relay, pole):
        self._check_pole(relay, pole)
        io = self._transport()
        io.write(f"W{relay * 8 + pole:02d}L")
        io.write("W08F")
        io.write(f"P07{self.pulse_width:03d}")
        self.park()


@experimental
class DigitalAttenuator(SCPIInstrument):
    """BBN three-channel serial attenuator with explicit END acknowledgments."""

    KIND = InstrumentKind.ATTENUATOR
    NUM_CHANNELS = 3

    def connect(self, resource_name=None):
        """Configure serial I/O and retain V1's two-second startup wait."""
        super().connect(resource_name)
        try:
            self.interface.baud_rate = 115200
            self.interface.read_termination = "\r\n"
            self.interface.write_termination = "\n"
            if self.io_timeout is None:
                self.interface.timeout = 1000
            time.sleep(2)
        except Exception:
            self.disconnect()
            raise

    @classmethod
    def channel_check(cls, channel):
        if channel not in range(1, cls.NUM_CHANNELS + 1):
            raise ValueError(f"channel must be 1 through {cls.NUM_CHANNELS}")

    def _expect(self, expected):
        reply = self._transport().read()
        if reply != expected:
            raise InstrumentError(f"Expected attenuator reply {expected!r}, received {reply!r}")

    @operation
    def get_attenuation(self, chan):
        self.channel_check(chan)
        reply = self._transport().query(f"GET {chan:d}")
        self._expect("END")
        return float(reply)

    @operation
    def set_attenuation(self, chan, val):
        self.channel_check(chan)
        encoded = f"{val:.1f}"
        self._transport().write(f"SET {chan:d} {encoded}")
        # The acknowledgment describes the transmitted value, already rounded.
        self._expect(f"Setting channel {chan:d} to {float(encoded):.2f}")
        self._expect("END")

    @managed_property
    def ch1_attenuation(self):
        return self.get_attenuation(1)

    @ch1_attenuation.setter
    def ch1_attenuation(self, value):
        self.set_attenuation(1, value)

    @managed_property
    def ch2_attenuation(self):
        return self.get_attenuation(2)

    @ch2_attenuation.setter
    def ch2_attenuation(self, value):
        self.set_attenuation(2, value)

    @managed_property
    def ch3_attenuation(self):
        return self.get_attenuation(3)

    @ch3_attenuation.setter
    def ch3_attenuation(self, value):
        self.set_attenuation(3, value)


class SpectrumAnalyzer(SCPIInstrument):
    """Unavailable BBN analyzer stub, deliberately absent from __all__.

    Reference: v1/src/auspex/instruments/bbn.py. Review V1's repeated voltage
    queries and None-on-exhaustion policy before manual porting.
    """

    KIND = InstrumentKind.SPECTRUM_ANALYZER

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SpectrumAnalyzer requires manual V2 porting and hardware validation")
