# Copyright 2020 Raytheon BBN Technologies
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""DS Instruments VISA serial sources and attenuator ported from V1."""

import logging

from .core import (SCPIInstrument, BoolCommand, FloatCommand, experimental,
                   managed_property, operation)
from ..kinds import InstrumentKind

__all__ = ["DSInstrumentsSG12000", "DSInstrumentsSG12000Pro", "DAT64H"]
_logger = logging.getLogger(__name__)


class _DSSerialInstrument(SCPIInstrument):
    """Shared serial framing, with no device-setting changes on connection."""

    def connect(self, resource_name=None):
        """Apply V1 serial framing, retaining an explicit caller I/O timeout."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\r\n"
            self.interface.write_termination = "\r\n"
            self.interface.baud_rate = 115200
            if self.io_timeout is None:
                self.interface.timeout = 1000
        except Exception:
            self.disconnect()
            raise


class DSInstruments(_DSSerialInstrument):
    """Shared signal-generator protocol; only concrete models are admitted."""

    KIND = InstrumentKind.MICROWAVE_SOURCE
    output = BoolCommand(scpi_string="OUTP:STAT", value_map={True: "ON", False: "OFF"})

    @managed_property
    def frequency(self):
        return float(self._transport().query("FREQ:CW?").replace("HZ", ""))

    @frequency.setter
    def frequency(self, value):
        self._transport().write(f"FREQ:CW {value}HZ")

    @managed_property
    def power(self):
        return float(self._transport().query("POWER?").replace("dBm", ""))

    @power.setter
    def power(self, value):
        rounded = round(value * 2) / 2
        if rounded != value:
            _logger.info("Set power rounded to %s dBm", rounded)
        self._transport().write(f"POWER {rounded}dBm")

    @managed_property
    def vernier(self):
        return float(self._transport().query("VERNIER?"))

    @vernier.setter
    def vernier(self, value):
        self._transport().write(f"VERNIER {value}")

    @operation
    def set_buzzer(self, value):
        self._transport().write(f"*BUZZER {self._on_off(value)}")

    @operation
    def set_display(self, value):
        self._transport().write(f"*DISPLAY {self._on_off(value)}")

    @staticmethod
    def _on_off(value):
        if value not in (0, 1):
            raise ValueError("Expected a boolean or 0/1")
        return "ON" if value else "OFF"

    @operation
    def reset(self):
        self._transport().write("*RST")

    @operation
    def ping(self):
        return self._transport().query("*PING?")

    @operation
    def save(self):
        self._transport().write("*SAVESTATE")

    @operation
    def usbvoltage(self):
        return self._transport().query("*SYSVOLTS?")

    @managed_property
    def device_name(self):
        return self._transport().query("*UNITNAME?")

    @device_name.setter
    def device_name(self, value):
        self._transport().write(f"*UNITNAME {value}")

    @operation
    def errors(self):
        return self._transport().query("SYST:ERR?")

    @operation
    def cls(self):
        self._transport().write("*CLS")

    @operation
    def debug(self):
        return self._transport().query("SYST:DBG?")


@experimental
class DSInstrumentsSG12000(DSInstruments):
    """SG12000 serial RF source; reference accepts V1's string 0, 1 or A."""

    @managed_property
    def reference(self):
        return {"ON": 1, "OFF": 0}[self._transport().query("*INTREF?").strip()]

    @reference.setter
    def reference(self, value):
        if value not in ("0", "1", "A"):
            raise ValueError("reference must be '0', '1' or 'A'")
        self._transport().write(f"*INTERNALREF {value}")

    @operation
    def set_low_power_mode(self, value):
        """Reduce RF output by the model's approximately 4–7 dB low-power mode."""
        self._transport().write(f"*LPMODE {self._on_off(value)}")


@experimental
class DSInstrumentsSG12000Pro(DSInstruments):
    """SG12000Pro serial RF source with explicit reference control."""

    @managed_property
    def reference(self):
        return self._transport().query("SYSREF?")

    @reference.setter
    def reference(self, value):
        if value not in ("INT", "EXT", "AUTO", "FREE", "OFF"):
            raise ValueError("reference must be INT, EXT, AUTO, FREE or OFF")
        self._transport().write(f"SYSREF {value}")

    @operation
    def update_reference(self):
        self._transport().write("SYSREF UPDATE")

    @operation
    def detect_ext_ref(self):
        return self._transport().query("SYSREF STATUS?")

    @operation
    def temp(self):
        """Read the V1 temperature command, still awaiting hardware validation."""
        return self._transport().query("*TEMPC?")


@experimental
class DAT64H(_DSSerialInstrument):
    """DAT64H tunable serial attenuator, with numeric attenuation readback."""

    KIND = InstrumentKind.ATTENUATOR
    attenuation = FloatCommand(scpi_string="ATT")
    step = FloatCommand(scpi_string="STEP")

    @operation
    def increment(self):
        self._transport().write("INCR")

    @operation
    def decrement(self):
        self._transport().write("DECR")
