# Copyright 2016 Raytheon BBN Technologies
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Picosecond VISA pulse generator ported from V1."""

from .core import (SCPIInstrument, StringCommand, FloatCommand, experimental,
                   managed_property, operation)
from ..kinds import InstrumentKind

__all__ = ["Picosecond10070A"]


@experimental
class Picosecond10070A(SCPIInstrument):
    """10070A pulser; select triggering explicitly after connection."""

    KIND = InstrumentKind.PULSE_GENERATOR
    amplitude = FloatCommand(scpi_string="amplitude")
    delay = FloatCommand(scpi_string="delay")
    duration = FloatCommand(scpi_string="duration")
    trigger_level = FloatCommand(scpi_string="level")
    period = FloatCommand(scpi_string="period")
    frequency = FloatCommand(scpi_string="frequency")
    offset = FloatCommand(scpi_string="offset")
    trigger_source = StringCommand(scpi_string="trigger", allowed_values=("INT", "EXT", "GPIB"))

    def connect(self, resource_name=None):
        """Disable response headers for parsing, without changing trigger settings."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\n"
            self.interface.write("header off")
        except Exception:
            self.disconnect()
            raise

    @managed_property
    def output(self):
        return self._transport().query("enable?").strip() == "YES"

    @output.setter
    def output(self, value):
        if type(value) is not bool:
            raise ValueError("output requires a boolean")
        self._transport().write("enable" if value else "disable")

    @operation
    def trigger(self):
        self._transport().write("*TRG")
