# Copyright 2016 Raytheon BBN Technologies
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Kepco VISA power supplies ported from V1."""

import numpy as np

from .core import SCPIInstrument, StringCommand, RampCommand, experimental, operation
from ..kinds import InstrumentKind

__all__ = ["BOP2020M"]


@experimental
class BOP2020M(SCPIInstrument):
    """BOP2020M supply through its GPIB card; no automatic output enabling."""

    KIND = InstrumentKind.DC_SOURCE
    output = StringCommand(scpi_string="OUTPUT", value_map={True: "1", False: "0"})
    current = RampCommand(increment=0.1, pause=0.02, get_string=":CURR?",
                          set_string=":CURR:LEV:IMM {:g}", value_range=(-20, 20))
    voltage = RampCommand(increment=0.1, pause=0.02, get_string=":VOLT?",
                          set_string=":VOLT:LEV:IMM {:g}", value_range=(-20, 20))
    mode = StringCommand(scpi_string="FUNC:MODE",
                         value_map={"voltage": "VOLT", "current": "CURR"})

    def connect(self, resource_name=None):
        """Open transport without changing the source mode, limits or output."""
        super().connect(resource_name)
        try:
            self.interface.write_termination = "\n"
            self.interface.read_termination = "\n"
        except Exception:
            self.disconnect()
            raise

    @operation
    def configure_current_mode(self):
        """Explicitly select current mode and maximum voltage; output is unchanged."""
        self.mode = "current"
        self._transport().write("VOLT MAX")

    @operation
    def shutdown(self):
        """Retain V1's current/voltage ramps to zero before disabling output.

        This issues commands, not an independent confirmation of a safe state.
        Transport failures propagate without retrying uncertain writes.
        """
        if self.output:
            current = self.current
            if current != 0:
                for value in np.linspace(current, 0, 20):
                    self.current = value
            voltage = self.voltage
            if voltage != 0:
                for value in np.linspace(voltage, 0, 20):
                    self.voltage = value
            self.output = False
