# Copyright 2016 Raytheon BBN Technologies
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Berkeley Nucleonics VISA source drivers ported from V1."""

import time
import warnings

from .core import (SCPIInstrument, FloatCommand, BoolCommand, StringCommand,
                   experimental, operation)
from ..kinds import InstrumentKind

__all__ = ["BNC845"]


@experimental
class BNC845(SCPIInstrument):
    """BNC845-M source; connection leaves RF output and reference unchanged.

    Pulse, modulation and ALC use active-low boolean encodings on this model.
    These intentionally differ from the RF output encoding.
    """

    KIND = InstrumentKind.MICROWAVE_SOURCE
    IDN_STRING = "BNC845"
    VISA_IPV4_DEVICE = "inst0::INSTR"
    frequency = FloatCommand(scpi_string="SOURCE:FREQUENCY:FIXED")
    power = FloatCommand(scpi_string="SOURCE:POWER:LEVEL:IMMEDIATE:AMPLITUDE")
    output = BoolCommand(scpi_string="OUTPUT:STATE", value_map={True: "1", False: "0"})
    pulse = StringCommand(scpi_string="PULSE", value_map={True: "0", False: "1"})
    mod = StringCommand(scpi_string="MOD", value_map={True: "0", False: "1"})
    alc = StringCommand(scpi_string="SOURCE:POWER:ALC ", value_map={True: "0", False: "1"})
    pulse_source = StringCommand(scpi_string=":PULSE:SOUR",
                                 value_map={"INTERNAL": "INT", "EXTERNAL": "EXT"})
    freq_source = StringCommand(scpi_string=":FREQ:SOUR",
                                value_map={"INTERNAL": "INT", "EXTERNAL": "EXT"})

    def connect(self, resource_name=None):
        """Open transport only; reference selection and RF enabling are explicit."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\n"
            self.interface.write_termination = "\n"
        except Exception:
            self.disconnect()
            raise

    @operation
    def configure_external_reference(self):
        """Select external 10 MHz reference and poll lock; do not enable RF.

        Retain V1's ten half-second lock checks, warning rather than failing if
        unlocked. Return the observed lock state, not a claim of RF readiness.
        """
        io = self._transport()
        io.write("SOURCE:ROSC:EXT:FREQ 10E6")
        io.write("SOUR:ROSC:SOUR EXT")
        for _ in range(10):
            if io.query("SOURCE:ROSC:LOCKED?").strip() == "1":
                return True
            time.sleep(0.5)
        warnings.warn(f"BNC845 at {self.resource_name} is unlocked.", UserWarning, stacklevel=2)
        return False
