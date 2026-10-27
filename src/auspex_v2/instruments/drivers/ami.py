# Copyright 2016 Raytheon BBN Technologies
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""American Magnetics VISA supply programmer ported from V1."""

import time

from .core import (SCPIInstrument, StringCommand, FloatCommand, IntCommand,
                   InstrumentError, experimental, operation)
from ..kinds import InstrumentKind

__all__ = ["AMI430"]


@experimental
class AMI430(SCPIInstrument):
    """AMI430 magnet supply programmer; field units must be selected explicitly.

    V1 current/field bounds are retained, not inferred from the attached magnet.
    Blocking set_field uses the instrument's existing units and ramp configuration.
    """

    KIND = InstrumentKind.MAGNET_SUPPLY
    VISA_IPV4_DEVICE = "7180::SOCKET"
    SUPPLY_TYPES = [
        "AMI 12100PS", "AMI 12200PS", "AMI 4Q05100PS", "AMI 4Q06125PS", "AMI 4Q06250PS",
        "AMI 4Q12125PS", "AMI 10100PS", "AMI 10200PS", "HP 6260B", "Kepco BOP 20-5M",
        "Kepco BOP 20-10M", "Xantrex XFR 7.5-140", "Custom", "AMI Model 05100PS-430-601",
        "AMI Model 05200PS-430-601", "AMI Model 05300PS-430-601", "AMI Model 05400PS-430-601",
        "AMI Model 05500PS-430-601",
    ]
    RAMPING_STATES = [
        "RAMPING to target field/current", "HOLDING at the target field/current", "PAUSED",
        "Ramping in MANUAL UP mode", "Ramping in MANUAL DOWN mode", "ZEROING CURRENT (in progress)",
        "Quench detected", "At ZERO current", "Heating persistent switch", "Cooling persistent switch",
    ]
    supply_type = StringCommand(get_string="SUPPly:TYPE?",
                                value_map={v: str(i) for i, v in enumerate(SUPPLY_TYPES)})
    voltage_min = FloatCommand(get_string="SUPPly:VOLTage:MINimum?")
    voltage_max = FloatCommand(get_string="SUPPly:VOLTage:MAXimum?")
    current_min = FloatCommand(get_string="SUPPly:CURRent:MINimum?")
    current_max = FloatCommand(get_string="SUPPly:CURRent:MAXimum?")
    current_limit = FloatCommand(set_string="CONFigure:CURRent:LIMit {:f}", get_string="CURRent:LIMit?")
    current_rating = FloatCommand(set_string="CONFigure:CURRent:RATING {:f}", get_string="CURRent:RATING?")
    stability = FloatCommand(set_string="CONFigure:STABility {:f}", get_string="STABility?", value_range=(0, 100))
    coil_const = FloatCommand(set_string="CONFigure:COILconst {:f}", get_string="COILconst?")
    persistent_switch = StringCommand(set_string="CONFigure:PSwitch {}", get_string="PSwitch:INSTalled?",
                                      value_map={False: "0", True: "1"})
    absorber = StringCommand(set_string="CONFigure:ABsorber {}", get_string="ABsorber?",
                             value_map={False: "0", True: "1"})
    field_units = StringCommand(set_string="CONFigure:FIELD:UNITS {}", get_string="FIELD:UNITS?",
                                value_map={"kG": "0", "T": "1"})
    voltage_limit = FloatCommand(set_string="CONFigure:VOLTage:LIMit {:f}", get_string="VOLTage:LIMit?")
    current_target = FloatCommand(set_string="CONFigure:CURRent:TARGet {:f}", get_string="CURRent:TARGet?",
                                  value_range=(-44.2, 44.2))
    field_target = FloatCommand(set_string="CONFigure:FIELD:TARGet {:f}", get_string="FIELD:TARGet?",
                                value_range=(-0.4, 0.4))
    ramp_num_segments = IntCommand(set_string="CONFigure:RAMP:RATE:SEGments {:d}",
                                   get_string="RAMP:RATE:SEGments?", value_range=(1, 10))
    ramp_rate_units = StringCommand(set_string="CONFigure:RAMP:RATE:UNITS {}", get_string="RAMP:RATE:UNITS?",
                                    value_map={"seconds": "0", "minutes": "1"})
    voltage = FloatCommand(get_string="VOLTage:SUPPly?")
    current_magnet = FloatCommand(get_string="CURRent:MAGnet?")
    current_supply = FloatCommand(get_string="CURRent:SUPPly?")
    field = FloatCommand(get_string="FIELD:MAGnet?")
    inductance = FloatCommand(get_string="INDuctance?")
    ramping_state = StringCommand(get_string="STATE?",
                                  value_map={v: str(i + 1) for i, v in enumerate(RAMPING_STATES)})

    def connect(self, resource_name=None):
        """Read the IP-interface greeting without changing units or magnet settings."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\r\n"
            if (self.interface.read() != "American Magnetics Model 430 IP Interface"
                    or self.interface.read() != "Hello."):
                raise InstrumentError("Unexpected AMI430 IP-interface greeting")
        except Exception:
            self.disconnect()
            raise

    @operation
    def ramp(self):
        """Start automatic ramping to the configured target."""
        self._transport().write("RAMP")

    @operation
    def pause(self):
        self._transport().write("PAUSE")

    @operation
    def ramp_up(self):
        """Start manual upward ramping until the configured current limit."""
        self._transport().write("INCR")

    @operation
    def ramp_down(self):
        """Start manual downward ramping toward the configured limit or zero."""
        self._transport().write("DECR")

    @operation
    def zero(self):
        """Initiate the programmer's zero-current ramp, not an immediate output-off."""
        self._transport().write("ZERO")

    @operation
    def set_field(self, value):
        """Set target and ramp, blocking until HOLDING as in V1 (no total timeout)."""
        self.field_target = value
        self.ramp()
        while self.ramping_state != "HOLDING at the target field/current":
            time.sleep(0.1)
