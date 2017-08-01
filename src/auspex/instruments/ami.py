# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['AMI430']

from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand

import time

class AMI430(SCPIInstrument):
    """AMI430 Power Supply Programmer"""
    instrument_type = "Magnet"

    SUPPLY_TYPES = \
    ["AMI 12100PS", "AMI 12200PS", "AMI 4Q05100PS", "AMI 4Q06125PS", "AMI 4Q06250PS",
    "AMI 4Q12125PS", "AMI 10100PS", "AMI 10200PS", "HP 6260B", "Kepco BOP 20-5M",
    "Kepco BOP 20-10M", "Xantrex XFR 7.5-140", "Custom", "AMI Model 05100PS-430-601",
    "AMI Model 05200PS-430-601", "AMI Model 05300PS-430-601", "AMI Model 05400PS-430-601", "AMI Model 05500PS-430-601"]

    RAMPING_STATES = \
    ["RAMPING to target field/current", "HOLDING at the target field/current", "PAUSED",
    "Ramping in MANUAL UP mode", "Ramping in MANUAL DOWN mode", "ZEROING CURRENT (in progress)",
    "Quench detected", "At ZERO current", "Heating persistent switch", "Cooling persistent switch"]

    #Configure commands
    supply_type = StringCommand(get_string="SUPPly:TYPE?",
        value_map={v:str(ct) for ct,v in enumerate(SUPPLY_TYPES)}) # Supply type",

    voltage_min    = FloatCommand(get_string="SUPPly:VOLTage:MINimum?")
    voltage_max    = FloatCommand(get_string="SUPPly:VOLTage:MAXimum?")

    current_min    = FloatCommand(get_string="SUPPly:CURRent:MINimum?") # Minimum supply current
    current_max    = FloatCommand(get_string="SUPPly:CURRent:MAXimum?") # Maximum supply current
    current_limit  = FloatCommand(set_string="CONFigure:CURRent:LIMit {:f}", get_string="CURRent:LIMit?") # Maximum magnitude of current (A)
    current_rating = FloatCommand(set_string="CONFigure:CURRent:RATING {f}", get_string="CURRent:RATING?") # Magnet current rating (A)
    stability      = FloatCommand(set_string="CONFigure:STABility {:f}", get_string="STABility?", value_range=(0,100)) # Stability setting in percent
    coil_const     = FloatCommand(set_string="CONFigure:COILconst {:f}",get_string="COILconst?") # Field-to-current ratio (kG/A or T/A)

    persistent_switch = StringCommand(set_string="CONFigure:PSwitch {}",
                         get_string="PSwitch:INSTalled?", value_map={False:"0", True:"1"}) # Persistent switch installed (bool)
    absorber          = StringCommand(set_string="CONFigure:ABsorber {}", get_string="ABsorber?",
                         value_map={False:"0", True:"1"})  # Absorber installed (bool)
    field_units       = StringCommand(set_string="CONFigure:FIELD:UNITS {}",
                         get_string="FIELD:UNITS?", value_map={"kG":"0", "T":"1"})

    #Ramp commands
    voltage_limit     = FloatCommand(set_string="CONFigure:VOLTage:LIMit {:f}", get_string="VOLTage:LIMit?") # Ramping voltage limit (V)
    current_target    = FloatCommand(set_string="CONFigure:CURRent:TARGet {:f}",
                         get_string = "CURRent:TARGet?", value_range=(-44.2,44.2))
    field_target      = FloatCommand(set_string="CONFigure:FIELD:TARGet {:f}",
                         get_string = "FIELD:TARGet?", value_range=(-0.4,0.4))
    ramp_num_segments = IntCommand(set_string="CONFigure:RAMP:RATE:SEGments {:d}",
                         get_string = "RAMP:RATE:SEGments?", value_range=(1,10))
    ramp_rate_units   = StringCommand(set_string="CONFigure:RAMP:RATE:UNITS {}",
                         get_string = "RAMP:RATE:UNITS?", value_map={"seconds":"0", "minutes":"1"})

    #Current operating conditions
    voltage        = FloatCommand(get_string="VOLTage:SUPPly?") # Voltage at supply (V)
    current_magnet = FloatCommand(get_string="CURRent:MAGnet?") # Current at magnet
    current_supply = FloatCommand(get_string="CURRent:SUPPly?") # Current at supply
    field          = FloatCommand(get_string="FIELD:MAGnet?") # Calculated magnet field
    inductance     = FloatCommand(get_string="INDuctance?") # Measured inductance (H)
    ramping_state  = StringCommand(get_string="STATE?", value_map={v:str(ct+1) for ct,v in enumerate(RAMPING_STATES)}) # Current ramping state

    def __init__(self, resource_name, *args, **kwargs):
        super(AMI430, self).__init__(resource_name, *args, **kwargs)
        self.name = "American Magnetics Model 430"

    #TODO when we want more than one segment
    # def set_ramp_rate_current(self, segment, current, ) = FloatCommand("Ramp rate for specified segement (A/sec or A/min)", set_string="CONFigure:RAMP:RATE:CURRent {segment:d},{:f}",
    #     get_string="RAMP:RATE:CURRent:{segment:d}?", additional_args=["segment"])

    def connect(self, resource_name=None, interface_type=None):
        if resource_name:
            self.resource_name = resource_name
        if "::7180::SOCKET" not in self.resource_name: #user guide recommends HiSLIP protocol
            self.resource_name += "::7180::SOCKET"
        super(AMI430, self).connect(resource_name=self.resource_name, interface_type=None)
        self.interface._resource.read_termination = u"\r\n"
        #device responds with 'American Magnetics Model 430 IP Interface\r\nHello\r\n' on connect
        assert self.interface.read() == "American Magnetics Model 430 IP Interface"
        assert self.interface.read() == "Hello."

        #Default to T field units
        self.field_units = "T"

    #Ramping state controls
    def ramp(self):
        """
        Places the Model 430 Programmer in automatic ramping mode. The Model
        430 will continue to ramp at the configured ramp rate(s) until the target
        field/current is achieved.
        """
        self.interface.write("RAMP")

    def pause(self):
        """
        Pauses the Model 430 Programmer at the present operating field/current.
        """
        self.interface.write("PAUSE")

    def ramp_up(self):
        """
        Places the Model 430 Programmer in the MANUAL UP ramping mode.
        Ramping continues at the ramp rate until the Current Limit is achieved.
        """
        self.interface.write("INCR")

    def ramp_down(self):
        """
        Places the Model 430 Programmer in the MANUAL DOWN ramping
        mode. Ramping continues at the ramp rate until the Current Limit is
        achieved (or zero current is achieved for unipolar power supplies).
        """
        self.interface.write("DECR")

    def zero(self):
        """
        Places the Model 430 Programmer in ZEROING CURRENT mode.
        Ramping automatically initiates and continues at the ramp rate until the
        power supply output current is less than 0.1% of Imax, at which point the
        AT ZERO status becomes active.
        """
        self.interface.write("ZERO")

    def set_field(self, val):
        """
        Blocking field setter
        """
        self.field_target = val
        self.ramp()
        while self.ramping_state != "HOLDING at the target field/current":
            time.sleep(0.1)
