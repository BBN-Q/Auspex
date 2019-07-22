# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['YokogawaGS200']

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, RampCommand, BoolCommand

class YokogawaGS200(SCPIInstrument):
    """YokogawaGS200 Current source"""
    instrument_type = "Current source"

    mode               = StringCommand(scpi_string=":source:function",
                          value_map={"current": "CURR", "voltage": "VOLT"})
    level              = FloatCommand(scpi_string=":source:level", aliases=['value'])
    output_range       = FloatCommand(scpi_string=":source:range")
    protection_volts   = FloatCommand(scpi_string=":source:protection:voltage")
    protection_current = FloatCommand(scpi_string=":source:protection:current")
    sense              = BoolCommand(scpi_string=":sense:state", value_map={True: "1", False: "0"})
    output             = BoolCommand(scpi_string=":output:state", value_map={True: "1", False: "0"})
    sense_value        = FloatCommand(get_string=":fetch?")
    averaging_nplc     = IntCommand(scpi_string=":sense:nplc") # Number of power level cycles (60Hz)
    ramp               = RampCommand(increment=1e-4, pause=20e-3, scpi_string=":source:level", value_range=(-100e-3,100e-3))

    def __init__(self, resource_name=None, *args, **kwargs):
        super(YokogawaGS200, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type="VISA"):
        if resource_name is not None:
            self.resource_name = resource_name

        super(YokogawaGS200, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface.write(":sense:trigger immediate")
        self.interface._resource.read_termination = "\n"
