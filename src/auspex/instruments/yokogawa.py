# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand

class GS200(SCPIInstrument):
    """GS200 Current source"""

    function           = StringCommand(scpi_string=":source:function",
                          value_map={"current": "CURR", "voltage": "VOLT"})
    level              = FloatCommand(scpi_string=":source:level", aliases=['value'])
    protection_volts   = FloatCommand(scpi_string=":source:protection:voltage")
    protection_current = FloatCommand(scpi_string=":source:protection:current")
    sense              = StringCommand(scpi_string=":sense:state", value_map={True: "1", False: "0"})
    output             = StringCommand(scpi_string=":output:state", value_map={True: "1", False: "0"})
    sense_value        = FloatCommand(get_string=":fetch?")
    averaging_nplc     = IntCommand(scpi_string=":sense:nplc") # Number of power level cycles (60Hz)

    def __init__(self, resource_name, *args, **kwargs):
        super(GS200, self).__init__(resource_name, *args, **kwargs)
        self.interface.write(":sense:trigger immediate")
        self.interface._resource.read_termination = "\n"
