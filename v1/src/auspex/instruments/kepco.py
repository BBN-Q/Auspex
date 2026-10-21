# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['BOP2020M']

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, RampCommand

class BOP2020M(SCPIInstrument):
    """For controlling the BOP2020M power supply via GPIB interface card"""
    output  = StringCommand(scpi_string="OUTPUT", value_map={True: '1', False: '0'})
    current = RampCommand(increment=0.1, pause=20e-3, get_string=":CURR?", set_string=":CURR:LEV:IMM {:g}", value_range=(-20,20))
    voltage = RampCommand(increment=0.1, pause=20e-3, get_string=":VOLT?", set_string=":VOLT:LEV:IMM {:g}", value_range=(-20,20))
    mode    = StringCommand(scpi_string="FUNC:MODE", value_map={'voltage': "VOLT", 'current': "CURR"})

    def __init__(self, name, resource_name, mode='current', **kwargs):
        super(BOP2020M, self).__init__(name, resource_name, **kwargs)
        self.name = "BOP2020M power supply"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.read_termination = u"\n"
        self.mode = 'current'
        self.interface.write('VOLT MAX')
        self.output = True

    def shutdown(self):
        if self.output:
            if self.current != 0.0:
                for i in np.linspace(self.current, 0.0, 20):
                    self.current = i
            if self.voltage != 0.0:
                for v in np.linspace(self.voltage, 0.0, 20):
                    self.voltage = v
            self.output = False
