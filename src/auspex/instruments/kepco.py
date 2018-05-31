# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['BOP2020M']

from auspex.log import logger
from .instrument import SCPIInstrument, Command, RampCommand

class BOP2020M(SCPIInstrument):
    """For controlling the BOP2020M power supply via GPIB interface card"""
    output  = Command(scpi_string="OUTPUT", value_map={True: '1', False: '0'})
    current = RampCommand(increment=0.1, pause=20e-3, get_string=":CURR?", set_string=":CURR:LEV:IMM {:g}", value_range=(-20,20))
    voltage = RampCommand(increment=0.1, pause=20e-3, get_string=":VOLT?", set_string=":VOLT:LEV:IMM {:g}", value_range=(-20,20))
    mode    = Command(scpi_string="FUNC:MODE", value_map={'voltage': "VOLT", 'current': "CURR"})

    def __init__(self, resource_name, mode='current', **kwargs):
        super(BOP2020M, self).__init__(resource_name, **kwargs)
        self.name = "BOP2020M power supply"

    def connect(self, resource_name=None, interface_type=None):
        super(BOP2020M, self).connect(resource_name=resource_name, interface_type=interface_type)
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.read_termination = u"\n"
        self.interface.write("FUNC:MODE CURR")
        self.interface.write('VOLT MAX')

    def shutdown(self):
        if self.output:
            if self.current != 0.0:
                for i in np.linspace(self.current, 0.0, 20):
                    self.current = i
            if self.voltage != 0.0:
                for v in np.linspace(self.voltage, 0.0, 20):
                    self.voltage = v
            self.output = False
