# Copyright 2020 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['DSInstrumentsSG1200']

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, RampCommand, BoolCommand

class DSInstrumentsSG1200(SCPIInstrument):
    """DSInstruments SG1200 Signal Geneator"""
    instrument_type = "Microwave Source"
    bool_map = {'ON':1, 'OFF':0}
    bool_map_inv = {1:'ON', 0:'OFF'}
    output  = BoolCommand(scpi_string="OUTP:STAT", value_map={True: "ON", False: "OFF"})

    def __init__(self, resource_name=None, *args, **kwargs):
        super(DSInstrumentsSG1200, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type="VISA"):
        if resource_name is not None:
            self.resource_name = resource_name
        super(DSInstrumentsSG1200, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\r\n"
        self.interface._resource.write_termination = u"\r\n"
        self.interface._resource.timeout = 100
        self.interface._resource.baud_rate = 115200

    @property
    def frequency(self):
        return float(self.interface.query("FREQ:CW?").replace('HZ',''))
    @frequency.setter
    def frequency(self, value):
        self.interface.write(f'FREQ:CW {value}HZ') # query times out

    @property
    def power(self):
        return float(self.interface.query("POWER?").replace('dBm',''))
    @power.setter
    def power(self, value):
        if value % 0.5:
            value = round(value*2)/2
            logger.info(f'Set power rounded to {value} dBm')
        self.interface.write(f'POWER {value}dBm') # query times out

    @property
    #TODO: calibrate
    def vernier(self):
        return float(self.interface.query("VERNIER?"))
    @vernier.setter
    def vernier(self, value):
        self.interface.write(f'VERNIER {value}')

    @property
    def internal_ref(self):
        return self.bool_map[self.interface.query('*INTREF?')]
    @internal_ref.setter
    def internal_ref(self, value):
        value = int(value)
        self.interface.write(f'*INTERNALREF {value}')

    def detect_ext_ref(self):
        return self.bool_map[self.interface.query('*EXTREF?')]

    def set_buzzer(self, value):
        return self.interface.write(f'*BUZZER {self.bool_map_inv[value]}')

    def set_display(self, value):
        return self.interface.write(f'*DISPLAY {self.bool_map_inv[value]}')
