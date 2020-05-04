# Copyright 2020 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# Drivers for controling the DS Instruments puresine rf signal generators

__all__ = ['DSInstrumentsSG12000', 'DSInstrumentsSG12000Pro']

from auspex.log import logger
from .instrument import SCPIInstrument, MetaInstrument, StringCommand, FloatCommand, IntCommand, RampCommand, BoolCommand

class MakeSettersGetters(MetaInstrument):
    def __init__(self, name, bases, dct):
        super(MakeSettersGetters, self).__init__(name, bases, dct)

        for k,v in dct.items():
            if isinstance(v, property):
                logger.debug("Adding '%s' command to DS source", k)
                setattr(self, 'set_'+k, v.fset)
                setattr(self, 'get_'+k, v.fget)

class DSInstruments(SCPIInstrument,metaclass=MakeSettersGetters):
    """Base class for DSInstruments Signal Generators"""
    instrument_type = "Microwave Source"
    bool_map = {'ON':1, 'OFF':0}
    bool_map_inv = {1:'ON', 0:'OFF'}
    output  = BoolCommand(scpi_string="OUTP:STAT", value_map={True: "ON", False: "OFF"})

    def __init__(self, resource_name=None, *args, **kwargs):
        super(DSInstruments, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type="VISA"):
        if resource_name is not None:
            self.resource_name = resource_name
        super(DSInstruments, self).connect(resource_name=self.resource_name, interface_type=interface_type)
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

    def set_buzzer(self, value):
        return self.interface.write(f'*BUZZER {self.bool_map_inv[value]}')

    def set_display(self, value):
        return self.interface.write(f'*DISPLAY {self.bool_map_inv[value]}')

    def reset(self):
        return self.interface.write("*RST")

    def ping(self):
        return self.interface.query("*PING?") #Should return PONG!

    def save(self):
        self.interface.write("*SAVESTATE")

    def usbvoltage(self):
        return self.interface.query("*SYSVOLTS?")

    @property
    def device_name(self):
        return self.interface.query("*UNITNAME?")
    @device_name.setter
    def device_name(self,value):
        self.interface.write(f"*UNITNAME {value}")

    def errors(self):
        return self.interface.query("SYST:ERR?")

    def cls(self):
        self.interface.write("*CLS")

    def debug(self):
        self.interface.query("SYST:DBG?")

class DSInstrumentsSG12000(DSInstruments,metaclass=MakeSettersGetters):
    def __init__(self, resource_name=None, *args, **kwargs):
        super(DSInstrumentsSG12000, self).__init__(resource_name, *args, **kwargs)

    @property
    def reference(self):
        return self.bool_map[self.interface.query('*INTREF?')]
    @reference.setter
    def reference(self, value):
        if value not in ["0","1","A"]:
            print("Only valid values are 0, 1, A")
            raise
        self.interface.write(f'*INTERNALREF {value}')

    def set_low_power_mode(self, value): # reduces RF output by 4-7 dB
        return self.interface.write(f'*LPMODE {self.bool_map_inv[value]}')

class DSInstrumentsSG12000Pro(DSInstruments,metaclass=MakeSettersGetters):
    def __init__(self, resource_name=None, *args, **kwargs):
        super(DSInstrumentsSG12000Pro, self).__init__(resource_name, *args, **kwargs)

    @property
    def reference(self):
        return self.interface.query('SYSREF?')
    @reference.setter
    def reference(self, value):
        if value not in ["INT","EXT","AUTO","FREE", "OFF"]:
            print("Only valid values are INT, EXT, AUTO, FREE", "OFF") #OFF disables the internal 100MHz vcxo, has lower noise, requires external source
            print("Setting default to ext")
            value = "EXT"
        self.interface.write(f'SYSREF {value}')

    def update_reference(self):
        self.interface.write("SYSREF UPDATE")

    def detect_ext_ref(self):
        return self.interface.query('SYSREF STATUS?')

    def temp(self): #Is this working?
        return self.interface.query("*TEMPC?")
