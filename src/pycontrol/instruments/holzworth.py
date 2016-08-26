# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.instrument import Instrument

class HS9000(Instrument):
    """Holzworth HS9000 microwave source"""
    instrument_type = "Digitizer"

    def get_alc(self):
        return lib.get_alc()
    def set_alc(self, value):
        lib.set_alc(value)
    @property
    def alc(self):
        return self.get_alc()
    @alc.setter
    def alc(self, value):
        self.set_alc(value)

    def get_deviceName(self):
        return lib.get_deviceName()
    def set_deviceName(self, value):
        lib.set_deviceName(value)
    @property
    def deviceName(self):
        return self.get_deviceName()
    @deviceName.setter
    def deviceName(self, value):
        self.set_deviceName(value)

    def get_frequency(self):
        return lib.get_frequency()
    def set_frequency(self, value):
        lib.set_frequency(value)
    @property
    def frequency(self):
        return self.get_frequency()
    @frequency.setter
    def frequency(self, value):
        self.set_frequency(value)

    def get_gateBuffer(self):
        return lib.get_gateBuffer()
    def set_gateBuffer(self, value):
        lib.set_gateBuffer(value)
    @property
    def gateBuffer(self):
        return self.get_gateBuffer()
    @gateBuffer.setter
    def gateBuffer(self, value):
        self.set_gateBuffer(value)

    def get_gateDelay(self):
        return lib.get_gateDelay()
    def set_gateDelay(self, value):
        lib.set_gateDelay(value)
    @property
    def gateDelay(self):
        return self.get_gateDelay()
    @gateDelay.setter
    def gateDelay(self, value):
        self.set_gateDelay(value)

    def get_gateMinWidth(self):
        return lib.get_gateMinWidth()
    def set_gateMinWidth(self, value):
        lib.set_gateMinWidth(value)
    @property
    def gateMinWidth(self):
        return self.get_gateMinWidth()
    @gateMinWidth.setter
    def gateMinWidth(self, value):
        self.set_gateMinWidth(value)

    def get_mod(self):
        return lib.get_mod()
    def set_mod(self, value):
        lib.set_mod(value)
    @property
    def mod(self):
        return self.get_mod()
    @mod.setter
    def mod(self, value):
        self.set_mod(value)

    def get_output(self):
        return lib.get_output()
    def set_output(self, value):
        lib.set_output(value)
    @property
    def output(self):
        return self.get_output()
    @output.setter
    def output(self, value):
        self.set_output(value)

    def get_power(self):
        return lib.get_power()
    def set_power(self, value):
        lib.set_power(value)
    @property
    def power(self):
        return self.get_power()
    @power.setter
    def power(self, value):
        self.set_power(value)

    def get_pulse(self):
        return lib.get_pulse()
    def set_pulse(self, value):
        lib.set_pulse(value)
    @property
    def pulse(self):
        return self.get_pulse()
    @pulse.setter
    def pulse(self, value):
        self.set_pulse(value)

    def get_pulseSource(self):
        return lib.get_pulseSource()
    def set_pulseSource(self, value):
        lib.set_pulseSource(value)
    @property
    def pulseSource(self):
        return self.get_pulseSource()
    @pulseSource.setter
    def pulseSource(self, value):
        self.set_pulseSource(value)


    def __init__(self, resource_name, *args, **kwargs):
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        #super(HS9000, self).__init__(resource_name, *args, **kwargs)
        self.name = name
        self.resource_name = resource_name
        self._freeze()

      #     "address": "HS9004A-393-2", 
      # "alc": false, 
      # "deviceName": "HolzworthHS9000", 
      # "frequency": 5.0471484337709995, 
      # "gateBuffer": 4e-08, 
      # "gateDelay": 1e-08, 
      # "gateMinWidth": 1e-07, 
      # "mod": false, 
      # "output": true, 
      # "power": 14.5, 
      # "pulse": false, 
      # "pulseSource": "External"