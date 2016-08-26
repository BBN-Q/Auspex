# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.instrument import Instrument
import ctypes

class HS9000(Instrument):
    """Holzworth HS9000 microwave source"""
    instrument_type = "Microwave Source"
    _lib = ctypes.CDLL("HolzworthMulti64.dll")

    def __init__(self, resource_name):
        self.name = "Holzworth HS9000"
        self.resource_name = resource_name
        
        # parse resource_name: expecting something like "HS9004A-009-1"
        self.model, self.serial, self.chan = resource_name.split("-")
        self.resource_name = resource_name

        self._lib.usbCommWrite.restype = ctypes.c_char_p

    def query(self, scpi_string):
        return self._lib.usbCommWrite(self.resource_name.encode('ascii'), scpi_string.encode('ascii')).decode('ascii')

    @property
    def frequency(self):
        v = self.query(":CH{}:FREQ?".format(self.chan))
        return float(v.split()[0])*1e6
    @frequency.setter
    def frequency(self, value):
        self.query(":CH{}:FREQ:{} GHz".format(self.chan, value*1e-9))

    def __del__(self):
        pass

    def set_all(self, thing):
        pass

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