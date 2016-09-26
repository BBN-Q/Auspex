# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments.instrument import Instrument, MetaInstrument
from auspex.log import logger
from unittest.mock import MagicMock
import ctypes

class MakeSettersGetters(MetaInstrument):
    def __init__(self, name, bases, dct):
        super(MakeSettersGetters, self).__init__(name, bases, dct)

        for k,v in dct.items():
            if isinstance(v, property):
                logger.debug("Adding '%s' command to Holzworth", k)
                setattr(self, 'set_'+k, v.fset)
                setattr(self, 'get_'+k, v.fget)

class HS9000(Instrument, metaclass=MakeSettersGetters):
    """Holzworth HS9000 microwave source"""
    instrument_type = "Microwave Source"

    def __init__(self, resource_name, name="Unlabeled Holzworth HS9000"):
        self.name = name
        self.resource_name = resource_name
        try:
            self._lib = ctypes.CDLL("HolzworthMulti64.dll")
        except:
            logger.warning("Could not find APS2 python driver.")
            self._lib = MagicMock()
        
        # parse resource_name: expecting something like "HS9004A-009-1"
        self.model, self.serial, self.chan = resource_name.split("-")
        self.resource_name = resource_name

        self._lib.usbCommWrite.restype = ctypes.c_char_p

    def query(self, scpi_string):
        return self._lib.usbCommWrite(self.resource_name.encode('ascii'), scpi_string.encode('ascii')).decode('ascii')

    def ch_query(self, scpi_string):
        chan_string = ":CH{}".format(self.chan)
        scpi_string = chan_string + scpi_string
        return self._lib.usbCommWrite(self.resource_name.encode('ascii'), scpi_string.encode('ascii')).decode('ascii')

    @property
    def frequency(self):
        v = self.ch_query(":FREQ?")
        return float(v.split()[0])*1e6
    @frequency.setter
    def frequency(self, value):
        self.ch_query(":FREQ:{} GHz".format(value*1e-9))

    @property
    def power(self):
        v = self.ch_query(":PWR?")
        return float(v.split()[0])
    @power.setter
    def power(self, value):
        self.ch_query(":PWR:{} dBm".format(value))

    @property
    def phase(self):
        v = self.ch_query(":PHASE?")
        return float(v.split()[0])
    @phase.setter
    def phase(self, value):
        self.ch_query(":PHASE:{} deg".format(value))

    @property
    def output(self):
        v = self.ch_query(":PWR:RF?")
        return bool(v.split()[0])
    @output.setter
    def output(self, value):
        if value:
            self.ch_query(":PWR:RF:ON")
        else:
            self.ch_query(":PWR:RF:OFF")

    @property
    def reference(self):
        v = self.query(":REF:STATUS?")
        return float(v.split()[0])*1e6
    @reference.setter
    def reference(self, value):
        ref_opts = ["INT", "10MHz", "100MHz"]
        if value in ref_opts:
            if value == "INT":
                self.query(":REF:INT:100MHz")
            else:
                self.query(":REF:EXT:{}".format(value))
        else:
            raise ValueError("Reference must be one of {}.".format(ref_opts))

    def __del__(self):
        self._lib.close_all()
