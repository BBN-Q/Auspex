# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['HolzworthHS9000']

from .instrument import Instrument, MetaInstrument
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

class HolzworthHS9000(Instrument, metaclass=MakeSettersGetters):
    """Holzworth HS9000 microwave source"""
    instrument_type = "Microwave Source"

    def __init__(self, resource_name=None, name="Unlabeled Holzworth HS9000"):
        self.name = name
        self.resource_name = resource_name
        try:
            self._lib = ctypes.CDLL("HolzworthMulti.dll")
            self.fake_holz = False
        except:
            logger.warning("Could not find the Holzworth driver.")
            self._lib = MagicMock()
            self.fake_holz = True

        self._lib.usbCommWrite.restype = ctypes.c_char_p
        self._lib.openDevice.restype = ctypes.c_int

    @classmethod
    def enumerate(cls):
        try:
            lib = ctypes.CDLL("HolzworthMulti.dll")
        except:
            logger.error("Could not find the Holzworth driver.")
            return
        lib.getAttachedDevices.restype = ctypes.c_char_p
        devices = lib.getAttachedDevices()
        return devices.decode('ascii').split(',')

    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name
        # parse resource_name: expecting something like "HS9004A-009-1"
        model, serial, self.chan = self.resource_name.split("-")
        self.serial = model + '-' + serial
        success = self._lib.openDevice(self.serial.encode('ascii'))
        if success != 0:
            logger.debug("Could not open Holzworth at address: {}, might already be open on another channel.".format(self.serial))
        # read frequency and power ranges
        self.fmin = float((self.ch_query(":FREQ:MIN?")).split()[0]) * 1e6 #Hz
        self.fmax = float((self.ch_query(":FREQ:MAX?")).split()[0]) * 1e6 #Hz
        self.pmin = float((self.ch_query(":PWR:MIN?")).split()[0]) #dBm
        self.pmax = float((self.ch_query(":PWR:MAX?")).split()[0]) #dBm

    def ref_query(self, scpi_string):
        serial = self.serial + '-R'
        return self._lib.usbCommWrite(serial.encode('ascii'), scpi_string.encode('ascii')).decode('ascii')

    def ch_query(self, scpi_string):
        chan_string = ":CH{}".format(self.chan)
        scpi_string = chan_string + scpi_string
        return self._lib.usbCommWrite(self.resource_name.encode('ascii'), scpi_string.encode('ascii')).decode('ascii')

    @property
    def frequency(self):
        v = self.ch_query(":FREQ?")
        return float(v.split()[0]) * 1e6
    @frequency.setter
    def frequency(self, value):
        if not self.fake_holz:
            if self.fmin <= value <= self.fmax:
                # WARNING!!! The Holzworth might blow up if you ask for >12 digits of precision here
                self.ch_query(":FREQ:{:.12g} Hz".format(value))
            else:
                err_msg = "The value {} GHz is outside of the allowable range {}-{} GHz specified for instrument '{}'.".format(value*1e-9, self.fmin*1e-9, self.fmax*1e-9, self.name)
                raise ValueError(err_msg)

    @property
    def power(self):
        v = self.ch_query(":PWR?")
        return float(v.split()[0])
    @power.setter
    def power(self, value):
        if not self.fake_holz:
            if self.pmin <= value <= self.pmax:
                self.ch_query(":PWR:{} dBm".format(value))
            else:
                err_msg = "The value {} dBm is outside of the allowable range {}-{} dBm specified for instrument '{}'.".format(value, self.pmin, self.pmax, self.name)
                raise ValueError(err_msg)

    @property
    def phase(self):
        v = self.ch_query(":PHASE?")
        return float(v.split()[0])
    @phase.setter
    def phase(self, value):
        self.ch_query(":PHASE:{} deg".format(value % 360))

    @property
    def output(self):
        v = self.ch_query(":PWR:RF?")
        if v == 'ON':
            return True
        else:
            return False
    @output.setter
    def output(self, value):
        if value:
            self.ch_query(":PWR:RF:ON")
        else:
            self.ch_query(":PWR:RF:OFF")

    @property
    def reference(self):
        v = self.ref_query(":REF:STATUS?")
        return v
    @reference.setter
    def reference(self, value):
        ref_opts = ["INT", "10MHz", "100MHz"]
        if value in ref_opts:
            if value == "INT":
                self.ref_query(":REF:INT:100MHz")
            else:
                self.ref_query(":REF:EXT:{}".format(value))
        else:
            raise ValueError("Reference must be one of {}.".format(ref_opts))

    def __del__(self):
        self._lib.close_all()
