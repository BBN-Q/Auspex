# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Labbrick']

from .instrument import Instrument, MetaInstrument
from auspex.log import logger
import time
import os
from unittest.mock import MagicMock
import numpy as np
import ctypes
import cffi

class MakeSettersGetters(MetaInstrument):
    def __init__(self, name, bases, dct):
        super(MakeSettersGetters, self).__init__(name, bases, dct)

        for k,v in dct.items():
            if isinstance(v, property):
                logger.debug("Adding '%s' command to Labbrick", k)
                setattr(self, 'set_'+k, v.fset)
                setattr(self, 'get_'+k, v.fget)

class Labbrick(Instrument, metaclass=MakeSettersGetters):
    """Vaunix Lab Brick microwave source"""
    instrument_type = "Microwave Source"

    STATUS_INVALID_DEVID     = 0x80000000 # MSB is set if the device ID is invalid
    STATUS_DEV_CONNECTED     = 0x00000001 # LSB is set if a device is connected
    STATUS_DEV_OPENED        = 0x00000002 # set if the device is opened
    STATUS_SWP_ACTIVE        = 0x00000004 # set if the device is sweeping
    STATUS_SWP_UP            = 0x00000008 # set if the device is sweeping up in frequency
    STATUS_SWP_REPEAT        = 0x00000010 # set if the device is in continuous sweep mode
    STATUS_SWP_BIDIRECTIONAL = 0x00000020 # set if the device is in bidirectional sweep mode
    STATUS_PLL_LOCKED        = 0x00000040 # set if the PLL lock status is TRUE (both PLL's are locked)
    STATUS_FAST_PULSE_OPTION = 0x00000080 # set if the fast pulse mode option is installed

    def __init__(self, resource_name=None, name="Unlabeled Lab Brick"):
        self.name = name
        self.resource_name = resource_name
        self.previous_num_devices = None
        self.dev_ids = None
        try:
            self.ffi = cffi.FFI()
            path = os.path.realpath(__file__)
            with open(os.path.join(os.path.dirname(path),"vnx_LMS_api_python.h")) as fid:
                self.ffi.cdef(fid.read())
            self._lib = self.ffi.dlopen("vnx_fmsynth.dll")
        except:
            logger.warning("Could not find the Lab Brick driver.")
            self._lib = MagicMock()

    def enumerate(self):
        if self.previous_num_devices is None:
            self.previous_num_devices = 0
        if self.dev_ids is None:
            num_devices = self._lib.fnLMS_GetNumDevices()
            dev_ids = self.ffi.new("unsigned int[]", [0 for i in range(num_devices)])
            self._lib.fnLMS_GetDevInfo(dev_ids)
            dev_from_serial_nums = {self._lib.fnLMS_GetSerialNumber(d): d for d in dev_ids}
            self.dev_ids = [d for d in dev_ids]
            self.previous_num_devices = num_devices
        return dev_from_serial_nums

    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name
        self._lib.fnLMS_SetTestMode(False)
        self.device_id = self.enumerate()[int(self.resource_name)]

        status = self._lib.fnLMS_InitDevice(self.device_id)
        if status != 0:
            logger.warning('Could not open Lab Brick device with id: %d, returned error %d', self.device_id, status)

        self.max_power = self._lib.fnLMS_GetMaxPwr(self.device_id) / 4.0
        self.min_power = self._lib.fnLMS_GetMinPwr(self.device_id) / 4.0
        self.max_freq = self._lib.fnLMS_GetMaxFreq(self.device_id) / 1.0e8
        self.min_freq = self._lib.fnLMS_GetMinFreq(self.device_id) / 1.0e8

    def disconnect(self):
        status = self._lib.fnLMS_CloseDevice(self.device_id)
        if status != 0:
            logger.warning('Could not close Lab Brick device with id: %d, returned error %d', self.device_id, status)

    @property
    def frequency(self):
        return self._lib.fnLMS_GetFrequency(self.device_id) * 10 # Convert from tens of Hz to Hz
    @frequency.setter
    def frequency(self, value):
        self._lib.fnLMS_SetFrequency(self.device_id, int(value * 0.1)) # Convert to tens of Hz from Hz

    @property
    def power(self):
        atten = self._lib.fnLMS_GetPowerLevel(self.device_id) * 0.25 # Convert from 0.25 dB
        return self.max_power - atten
    @power.setter
    def power(self, value):
        if value > self.max_power:
            value = self.max_power
            logger.warning('Lab Brick power out of range')
        elif value < self.min_power:
            value = self.min_power
            logger.warning('Lab Brick power out of range')
        self._lib.fnLMS_SetPowerLevel(self.device_id, int(value * 4)) # Convert to 0.25 dB
