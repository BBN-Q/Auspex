# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['HolzworthHS9000']

from .instrument import Instrument, MetaInstrument
import usb
from auspex.log import logger
from auspex import config
from unittest.mock import MagicMock
import ctypes

class HolzworthDevice(object):

    TIMEOUT = 100

    def __init__(self, device):
        super(HolzworthDevice, self).__init__()

        if not isinstance(device, usb.core.Device):
            raise TypeError("Holzworth device must receive a USB device!")

        self._device = device
        self._device.reset()
        if self._device.is_kernel_driver_active(0):
            self._device.detach_kernel_driver(0)
        self._device.set_configuration()

        if "Holzworth" not in usb.util.get_string(self._device, self._device.iManufacturer):
            raise ValueError("Not a Holzworth!")

        self._e_in = self._device[0][(0,0)][0]
        self._e_out = self._device[0][(0,0)][1]

        self.channels = self.query(":ATTACH?").split(":")[1:-1]
        if not self.channels:
            raise ValueError("Holzworth has no channels!")

        self.serial = self.query(":{}:IDN?".format(self.channels[0])).split(',')[-1]

    def __del__(self):
        if self._device is not None:
            usb.util.dispose_resources(self._device)
        super(HolzworthDevice, self).__del__()

    def write(self, command):
        try:
            wlen = self._e_out.write(command.encode(), timeout=self.TIMEOUT)
            assert wlen == len(command.encode())
        except usb.core.USBError:
            logger.error("Command {} to Holzworth {} timed out!".format(command, self.serial))

    def read(self, nbytes=64):
        try:
            data = elf._e_in.read(nbytes, timeout=self.TIMEOUT)
        except usb.core.USBError:
            logger.error("Read from Holzworth {} timed out!".format(command, self.serial))
        #Strip NULLs from reply and decode
        return bytes(data).partition(b'\0')[0].decode()

    def query(self, command, nbytes=64):
        self.write(command)
        response = self.read(nbytes=nbytes)
        if response == "Invalid Command":
            logger.error("Invalid command {} to Holzworth {}.".format(command, self.serial))
        return response

class HolzworthPythonDriver(object):

    HOLZWORTH_VENDOR_ID = 0x1bb3
    HOLZWORTH_PRODUCT_ID = 0x1001

    def __init__(self):
        super(HolzworthPythonDriver, self).__init__()

        devices = {}
        for dev in usb.core.find(idVendor = self.HOLZWORTH_VENDOR_ID,
                             idProduct = self.HOLZWORTH_PRODUCT_ID,
                             find_all=True):
            holz = HolzworthDevice(dev)
            devices[holz.serial] = holz

        if not devices:
            raise IOError("No Holzworth devices found.")

    def enumerate(self):
        return self.devices.keys()

    def ch_check(serial, channel):
        if serial not in self.devices.keys():
            ValueError("Holzworth {} not connected!".format(serial))
        if channel not in self.devices[serial]:
            ValueError("Holzworth {} does not have channel {}".format(serial, channel))

    def read(self, serial):
        return self.devices[serial].read()

    def write(self, serial, command):
        self.devices[serial].write(command)

    def query(self, serial, command):
        self.devices[serial].query(command)

if config.auspex_dummy_mode:
    fake_holzworth = True
else:
    try:
        holzworth_driver = HolzworthPythonDriver()
        fake_holzworth = False
    except Exception as e:
        logger.warning("Could not connect to Holzworths: {}".format(e))
        if str(e) == "No backend available":
            logger.warning("You may not have the libusb backend: please install it!")
        holzworth_driver = MagicMock()
        fake_holzworth = True

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

    @classmethod
    def enumerate(cls):
        return holzworth_driver.enumerate()

    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name
        # parse resource_name: expecting something like "HS9004A-009-1"
        model, serial, self.chan = self.resource_name.split("-")
        self.serial = model + '-' + serial
        holzworth_driver.ch_check(self.serial, self.chan)

        # read frequency and power ranges
        self.fmin = float((self.ch_query(":FREQ:MIN?")).split()[0]) * 1e6 #Hz
        self.fmax = float((self.ch_query(":FREQ:MAX?")).split()[0]) * 1e6 #Hz
        self.pmin = float((self.ch_query(":PWR:MIN?")).split()[0]) #dBm
        self.pmax = float((self.ch_query(":PWR:MAX?")).split()[0]) #dBm

    def ref_query(self, scpi_string):
        serial = self.serial
        return holzworth_driver.query(self.serial, ":REF{}".format(scpi_string))

    def ch_query(self, scpi_string):
        chan_string = ":CH{}".format(self.chan)
        scpi_string = chan_string + scpi_string
        return holzworth_driver.query(self.serial, scpi_string)

    @property
    def frequency(self):
        v = self.ch_query(":FREQ?")
        return float(v.split()[0]) * 1e6
    @frequency.setter
    def frequency(self, value):
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
        v = self.ref_query(":STATUS?")
        return v
    @reference.setter
    def reference(self, value):
        ref_opts = ["INT", "10MHz", "100MHz"]
        if value in ref_opts:
            if value == "INT":
                self.ref_query(":INT:100MHz")
            else:
                self.ref_query(":EXT:{}".format(value))
        else:
            raise ValueError("Reference must be one of {}.".format(ref_opts))
