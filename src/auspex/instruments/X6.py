# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['X6Channel', 'X6']

import time
import socket
import struct
import datetime
import asyncio
import numpy as np
import os

import auspex.globals
from auspex.log import logger
import auspex.config as config
from .instrument import Instrument, DigitizerChannel
from unittest.mock import MagicMock

# Dirty trick to avoid loading libraries when scraping
# This code using quince.
if auspex.globals.auspex_dummy_mode:
    fake_x6 = True
else:
    try:
        import libx6
        fake_x6 = False
    except:
        # logger.warning("Could not load x6 library")
        fake_x6 = True

class X6Channel(DigitizerChannel):
    """Channel for an X6"""

    def __init__(self, settings_dict=None):
        self.stream_type       = "Raw"
        self.if_freq           = 0.0
        self.kernel            = None
        self.kernel_bias       = 0.0
        self.threshold         = 0.0
        self.threshold_invert  = False

        self.phys_channel   = 1
        self.dsp_channel    = 0
        self.channel_tuple  = (1,0,0)

        self.dtype = np.float64

        if settings_dict:
            self.set_all(settings_dict)

    def set_all(self, settings_dict):
        for name, value in settings_dict.items():

            if name == "kernel" and isinstance(value, str) and value:
                #assume that the kernel is saved as a complex array
                self.kernel = np.loadtxt(os.path.join(config.KernelDir, value+'.txt'), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
            elif name == "kernel_bias" and isinstance(value, str) and value:
                self.kernel_bias = eval(value)
            #elif hasattr(self, name):
            #        setattr(self, name, value)
            elif name == "channel":
                setattr(self, 'phys_channel', int(value))
            else:
                try:
                    setattr(self, name, value)
                except AttributeError:
                    logger.debug("Could not set channel attribute: {} on X6 {} channel.".format(name, self.stream_type))
                    pass

        if self.stream_type == "Integrated":
            demod_channel = 0
            result_channel = self.dsp_channel
            self.dtype = np.complex128
        elif self.stream_type == "Demodulated":
            demod_channel = self.dsp_channel
            result_channel = 0
            self.dtype = np.complex128
        else: #Raw
            demod_channel  = 0
            result_channel = 0
            self.dtype = np.float64

        self.channel_tuple = (int(self.phys_channel), int(demod_channel), int(result_channel))

class X6(Instrument):
    """BBN QDSP running on the II-X6 digitizer"""
    instrument_type = ("Digitizer")

    def __init__(self, resource_name=None, name="Unlabeled X6", gen_fake_data=False):
        # X6Channel objects
        self._channels = []
        # socket r/w pairs for each channel
        self._chan_to_rsocket = {}
        self._chan_to_wsocket = {}

        self.resource_name = resource_name
        self.name          = name

        self.last_timestamp = datetime.datetime.now()
        self.gen_fake_data = gen_fake_data

        if fake_x6:
            self._lib = MagicMock()
        else:
            self._lib = libx6.X6()

    def __str__(self):
        return "<X6({}/{})>".format(self.name, self.resource_name)

    def __del__(self):
        self.disconnect()

    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name

        # pass thru functions
        self.acquire    = self._lib.acquire
        self.stop       = self._lib.stop
        self.disconnect = self._lib.disconnect

        if self.gen_fake_data or fake_x6:
            self._lib = MagicMock()
            logger.warning("Could not load x6 library")
            logger.warning("X6 GENERATING FAKE DATA")
        self._lib.connect(int(self.resource_name))

    def disconnect(self):
        for sock in self._chan_to_rsocket.values():
            sock.close()
        for sock in self._chan_to_wsocket.values():
            sock.close()
        self._chan_to_rsocket.clear()
        self._chan_to_wsocket.clear()
        self._lib.disconnect()

    def set_all(self, settings_dict):
        # Call the non-channel commands
        super(X6, self).set_all(settings_dict)
        # perform channel setup
        for chan in self._channels:
            self.channel_setup(chan)

    def channel_setup(self, channel):
        a, b, c = channel.channel_tuple
        self._lib.enable_stream(a, b, c)

        if channel.stream_type == "Raw":
            return
        elif channel.stream_type == "Demodulated":
            self._lib.set_nco_frequency(a, b, channel.if_freq)
        elif channel.stream_type == "Integrated":
            if channel.kernel is None:
                logger.error("Integrated streams must specify a kernel")
                return
            # convert to complex128
            channel.kernel = channel.kernel.astype(complex)
            self._lib.write_kernel(a, b, c, channel.kernel)
            self._lib.set_kernel_bias(a, b, c, channel.kernel_bias)
            self._lib.set_threshold(a, c, channel.threshold)
            self._lib.set_threshold_invert(a, c, channel.threshold_invert)
        else:
            logger.error("Unrecognized stream type %s" % channel.stream_type)

    def data_available(self):
        return self._lib.get_data_available()

    def done(self):
        if self.data_available():
            return False
        if self._lib.get_is_running():
            return False
        return True

    def get_socket(self, channel):
        if channel in self._chan_to_rsocket:
            return self._chan_to_rsocket[channel]

        try:
            rsock, wsock = socket.socketpair()
        except:
            raise Exception("Could not create read/write socket pair")
        self._lib.register_socket(*channel.channel_tuple, wsock)
        self._chan_to_rsocket[channel] = rsock
        self._chan_to_wsocket[channel] = wsock
        return rsock

    def add_channel(self, channel):
        if not isinstance(channel, X6Channel):
            raise TypeError("X6 passed {} rather than an X6Channel object.".format(str(channel)))

        if channel.stream_type not in ['Raw', 'Demodulated', 'Integrated']:
            raise ValueError("Stream type of {} not recognized by X6".format(str(channel.stream_type)))

        # todo: other checking here
        self._channels.append(channel)

    def spew_fake_data(self):
        for chan, wsock in self._chan_to_wsocket.items():
            if chan.stream_type == "Integrated":
                length = 1
                data = 0.5 + 0.2*np.random.random(length).astype(chan.dtype)
            elif chan.stream_type == "Demodulated":
                length = int(self._lib.record_length/32)
                data = np.zeros(length, dtype=chan.dtype)
                data[int(length/4):int(3*length/4)] = 1.0
                data += 0.1*np.random.random(length)
            else: #Raw
                length = int(self._lib.record_length/4)
                signal = np.sin(np.linspace(0,10.0*np.pi,int(length/2)))
                data = np.zeros(length, dtype=chan.dtype)
                data[int(length/4):int(length/4)+len(signal)] = signal
                data += 0.1*np.random.random(length)
            wsock.send(struct.pack('n', length*data.dtype.itemsize) + data.tostring())

    def receive_data(self, channel, oc):
        # push data from a socket into an OutputConnector (oc)
        self.last_timestamp = datetime.datetime.now()
        # wire format is just: [size, buffer...]
        sock = self._chan_to_rsocket[channel]
        # TODO receive 4 or 8 bytes depending on sizeof(size_t)
        msg = sock.recv(8)
        # reinterpret as int (size_t)
        msg_size = struct.unpack('n', msg)[0]
        buf = sock.recv(msg_size, socket.MSG_WAITALL)
        if len(buf) != msg_size:
            logger.error("Channel %s socket msg shorter than expected" % channel.channel)
            logger.error("Expected %s bytes, received %s bytes" % (msg_size, len(buf)))
            # assume that we cannot recover, so stop listening.
            loop = asyncio.get_event_loop()
            loop.remove_reader(sock)
            return
        data = np.frombuffer(buf, dtype=channel.dtype)
        asyncio.ensure_future(oc.push(data))

    def get_buffer_for_channel(self, channel):
        return self._lib.transfer_stream(*channel.channel)

    async def wait_for_acquisition(self, timeout=5):
        if self.gen_fake_data:
            for i in range(self._lib.nbr_segments):
                for j in range(self._lib.nbr_round_robins):
                    self.spew_fake_data()
                    await asyncio.sleep(0.005)
        else:
            while not self.done():
                if (datetime.datetime.now() - self.last_timestamp).seconds > timeout:
                    logger.error("Digitizer %s timed out.", self.name)
                    break
                await asyncio.sleep(0.1)

    # pass thru properties
    @property
    def reference(self):
        return self._lib.reference
    @reference.setter
    def reference(self, value):
        self._lib.reference = value

    @property
    def acquire_mode(self):
        return self._lib.acquire_mode
    @acquire_mode.setter
    def acquire_mode(self, value):
        self._lib.acquire_mode = value

    @property
    def record_length(self):
        return self._lib.record_length
    @record_length.setter
    def record_length(self, value):
        self._lib.record_length = value

    @property
    def nbr_waveforms(self):
        return self._lib.nbr_waveforms
    @nbr_waveforms.setter
    def nbr_waveforms(self, value):
        self._lib.nbr_waveforms = value

    @property
    def nbr_segments(self):
        return self._lib.nbr_segments
    @nbr_segments.setter
    def nbr_segments(self, value):
        self._lib.nbr_segments = value

    @property
    def nbr_round_robins(self):
        return self._lib.nbr_round_robins
    @nbr_round_robins.setter
    def nbr_round_robins(self, value):
        self._lib.nbr_round_robins = value
