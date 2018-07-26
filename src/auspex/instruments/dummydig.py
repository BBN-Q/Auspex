# Copyright 2018 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['DummydigChannel', 'Dummydig']

import time
import socket
import struct
import datetime
import asyncio
import numpy as np
import os

from auspex.log import logger
import auspex.config as config
from .instrument import Instrument, DigitizerChannel
from unittest.mock import MagicMock

class DummydigChannel(DigitizerChannel):
    """Channel for an Dummy digitizer"""

    def __init__(self, settings_dict=None):
        self.stream_type       = "Raw"
        self.if_freq           = 0.0
        self.kernel            = None
        self.kernel_bias       = 0.0
        self.threshold         = 0.0
        self.threshold_invert  = False

        self.phys_channel   = 1
        self.dsp_channel    = 0

        self.dtype = np.complex128
        self.ideal_data = None

        if settings_dict:
            self.set_all(settings_dict)

    def set_all(self, settings_dict):
        for name, value in settings_dict.items():
            if name == "kernel" and value:
                #check if the kernel is given as an existing path or an expression to eval
                if os.path.exists(os.path.join(config.KernelDir, value+'.txt')):
                    self.kernel = np.loadtxt(os.path.join(config.KernelDir, value+'.txt'), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
                else:
                    try:
                        self.kernel = eval(value)
                    except:
                        raise ValueError('Kernel invalid. Provide a file name or an expression to evaluate')
            elif name == "kernel_bias" and isinstance(value, str) and value:
                self.kernel_bias = eval(value)
            #elif hasattr(self, name):
            #        setattr(self, name, value)
            elif name == "channel":
                setattr(self, 'phys_channel', int(value))
            elif name == 'threshold':
                setattr(self, 'threshold', value)
            elif name == 'threshold_invert':
                setattr(self, 'threshold_invert', bool(value))
            elif name == 'ideal_data': # for testing purposes
                self.ideal_data = np.load(os.path.abspath(value+'.npy'))
            else:
                try:
                    setattr(self, name, value)
                except AttributeError:
                    logger.debug("Could not set channel attribute: {} on X6 {} channel.".format(name, self.stream_type))
                    pass

class Dummydig(Instrument):
    """BBN Dummy digitizer for examples and unit testing"""
    instrument_type = ("Digitizer")

    def __init__(self, resource_name=None, name="Unlabeled Dummydig", gen_fake_data=False):
        
        # X6Channel objects
        self._channels = []
        # socket r/w pairs for each channel
        self._chan_to_rsocket = {}
        self._chan_to_wsocket = {}

        self.resource_name = resource_name
        self.name          = name

        self.last_timestamp = datetime.datetime.now()


    def __str__(self):
        return "<Dummydig({}/{})>".format(self.name, self.resource_name)

    def __del__(self):
        self.disconnect()

    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name

    def disconnect(self):
        for sock in self._chan_to_rsocket.values():
            sock.close()
        for sock in self._chan_to_wsocket.values():
            sock.close()
        self._chan_to_rsocket.clear()
        self._chan_to_wsocket.clear()

    def set_all(self, settings_dict):
        # Call the non-channel commands
        super(Dummydig, self).set_all(settings_dict)

        # perform channel setup
        for chan in self._channels:
            self.channel_setup(chan)

    def channel_setup(self, channel):
        pass
    
    def data_available(self):
        return False #TODO

    def done(self):
        if self.data_available():
            return False
        return True

    def get_socket(self, channel):
        if channel in self._chan_to_rsocket:
            return self._chan_to_rsocket[channel]
        try:
            rsock, wsock = socket.socketpair()
            wsock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536*4) 
        except:
            raise Exception("Could not create read/write socket pair")
        self._chan_to_rsocket[channel] = rsock
        self._chan_to_wsocket[channel] = wsock
        return rsock

    def add_channel(self, channel):
        if not isinstance(channel, DummydigChannel):
            raise TypeError("Dummydig passed {} rather than an DummydigChannel object.".format(str(channel)))

        if channel.stream_type not in ['Raw', 'Demodulated', 'Integrated']:
            raise ValueError("Stream type of {} not recognized by Dummydig".format(str(channel.stream_type)))

        # todo: other checking here
        self._channels.append(channel)

    def spew_fake_data(self, ideal_datapoint=None):
        """
        Generate fake data on the stream. For unittest usage.
        """
            
        for chan, wsock in self._chan_to_wsocket.items():
                length = int(self.record_length)
                signal = np.exp(1j*np.linspace(0,10.0*np.pi,int(length/2)))
                data = np.zeros(length, dtype=chan.dtype)
                data[int(length/4):int(length/4)+len(signal)] = signal
                #data += 0.1*np.random.random(length)
                wsock.send(struct.pack('n', length*data.dtype.itemsize) + data.tostring())

    def acquire(self):
        pass
    def stop(self):
        pass
    
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

    async def wait_for_acquisition(self, timeout=5):
        for j in range(self.nbr_round_robins):
            for i in range(self.nbr_segments):
                self.spew_fake_data()
                await asyncio.sleep(0.1)

    # pass thru properties
    '''
    @property
    def reference(self):
        return self.reference
    @reference.setter
    def reference(self, value):
        self.reference = value

    @property
    def acquire_mode(self):
        return self.acquire_mode
    @acquire_mode.setter
    def acquire_mode(self, value):
        self.acquire_mode = value

    @property
    def record_length(self):
        return self.record_length
    @record_length.setter
    def record_length(self, value):
        self.record_length = value

    @property
    def nbr_waveforms(self):
        return self.nbr_waveforms
    @nbr_waveforms.setter
    def nbr_waveforms(self, value):
        self.nbr_waveforms = value

    @property
    def nbr_segments(self):
        return self.nbr_segments
    @nbr_segments.setter
    def nbr_segments(self, value):
        self.nbr_segments = value

    @property
    def nbr_round_robins(self):
        return self.nbr_round_robins
    @nbr_round_robins.setter
    def nbr_round_robins(self, value):
        self.nbr_round_robins = value
    '''