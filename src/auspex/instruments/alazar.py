# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['AlazarATS9870', 'AlazarChannel']

import re
import socket
import struct
import datetime, time
import sys
import numpy as np

from multiprocessing import Value

from .instrument import Instrument, ReceiverChannel
from auspex.log import logger
import auspex.config as config

from unittest.mock import MagicMock

# win32 doesn't support MSG_WAITALL, so on windows we
# need to do things a slower, less efficient way.
# (we could optimize this, if performance becomes a problem)
#
# TODO: this code is repeated in the X6 driver.
#
if sys.platform == 'win32':
    def sock_recvall(s, data_len):
        buf = bytearray()
        while data_len > 0:
            new = s.recv(data_len)
            data_len -= len(new)
            buf.extend(new)
        return bytes(buf)
else:
    def sock_recvall(s, data_len):
        return s.recv(data_len, socket.MSG_WAITALL)

# Dirty trick to avoid loading libraries when scraping
# This code using quince.
if config.auspex_dummy_mode:
    fake_alazar = True
else:
    try:
        from libalazar import ATS9870
        fake_alazar = False
    except:
        # logger.warning("Could not load alazar library")
        fake_alazar = True

class AlazarChannel(ReceiverChannel):
    phys_channel = None

    def __init__(self, receiver_channel=None):
        if receiver_channel:
            self.set_by_receiver(receiver_channel)

    def set_all(self, settings_dict):
        for name, value in settings_dict.items():
            if hasattr(self, name):
                setattr(self, name, value)

    def set_by_receiver(self, receiver):
        self.phys_channel = receiver.channel

class AlazarATS9870(Instrument):
    """Alazar ATS9870 digitizer"""
    instrument_type = ("Digitizer")

    def __init__(self, resource_name=None, name="Unlabeled Alazar", gen_fake_data=True):
        self.name = name

        # A list of AlazarChannel objects
        self.channels = []

        self.resource_name = resource_name

        # For lookup
        self._chan_to_buf = {}
        self._chan_to_rsocket = {}
        self._chan_to_wsocket = {}

        self.last_timestamp = Value('d', datetime.datetime.now().timestamp())
        self.fetch_count    = Value('d', 0)

        self.gen_fake_data        = gen_fake_data
        self.increment_ideal_data = False
        self.ideal_counter        = 0
        self.ideal_data           = None

    def connect(self, resource_name=None):
        if fake_alazar or self.gen_fake_data:
            self._lib = MagicMock()
        else:
            self._lib = ATS9870()

        if fake_alazar:
            logger.warning("Could not load Alazar library")
        if resource_name:
            self.resource_name = resource_name

        self._lib.connect("{}/{}".format(self.name, int(self.resource_name)))
        for channel in self.channels:
            self.get_socket(channel)

    def acquire(self):
        self.fetch_count.value = 0
        self._lib.acquire()

    def stop(self):
        self._lib.stop()

    def data_available(self):
        return self._lib.data_available()

    def done(self):
        # logger.info(f"{self.fetch_count.value} {len(self.channels)} {self.number_acquisitions}")
        return self.fetch_count.value >= (len(self.channels) * self.number_acquisitions)

    def get_socket(self, channel):
        if channel in self._chan_to_rsocket:
            return self._chan_to_rsocket[channel]

        try:
            rsock, wsock = socket.socketpair()
        except:
            raise Exception("Could not create read/write socket pair")
        self._lib.register_socket(channel.phys_channel - 1, wsock)
        # logger.info(f"Passing socket {wsock} to libalazar driver")
        self._chan_to_rsocket[channel] = rsock
        self._chan_to_wsocket[channel] = wsock
        return rsock

    def add_channel(self, channel):
        if not isinstance(channel, AlazarChannel):
            raise TypeError("Alazar passed {} rather than an AlazarChannel object.".format(str(channel)))

        # We can have either 1 or 2, or both.
        if len(self.channels) < 2 and channel not in self.channels:
            self.channels.append(channel)
            self._chan_to_buf[channel] = channel.phys_channel

    def spew_fake_data(self, counter, ideal_datapoint=0, random_mag=0.1, random_seed=12345):
        """
        Generate fake data on the stream. For unittest usage.
        ideal_datapoint: mean of the expected signal 

        Returns the total number of fake data points, so that we can
        keep track of how many we expect to receive, when we're doing
        the test with fake data
        """
        total = 0
        np.random.seed(random_seed)

        for chan, wsock in self._chan_to_wsocket.items():
            length = int(self.record_length)
            signal = np.sin(np.linspace(0,10.0*np.pi,int(length/2)))
            data = np.zeros(length, dtype=np.float32)
            data[int(length/4):int(length/4)+len(signal)] = signal * (1.0 if ideal_datapoint == 0 else ideal_datapoint)
            data += random_mag*np.random.random(length)
            total += length
            # logger.info(f"Sending {struct.pack('n', length*np.float32().itemsize)}")
            wsock.send(struct.pack('n', length*np.float32().itemsize) + data.tostring())
            counter[chan] += length

        return total

    def receive_data(self, channel, oc, exit):
        sock = self._chan_to_rsocket[channel]
        # logger.info(f"Recovered socket {sock} for data acquisition")
        sock.settimeout(2)
        self.last_timestamp.value = datetime.datetime.now().timestamp()
        # logger.info("Entering receive data")
        while not exit.is_set():
            # push data from a socket into an OutputConnector (oc)
            # wire format is just: [size, buffer...]
            # TODO receive 4 or 8 bytes depending on sizeof(size_t)
            try:
                # logger.info("Trying to receive data")
                msg = sock.recv(8)
                self.last_timestamp.value = datetime.datetime.now().timestamp()
            except:
                # logger.info("Failed to receive data")
                continue
            # logger.info(f"In receive data: {msg}")
            # reinterpret as int (size_t)
            msg_size = struct.unpack('n', msg)[0]
            buf = sock_recvall(sock, msg_size)
            if len(buf) != msg_size:
                logger.error(f"Channel {channel} socket msg shorter than expected")
                logger.error(f"Expected {msg_size} bytes, received {len(buf)} bytes")
                return
            self.fetch_count.value += 1
            data = np.frombuffer(buf, dtype=np.float32)
            oc.push(data)

    def get_buffer_for_channel(self, channel):
        self.fetch_count.value += 1
        return getattr(self._lib, 'ch{}Buffer'.format(self._chan_to_buf[channel]))

    def wait_for_acquisition(self, timeout=5, ocs=None):
        if self.gen_fake_data:
            total_spewed = 0

            counter = {chan: 0 for chan in self._chan_to_wsocket.keys()}
            initial_points = {oc: oc.points_taken.value for oc in ocs}
            # print(self.number_averages, self.number_segments)
            for j in range(self.number_averages):
                for i in range(self.number_segments):
                    if self.ideal_data is not None:
                        #add ideal data for testing
                        if hasattr(self, 'exp_step') and self.increment_ideal_data:
                            raise Exception("Cannot use both exp_step and increment_ideal_data")
                        elif hasattr(self, 'exp_step'):
                            total_spewed += self.spew_fake_data(
                                    counter, self.ideal_data[self.exp_step][i])
                        elif self.increment_ideal_data:
                            total_spewed += self.spew_fake_data(
                                   counter, self.ideal_data[self.ideal_counter][i])
                        else:
                            total_spewed += self.spew_fake_data(
                                    counter, self.ideal_data[i])
                    else:
                        total_spewed += self.spew_fake_data(counter)

                    time.sleep(0.0001)

            self.ideal_counter += 1
            # logger.info("Counter: %s", str(counter))
            # logger.info('TOTAL fake data generated %d', total_spewed)
            if ocs:
                while True:
                    total_taken = 0
                    for oc in ocs:
                        total_taken += oc.points_taken.value - initial_points[oc]
                        # logger.info('TOTAL fake data received %d', oc.points_taken.value - initial_points[oc])
                    if total_taken == total_spewed:
                        break

                    # logger.info('WAITING for acquisition to finish %d < %d', total_taken, total_spewed)
                    time.sleep(0.025)

        else:
            while not self.done():
                if (datetime.datetime.now().timestamp() - self.last_timestamp.value) > timeout:
                    logger.error("Digitizer %s timed out.", self.name)
                    raise Exception("Alazar timed out.")
                time.sleep(0.2)

        logger.debug("Digitizer %s finished getting data.", self.name)

    def configure_with_dict(self, settings_dict):
        config_dict = {
            'acquireMode': 'digitizer',
            'bandwidth': "Full" ,
            'clockType': "int",
            'delay': 0.0,
            'enabled': True,
            'label': 'Alazar',
            'recordLength': settings_dict['record_length'],
            'nbrSegments': self.proxy_obj.number_segments,
            'nbrWaveforms': self.proxy_obj.number_waveforms,
            'nbrRoundRobins': self.proxy_obj.number_averages,
            'samplingRate': 500e6,
            'triggerCoupling': "DC",
            'triggerLevel': 100,
            'triggerSlope': "rising",
            'triggerSource': "Ext",
            'verticalCoupling': "DC",
            'verticalOffset': 0.0,
            'verticalScale': 1.0
        }

        self._lib.setAll(config_dict)
        self.record_length           = settings_dict['record_length']
        self.number_acquisitions     = self._lib.numberAcquisitions
        self.samples_per_acquisition = self._lib.samplesPerAcquisition
        self.number_segments         = self.proxy_obj.number_segments,
        self.number_waveforms        = self.proxy_obj.number_waveforms,
        self.number_averages         = self.proxy_obj.number_averages,
        self.ch1_buffer              = self._lib.ch1Buffer
        self.ch2_buffer              = self._lib.ch2Buffer

    def disconnect(self):
        self._lib.disconnect()
        for socket in self._chan_to_rsocket.values():
            socket.close()
        for socket in self._chan_to_wsocket.values():
            socket.close()
        self._chan_to_rsocket.clear()
        self._chan_to_wsocket.clear()
        self._lib.unregister_sockets()

    def __str__(self):
        return "<AlazarATS9870({}/{})>".format(self.name, self.resource_name)
