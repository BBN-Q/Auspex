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

    def __init__(self, resource_name=None, name="Unlabeled Alazar", gen_fake_data=False):
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
        self.total_received = Value('d', 0)

        self.gen_fake_data        = gen_fake_data
        self.increment_ideal_data = False
        self.ideal_counter        = 0
        self.ideal_data           = None
        np.random.seed(12345)

    def connect(self, resource_name=None):
        if config.auspex_dummy_mode or self.gen_fake_data:
            self.fake_alazar = True
            self._lib = MagicMock()
        else:
            try:
                from libalazar import ATS9870
                self._lib = ATS9870()
                self.fake_alazar = False
            except:
                raise Exception("Could not find libalazar. You can run in dummy mode by setting config.auspex_dummy_mode \
                    or setting the gen_fake_data property of this instrument.")
        if resource_name:
            self.resource_name = resource_name

        self._lib.connect("{}/{}".format(self.name, int(self.resource_name)))
        for channel in self.channels:
            self.get_socket(channel)

    def acquire(self):
        self.fetch_count.value = 0
        self.total_received.value = 0
        self._lib.acquire()

    def stop(self):
        self._lib.stop()

    def data_available(self):
        return self._lib.data_available()

    def done(self):
        # logger.warning(f"Checking alazar doneness: {self.total_received.value} {self.number_segments * self.number_averages * self.record_length}")
        return self.total_received.value >=  (self.number_segments * self.number_averages * self.record_length * len(self.channels))

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

    def spew_fake_data(self, counter, ideal_data, random_mag=0.1, random_seed=12345):
        """
        Generate fake data on the stream. For unittest usage.
        ideal_data: array or list giving means of the expected signal for each segment

        Returns the total number of fake data points, so that we can
        keep track of how many we expect to receive, when we're doing
        the test with fake data
        """
        for chan, wsock in self._chan_to_wsocket.items():
            length = int(self.record_length)
            buff = np.zeros((self.number_segments, length), dtype=np.float32)
            for i in range(self.number_segments):
                signal = np.sin(np.linspace(0,10.0*np.pi,int(length/2)))
                buff[i, int(length/4):int(length/4)+len(signal)] = signal * (1.0 if ideal_data[i] == 0 else ideal_data[i])

            buff += random_mag*np.random.random((self.number_segments, length))

            wsock.send(struct.pack('n', self.number_segments*length*np.float32().itemsize) + buff.flatten().tostring())
            counter[chan] += length*self.number_segments

        return length*self.number_segments*len(self._chan_to_wsocket)

    def receive_data(self, channel, oc, exit, ready, run):
        sock = self._chan_to_rsocket[channel]
        sock.settimeout(2)
        self.last_timestamp.value = datetime.datetime.now().timestamp()
        last_print = datetime.datetime.now().timestamp()
        ready.value += 1

        while not exit.is_set():
            # push data from a socket into an OutputConnector (oc)
            # wire format is just: [size, buffer...]
            # TODO receive 4 or 8 bytes depending on sizeof(size_t)
            run.wait() # Block until we are running again
            try:
                msg = sock.recv(8)
                self.last_timestamp.value = datetime.datetime.now().timestamp()
            except:
                logger.debug("Didn't find any data on socket within 2 seconds (this is normal during experiment shutdown).")
                continue
            msg_size = struct.unpack('n', msg)[0]
            buf = sock_recvall(sock, msg_size)
            while len(buf) < msg_size:
                # time.sleep(0.01)
                buf2 = sock_recvall(sock, msg_size-len(buf))
                buf = buf+buf2
            data = np.frombuffer(buf, dtype=np.float32)
            self.total_received.value += len(data)
            if datetime.datetime.now().timestamp() - last_print > 0.25:
                last_print = datetime.datetime.now().timestamp()
                # logger.info(f"Alz: {self.total_received.value}")
            oc.push(data)
            self.fetch_count.value += 1

    def get_buffer_for_channel(self, channel):
        self.fetch_count.value += 1
        return getattr(self._lib, 'ch{}Buffer'.format(self._chan_to_buf[channel]))

    def wait_for_acquisition(self, dig_run, timeout=5, ocs=None, progressbars=None):
        progress_updaters = {}
        if ocs and progressbars:
            for oc in ocs:
                if hasattr(progressbars[oc], 'goto'):
                    progress_updaters[oc] = lambda x: progressbars[oc].goto(x)
                else:
                    progress_updaters[oc] = lambda x: setattr(progressbars[oc], 'value', x)

        if self.gen_fake_data:
            total_spewed = 0

            counter = {chan: 0 for chan in self._chan_to_wsocket.keys()}
            initial_points = {oc: oc.points_taken.value for oc in ocs}
            # print(self.number_averages, self.number_segments)
            for j in range(self.number_averages):
                # for i in range(self.number_segments):
                if self.ideal_data is not None:
                    #add ideal data for testing
                    if hasattr(self, 'exp_step') and self.increment_ideal_data:
                        raise Exception("Cannot use both exp_step and increment_ideal_data")
                    elif hasattr(self, 'exp_step'):
                        total_spewed += self.spew_fake_data(
                                counter, self.ideal_data[self.exp_step])
                    elif self.increment_ideal_data:
                        total_spewed += self.spew_fake_data(
                               counter, self.ideal_data[self.ideal_counter])
                    else:
                        total_spewed += self.spew_fake_data(
                                counter, self.ideal_data)
                else:
                    total_spewed += self.spew_fake_data(counter, [0.0 for i in range(self.number_segments)])

                time.sleep(0.0001)

            self.ideal_counter += 1

        while not self.done():
            if not dig_run.is_set():
                self.last_timestamp.value = datetime.datetime.now().timestamp()
            if (datetime.datetime.now().timestamp() - self.last_timestamp.value) > timeout:
                logger.error("Digitizer %s timed out. Timeout was %f, time was %f", self.name, timeout, (datetime.datetime.now().timestamp() - self.last_timestamp.value))
                raise Exception("Alazar timed out.")
            if progressbars:
                for oc in ocs:
                    progress_updaters[oc](oc.points_taken.value)
            #time.sleep(0.2) Does this need to be here at all?
        if progressbars:
            try:
                progressbars[oc].next()
                progressbars[oc].finish()
            except AttributeError:
                pass

        logger.debug("Digitizer %s finished getting data.", self.name)

    def configure_with_dict(self, settings_dict):
        config_dict = {
            'acquireMode': 'digitizer',
            'bandwidth': "Full" ,
            'clockType': "ref",
            'delay': 0.0,
            'enabled': True,
            'label': 'Alazar',
            'recordLength': settings_dict['record_length'],
            'nbrSegments': self.proxy_obj.number_segments,
            'nbrWaveforms': self.proxy_obj.number_waveforms,
            'nbrRoundRobins': self.proxy_obj.number_averages,
            'samplingRate': self.proxy_obj.sampling_rate,
            'triggerCoupling': "DC",
            'triggerLevel': 100,
            'triggerSlope': "rising",
            'triggerSource': "Ext",
            'verticalCoupling': "DC",
            'verticalOffset': 0.0,
            'verticalScale': self.proxy_obj.vertical_scale
        }

        self._lib.setAll(config_dict)
        self.record_length           = settings_dict['record_length']
        self.number_acquisitions     = self._lib.numberAcquisitions
        self.samples_per_acquisition = self._lib.samplesPerAcquisition
        self.number_segments         = self.proxy_obj.number_segments
        self.number_waveforms        = self.proxy_obj.number_waveforms
        self.number_averages         = self.proxy_obj.number_averages
        self.ch1_buffer              = self._lib.ch1Buffer
        self.ch2_buffer              = self._lib.ch2Buffer
        self.record_length           = settings_dict['record_length']
        self.number_segments         = self.proxy_obj.number_segments
        self.number_waveforms        = self.proxy_obj.number_waveforms
        self.number_averages         = self.proxy_obj.number_averages

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
