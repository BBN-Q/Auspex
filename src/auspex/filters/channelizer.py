# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy
import asyncio, concurrent

import numpy as np
import scipy.signal
import scipy.fftpack

from auspex.parameter import Parameter, IntParameter, FloatParameter
from auspex.filters.filter import Filter, InputConnector, OutputConnector
from auspex.stream import  DataStreamDescriptor
from auspex.log import logger

class Channelizer(Filter):
    """Digital demodulation and filtering to select a particular frequency multiplexed channel"""

    sink              = InputConnector()
    source            = OutputConnector()
    decimation_factor = IntParameter(value_range=(1,100), default=2, snap=1)
    frequency         = FloatParameter(value_range=(-5e9,5e9), increment=1.0e6, default=-9e6)
    bandwidth         = FloatParameter(value_range=(0.00, 100e6), increment=0.1e6, default=5e6)

    def __init__(self, frequency=None, bandwidth=None, decimation_factor=None, **kwargs):
        super(Channelizer, self).__init__(**kwargs)
        if frequency:
            self.frequency.value = frequency
        if bandwidth:
            self.bandwidth.value = bandwidth
        if decimation_factor:
            self.decimation_factor.value = decimation_factor
        self.quince_parameters = [self.decimation_factor, self.frequency, self.bandwidth]

    def update_descriptors(self):
        logger.debug('Updating Channelizer "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        # extract record time sampling
        time_pts = self.sink.descriptor.axes[-1].points
        self.record_length = len(time_pts)
        self.time_step = time_pts[1] - time_pts[0]
        logger.debug("Channelizer time_step = {}".format(self.time_step))

        # store reference for mix down
        self.reference = np.exp(2j*np.pi * self.frequency.value * time_pts, dtype=np.complex64)

        # convert bandwidth to normalized Nyquist interval
        n_bandwidth = self.bandwidth.value * self.time_step

        # create brick wall filter coefficients
        freq_pts = scipy.fftpack.fftfreq(self.record_length)
        demod_cutoff = 0.5/self.decimation_factor.value
        self.demod_indices = np.where( np.logical_and(freq_pts >= -demod_cutoff, freq_pts < demod_cutoff) )
        self.filter_coefs = np.complex64(np.abs(freq_pts) < n_bandwidth/2)[self.demod_indices]

        # update output descriptors
        decimated_descriptor = DataStreamDescriptor()
        decimated_descriptor.axes = self.sink.descriptor.axes[:]
        decimated_descriptor.axes[-1] = deepcopy(self.sink.descriptor.axes[-1])
        decimated_descriptor.axes[-1].points = self.sink.descriptor.axes[-1].points[self.decimation_factor.value-1::self.decimation_factor.value]
        decimated_descriptor.exp_src = self.sink.descriptor.exp_src
        decimated_descriptor.dtype = np.complex64
        for os in self.source.output_streams:
            os.set_descriptor(decimated_descriptor)
            os.end_connector.update_descriptors()

    async def process_data(self, data):
        # Assume for now we get a integer number of records at a time
        # TODO: handle partial records
        num_records = data.size // self.record_length

        # mix with reference
        mix_product = self.reference * np.reshape(data, (num_records, self.record_length), order="C")

        # convert to frequency domain
        f_domain = scipy.fftpack.fft(mix_product, overwrite_x=True)

        # decimate
        f_domain = f_domain[..., self.demod_indices]

        # filter
        f_domain = self.filter_coefs * f_domain

        # back to time domain
        filtered = scipy.fftpack.ifft(f_domain, overwrite_x=True)

        # recover gain from selecting single sideband and decimation
        filtered *= 2*self.decimation_factor.value

        # push to ouptut connectors
        for os in self.source.output_streams:
            await os.push(filtered)
