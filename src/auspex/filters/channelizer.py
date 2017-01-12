# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import os
import platform
from copy import deepcopy
import asyncio, concurrent

import numpy as np
import scipy.signal
import scipy.fftpack

from auspex.parameter import Parameter, IntParameter, FloatParameter
from auspex.filters.filter import Filter, InputConnector, OutputConnector
from auspex.stream import  DataStreamDescriptor
from auspex.log import logger

# load libchannelizer to access Intel IPP filtering functions
import numpy.ctypeslib as npct
from ctypes import c_int, c_size_t
np_float  = npct.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
#On Windows add build path to system path to pick up DLL mingw dependencies
libchannelizer_path = os.path.abspath(os.path.join( os.path.dirname(__file__), "libchannelizer"))
if "Windows" in platform.platform():
    os.environ["PATH"] += ";" + libchannelizer_path
libipp = npct.load_library("libchannelizer",  libchannelizer_path)
libipp.filter_records_fir.argtypes = [np_float, c_size_t, c_int, np_float, c_size_t, c_size_t, np_float]
libipp.filter_records_iir.argtypes = [np_float, c_size_t, np_float, c_size_t, c_size_t, np_float]
libipp.init()

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

        # convert bandwidth normalized to Nyquist interval
        n_bandwidth = self.bandwidth.value * self.time_step * 2
        n_frequency = self.frequency.value * self.time_step * 2

        # arbitrarily decide on three stage filter pipeline
        # 1. first stage decimating filter on real data
        # 2. decond stage decimating filter on mixed product to boost n_bandwidth
        # 3. final channel selecting filter at n_bandwidth/2

        # anecdotally don't decimate more than a factor of eight for stability

        self.decim_factors = [1]*3
        self.filters = [None]*3

        # first stage decimating filter
        # maximize first stage decimation:
        #     * minimize subsequent stages time taken
        #     * filter and decimate while signal is still real
        #     * first stage decimation cannot be too large or then 2omega signal from mixing will alias
        d1 = 1
        while (d1 < 8) and (2*n_frequency <= 0.8/d1) and (d1 < self.decimation_factor.value):
            d1 *= 2
            n_bandwidth *= 2
            n_frequency *= 2

        if d1 > 1:
            # create an anti-aliasing filter
            # pass-band to 0.8 * decimation factor; anecdotally single precision needs order <= 4 for stability
            b,a = scipy.signal.cheby1(4, 3, 0.8/d1)
            b = np.float32(b)
            a = np.float32(a)
            self.decim_factors[0] = d1
            self.filters[0]  = (b,a)

        # store decimated reference for mix down
        ref = np.exp(2j*np.pi * self.frequency.value * time_pts[::d1], dtype=np.complex64)
        self.reference_r = np.real(ref)
        self.reference_i = np.imag(ref)

        # second stage filter to bring n_bandwidth/2 up
        # decimation cannot be too large or will impinge on channel bandwidth (keep n_bandwidth/2 <= 0.8)
        d2 = 1
        while (d2 < 8) and ((d1*d2) < self.decimation_factor.value) and (n_bandwidth/2 <= 0.8):
            d2 *= 2
            n_bandwidth *= 2
            n_frequency *= 2

        if d2 > 1:
            # create an anti-aliasing filter
            # pass-band to 0.8 * decimation factor; anecdotally single precision needs order <= 4 for stability
            b,a = scipy.signal.cheby1(4, 3, 0.8/d2)
            b = np.float32(b)
            a = np.float32(a)
            self.decim_factors[1] = d2
            self.filters[1]  = (b,a)


        # final channel selection filter
        if n_bandwidth < 0.1:
            raise(ValueError, "Insufficient decimation to achieve stable filter")

        b,a = scipy.signal.cheby1(4, 3, n_bandwidth/2)
        b = np.float32(b)
        a = np.float32(a)
        self.decim_factors[2] = self.decimation_factor.value // (d1*d2)
        self.filters[2]  = (b,a)

        # update output descriptors
        decimated_descriptor = DataStreamDescriptor()
        decimated_descriptor.axes = self.sink.descriptor.axes[:]
        decimated_descriptor.axes[-1] = deepcopy(self.sink.descriptor.axes[-1])
        decimated_descriptor.axes[-1].points = self.sink.descriptor.axes[-1].points[self.decimation_factor.value-1::self.decimation_factor.value]
        decimated_descriptor.exp_src = self.sink.descriptor.exp_src
        decimated_descriptor.dtype = np.complex64
        for os in self.source.output_streams:
            os.set_descriptor(decimated_descriptor)
            if os.end_connector is not None:
                os.end_connector.update_descriptors()

    async def process_data(self, data):
        # Assume for now we get a integer number of records at a time
        # TODO: handle partial records
        num_records = data.size // self.record_length
        reshaped_data = np.reshape(data, (num_records, self.record_length), order="C")

        # first stage decimating filter
        if self.filters[0] is not None:
            stacked_coeffs = np.concatenate(self.filters[0])
            # filter
            filtered = np.empty_like(reshaped_data)
            libipp.filter_records_iir(stacked_coeffs, self.filters[0][0].size-1, reshaped_data, self.record_length, num_records, filtered)

            # decimate
            if self.decim_factors[0] > 1:
                filtered = filtered[:, ::self.decim_factors[0]]

        # mix with reference
        # keep real and imaginary separate for filtering below
        filtered_r = self.reference_r * filtered
        filtered_i = self.reference_i * filtered

        # channel selection filters
        for ct in [1,2]:
            if self.filters[ct] == None:
                continue

            coeffs = self.filters[ct]
            stacked_coeffs = np.concatenate(self.filters[ct])
            out_r = np.empty_like(filtered_r)
            out_i = np.empty_like(filtered_i)
            libipp.filter_records_iir(stacked_coeffs, self.filters[ct][0].size-1, filtered_r, filtered_r.shape[-1], num_records, out_r)
            libipp.filter_records_iir(stacked_coeffs, self.filters[ct][0].size-1, filtered_i, filtered_i.shape[-1], num_records, out_i)

            # decimate
            if self.decim_factors[ct] > 1:
                filtered_r = np.copy(out_r[:, ::self.decim_factors[ct]], order="C")
                filtered_i = np.copy(out_i[:, ::self.decim_factors[ct]], order="C")
            else:
                filtered_r = out_r
                filtered_i = out_i

        filtered = filtered_r + 1j*filtered_i

        # recover gain from selecting single sideband
        filtered *= 2

        # push to ouptut connectors
        for os in self.source.output_streams:
            await os.push(filtered)
