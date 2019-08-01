# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Channelizer']

import os
import platform
from copy import deepcopy

import numpy as np
import scipy.signal

from .filter import Filter
from auspex.parameter import Parameter, IntParameter, FloatParameter
from auspex.stream import  DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger

try:
    # load libchannelizer to access Intel IPP filtering functions
    import numpy.ctypeslib as npct
    from ctypes import c_int, c_size_t
    np_float  = npct.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')

    libchannelizer_path = os.path.abspath(os.path.join( os.path.dirname(__file__), "libchannelizer"))
    if "Windows" in platform.platform():
        os.environ["PATH"] += ";" + libchannelizer_path
    libipp = npct.load_library("libchannelizer",  libchannelizer_path)
    libipp.filter_records_fir.argtypes = [np_float, c_size_t, c_int, np_float, c_size_t, c_size_t, np_float]
    libipp.filter_records_iir.argtypes = [np_float, c_size_t, np_float, c_size_t, c_size_t, np_float]
    libipp.init()

    load_fallback = False
except:
    logger.warning("Could not load channelizer library; falling back to python methods.")
    load_fallback = True


class Channelizer(Filter):
    """Digital demodulation and filtering to select a particular frequency multiplexed channel. If
    an axis name is supplied to `follow_axis` then the filter will demodulate at the freqency
    `axis_frequency_value - follow_freq_offset` otherwise it will demodulate at `frequency`. Note that
    the filter coefficients are still calculated with respect to the `frequency` paramter, so it should
    be chosen accordingly when `follow_axis` is defined."""

    sink               = InputConnector()
    source             = OutputConnector()
    follow_axis        = Parameter(default="") # Name of the axis to follow
    follow_freq_offset = FloatParameter(default=0.0) # Offset
    decimation_factor  = IntParameter(value_range=(1,100), default=4, snap=1)
    frequency          = FloatParameter(value_range=(-10e9,10e9), increment=1.0e6, default=10e6)
    bandwidth          = FloatParameter(value_range=(0.00, 100e6), increment=0.1e6, default=5e6)

    def __init__(self, frequency=None, bandwidth=None, decimation_factor=None,
                    follow_axis=None, follow_freq_offset=None, **kwargs):
        super(Channelizer, self).__init__(**kwargs)
        if frequency:
            self.frequency.value = frequency
        if bandwidth:
            self.bandwidth.value = bandwidth
        if decimation_factor:
            self.decimation_factor.value = decimation_factor
        if follow_axis:
            self.follow_axis.value = follow_axis
        if follow_freq_offset:
            self.follow_freq_offset.value = follow_freq_offset
        self.quince_parameters = [self.decimation_factor, self.frequency, self.bandwidth]
        self._phase = 0.0

    def final_init(self):
        self.init_filters(self.frequency.value, self.bandwidth.value)

        if self.follow_axis.value is not "":
            desc = self.sink.descriptor
            axis_num = desc.axis_num(self.follow_axis.value)
            self.pts_before_freq_update = desc.num_points_through_axis(axis_num + 1)
            self.pts_before_freq_reset  = desc.num_points_through_axis(axis_num)
            self.demod_freqs = desc.axes[axis_num].points - self.follow_freq_offset.value
            self.current_freq = 0
            self.update_references(self.current_freq)
        self.idx = 0

        # For storing carryover if getting uneven buffers
        self.carry = np.zeros(0, dtype=self.source.descriptor.dtype)


    def update_references(self, frequency):
        # store decimated reference for mix down
        # phase_drift = 2j*np.pi*0.5e-6 * (abs(frequency) - 100e6)
        ref = np.exp(2j*np.pi * -frequency * self.time_pts[::self.d1] + 1j*self._phase, dtype=np.complex64)

        self.reference   = ref
        self.reference_r = np.real(ref)
        self.reference_i = np.imag(ref)

    def init_filters(self, frequency, bandwidth):
        # convert bandwidth normalized to Nyquist interval
        n_bandwidth = bandwidth * self.time_step * 2
        n_frequency = abs(frequency) * self.time_step * 2

        # arbitrarily decide on three stage filter pipeline
        # 1. first stage decimating filter on real data
        # 2. second stage decimating filter on mixed product to boost n_bandwidth
        # 3. final channel selecting filter at n_bandwidth/2

        # anecdotally don't decimate more than a factor of eight for stability

        self.decim_factors = [1]*3
        self.filters = [None]*3

        # first stage decimating filter
        # maximize first stage decimation:
        #     * minimize subsequent stages time taken
        #     * filter and decimate while signal is still real
        #     * first stage decimation cannot be too large or then 2omega signal from mixing will alias
        self.d1 = 1
        while (self.d1 < 8) and (2*n_frequency <= 0.8/self.d1) and (self.d1 < self.decimation_factor.value):
            self.d1 *= 2
            n_bandwidth *= 2
            n_frequency *= 2

        if self.d1 > 1:
            # create an anti-aliasing filter
            # pass-band to 0.8 * decimation factor; anecdotally single precision needs order <= 4 for stability
            b,a = scipy.signal.cheby1(4, 3, 0.8/self.d1)
            b = np.float32(b)
            a = np.float32(a)
            self.decim_factors[0] = self.d1
            self.filters[0]  = (b,a)

        # store decimated reference for mix down
        self.update_references(frequency)

        # second stage filter to bring n_bandwidth/2 up
        # decimation cannot be too large or will impinge on channel bandwidth (keep n_bandwidth/2 <= 0.8)
        self.d2 = 1
        while (self.d2 < 8) and ((self.d1*self.d2) < self.decimation_factor.value) and (n_bandwidth/2 <= 0.8):
            self.d2 *= 2
            n_bandwidth *= 2
            n_frequency *= 2

        if self.d2 > 1:
            # create an anti-aliasing filter
            # pass-band to 0.8 * decimation factor; anecdotally single precision needs order <= 4 for stability
            b,a = scipy.signal.cheby1(4, 3, 0.8/self.d2)
            b = np.float32(b)
            a = np.float32(a)
            self.decim_factors[1] = self.d2
            self.filters[1]  = (b,a)


        # final channel selection filter
        if n_bandwidth < 0.1:
            raise ValueError("Insufficient decimation to achieve stable filter: {}.".format(n_bandwidth))

        b,a = scipy.signal.cheby1(4, 3, n_bandwidth/2)
        b = np.float32(b)
        a = np.float32(a)
        self.decim_factors[2] = self.decimation_factor.value // (self.d1*self.d2)
        self.filters[2]  = (b,a)

    def update_descriptors(self):
        logger.debug('Updating Channelizer "%s" descriptors based on input descriptor: %s.', self.filter_name, self.sink.descriptor)

        # extract record time sampling
        self.time_pts = self.sink.descriptor.axes[-1].points
        self.record_length = len(self.time_pts)
        self.time_step = self.time_pts[1] - self.time_pts[0]
        logger.debug("Channelizer time_step = {}".format(self.time_step))

        # We will be decimating along a time axis, which is always
        # going to be the last axis given the way we usually take data.
        # TODO: perform this function along a named axis rather than a numbered axis
        # in case something about this changes.

        # update output descriptors
        decimated_descriptor = DataStreamDescriptor()
        decimated_descriptor.axes = self.sink.descriptor.axes[:]
        decimated_descriptor.axes[-1] = deepcopy(self.sink.descriptor.axes[-1])
        decimated_descriptor.axes[-1].points = self.sink.descriptor.axes[-1].points[self.decimation_factor.value-1::self.decimation_factor.value]
        decimated_descriptor.axes[-1].original_points = decimated_descriptor.axes[-1].points
        decimated_descriptor._exp_src = self.sink.descriptor._exp_src
        decimated_descriptor.dtype = np.complex64
        self.source.descriptor = decimated_descriptor
        self.source.update_descriptors()

    def process_data(self, data):

        # Append any data carried from the last run
        if self.carry.size > 0:
            data = np.concatenate((self.carry, data))

        # This is the largest number of records we can handle
        num_records = data.size // self.record_length

        # This is the carryover that we'll store until next round.
        # If nothing is left then reset the carryover.
        remaining_points = data.size % self.record_length
        if remaining_points > 0:
            if num_records > 0:
                self.carry = data[-remaining_points:]
                data = data[:-remaining_points]
            else:
                self.carry = data
        else:
            self.carry = np.zeros(0, dtype=self.source.descriptor.dtype)

        if num_records > 0:
            # The records are processed in parallel after being reshaped here
            reshaped_data = np.reshape(data, (num_records, self.record_length), order="C")

            # Update demodulation frequency if necessary
            if self.follow_axis.value is not "":
                freq = self.demod_freqs[(self.idx % self.pts_before_freq_reset) // self.pts_before_freq_update]
                if freq != self.current_freq:
                    self.update_references(freq)
                    self.current_freq = freq

            self.idx += data.size

            # first stage decimating filter
            if self.filters[0] is None:
                filtered = reshaped_data
            else:
                stacked_coeffs = np.concatenate(self.filters[0])
                # filter
                if np.iscomplexobj(reshaped_data):
                    # TODO: compile complex versions of the IPP functions
                    filtered_r = np.empty_like(reshaped_data, dtype=np.float32)
                    filtered_i = np.empty_like(reshaped_data, dtype=np.float32)
                    libipp.filter_records_iir(stacked_coeffs, self.filters[0][0].size-1, np.ascontiguousarray(reshaped_data.real.astype(np.float32)), self.record_length, num_records, filtered_r)
                    libipp.filter_records_iir(stacked_coeffs, self.filters[0][0].size-1, np.ascontiguousarray(reshaped_data.imag.astype(np.float32)), self.record_length, num_records, filtered_i)
                    filtered = filtered_r + 1j*filtered_i
                    # decimate
                    if self.decim_factors[0] > 1:
                        filtered = filtered[:, ::self.decim_factors[0]]
                else:
                    filtered = np.empty_like(reshaped_data, dtype=np.float32)
                    libipp.filter_records_iir(stacked_coeffs, self.filters[0][0].size-1, np.ascontiguousarray(reshaped_data.real.astype(np.float32)), self.record_length, num_records, filtered)

                    # decimate
                    if self.decim_factors[0] > 1:
                        filtered = filtered[:, ::self.decim_factors[0]]

            # mix with reference
            # keep real and imaginary separate for filtering below
            if np.iscomplexobj(reshaped_data):
                filtered *= self.reference
                filtered_r = filtered.real
                filtered_i = filtered.imag
            else:
                filtered_r = self.reference_r * filtered
                filtered_i = self.reference_i * filtered

            # channel selection filters
            for ct in [1,2]:
                if self.filters[ct] == None:
                    continue

                coeffs = self.filters[ct]
                stacked_coeffs = np.concatenate(self.filters[ct])
                out_r = np.empty_like(filtered_r).astype(np.float32)
                out_i = np.empty_like(filtered_i).astype(np.float32)
                libipp.filter_records_iir(stacked_coeffs, self.filters[ct][0].size-1, np.ascontiguousarray(filtered_r.astype(np.float32)), filtered_r.shape[-1], num_records, out_r)
                libipp.filter_records_iir(stacked_coeffs, self.filters[ct][0].size-1, np.ascontiguousarray(filtered_i.astype(np.float32)), filtered_i.shape[-1], num_records, out_i)

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
                os.push(filtered)

class LibChannelizerFallback(object):
    @staticmethod
    def filter_records_fir(coeffs,
                           num_taps, # ignored
                           decim_factor,
                           recs,
                           record_length, # ignored (uses shape of recs)
                           num_records, # ignored (uses shape of recs)
                           result):
        # split out a, b coefficients
        b = coeffs[:len(coeffs)//2]
        a = coeffs[len(coeffs)//2:]

        filtered_signal = scipy.signal.lfilter(b, a, recs)
        result[:] = np.copy(filtered_signal[:, ::decim_factor], order="C")

    @staticmethod
    def filter_records_iir(coeffs,
                           order, # ignored
                           recs,
                           record_length, # ignored (uses shape of recs)
                           num_records, # ignored (uses shape of recs)
                           result):
        # split out a, b coefficients
        b = coeffs[:len(coeffs)//2]
        a = coeffs[len(coeffs)//2:]

        result[:] = scipy.signal.lfilter(b, a, recs)

if load_fallback:
    libipp = LibChannelizerFallback()
