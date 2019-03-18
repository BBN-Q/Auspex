# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['KernelIntegrator', 'WindowIntegrator']

import os
import numpy as np
from scipy.signal import chebwin, blackman, slepian, convolve

from .filter import Filter
from auspex.parameter import Parameter, FloatParameter, IntParameter, BoolParameter
from auspex.stream import DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger
import auspex.config as config

class KernelIntegrator(Filter):

    sink   = InputConnector()
    source = OutputConnector()
    kernel = Parameter()
    bias   = FloatParameter(default=0.0)
    simple_kernel = BoolParameter(default=True)
    box_car_start = FloatParameter(default=0.0)
    box_car_stop = FloatParameter(default=100e-9)
    frequency = FloatParameter(default=0.0)

    """Integrate with a given kernel. Kernel will be padded/truncated to match record length"""
    def __init__(self, **kwargs):
        super(KernelIntegrator, self).__init__(**kwargs)
        self.pre_int_op  = None
        self.post_int_op = None
        for k, v in kwargs.items():
            if hasattr(self, k) and isinstance(getattr(self,k), Parameter):
                getattr(self, k).value = v
        if "pre_integration_operation" in kwargs:
            self.pre_int_op = kwargs["pre_integration_operation"]
        if "post_integration_operation" in kwargs:
            self.post_int_op = kwargs["post_integration_operation"]
        self.quince_parameters = [self.simple_kernel, self.frequency, self.box_car_start, self.box_car_stop]

    def update_descriptors(self):
        if not self.simple_kernel and self.kernel.value is None:
            raise ValueError("Integrator was passed kernel None")

        logger.debug('Updating KernelIntegrator "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        record_length = self.sink.descriptor.axes[-1].num_points()
        if self.simple_kernel.value:
            time_pts = self.sink.descriptor.axes[-1].points
            time_step = time_pts[1] - time_pts[0]
            kernel = np.zeros(record_length, dtype=np.complex128)
            sample_start = int(self.box_car_start.value / time_step)
            sample_stop = int(self.box_car_stop.value / time_step) + 1
            kernel[sample_start:sample_stop] = 1.0
            # add modulation
            kernel *= np.exp(2j * np.pi * self.frequency.value * time_step * time_pts)
        elif os.path.exists(os.path.join(config.KernelDir, self.kernel.value+'.txt')):
            kernel = np.loadtxt(os.path.join(config.KernelDir, self.kernel.value+'.txt'), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
        else:
            try:
                kernel = eval(self.kernel.value.encode('unicode_escape'))
            except:
                raise ValueError('Kernel invalid. Provide a file name or an expression to evaluate')
        # pad or truncate the kernel to match the record length
        if kernel.size < record_length:
            self.aligned_kernel = np.append(kernel, np.zeros(record_length-kernel.size, dtype=np.complex128))
        else:
            self.aligned_kernel = np.resize(kernel, record_length)

        # Integrator reduces and removes axis on output stream
        # update output descriptors
        output_descriptor = DataStreamDescriptor()
        # TODO: handle reduction to single point
        output_descriptor.axes = self.sink.descriptor.axes[:-1]
        output_descriptor._exp_src = self.sink.descriptor._exp_src
        output_descriptor.dtype = np.complex128
        for ost in self.source.output_streams:
            ost.set_descriptor(output_descriptor)
            ost.end_connector.update_descriptors()

    async def process_data(self, data):

        # TODO: handle variable partial records
        if self.pre_int_op:
            data = self.pre_int_op(data)
        filtered = np.inner(np.reshape(data, (-1, len(self.aligned_kernel))), self.aligned_kernel)
        if self.post_int_op:
            filtered = self.post_int_op(filtered)
        # push to ouptut connectors
        for ost in self.source.output_streams:
            await ost.push(filtered)

class WindowIntegrator(Filter):
    """
    Allow a kernel from the set {'chebwin', 'blackman', 'slepian',
    'boxcar'} to be set for the duration of the start and stop values.

    YAML parameters are:
    type: WindowIntegrator
    source: Demod-q1
    kernel_type: 'chebwin'
    start: 5.0e-07
    stop: 9.0e-07

    See: https://docs.scipy.org/doc/scipy/reference/signal.html for more
    details on the filters specifics.
    """

    sink   = InputConnector()
    source = OutputConnector()
    bias   = FloatParameter(default=0.0)
    kernel_type = Parameter(default='boxcar', allowed_values=['chebwin',\
        'blackman', 'slepian', 'boxcar'])
    start = FloatParameter(default=0.0)
    stop = FloatParameter(default=100e-9)
    frequency = FloatParameter(default=0.0)

    """Integrate with a given kernel. Kernel will be padded/truncated to match record length"""
    def __init__(self, **kwargs):
        super(WindowIntegrator, self).__init__(**kwargs)
        self.pre_int_op  = None
        self.post_int_op = None
        for k, v in kwargs.items():
            if hasattr(self, k) and isinstance(getattr(self,k), Parameter):
                getattr(self, k).value = v
        if "pre_integration_operation" in kwargs:
            self.pre_int_op = kwargs["pre_integration_operation"]
        if "post_integration_operation" in kwargs:
            self.post_int_op = kwargs["post_integration_operation"]
        self.quince_parameters = [self.kernel_type, self.frequency, self.start, self.stop]

    def update_descriptors(self):
        if not self.kernel_type:
            raise ValueError("Integrator was passed kernel None")

        logger.debug('Updating WindowIntegrator "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        record_length = self.sink.descriptor.axes[-1].num_points()

        time_pts = self.sink.descriptor.axes[-1].points
        time_step = time_pts[1] - time_pts[0]
        kernel = np.zeros(record_length, dtype=np.complex128)
        sample_start = int(self.box_car_start.value / time_step)
        sample_stop = int(self.box_car_stop.value / time_step) + 1
        if self.kernel_type == 'boxcar':
            kernel[sample_start:sample_stop] = 1.0
        elif self.kernel_type == 'chebwin':
            # create a Dolph-Chebyshev window with 100 dB attenuation
            kernel[sample_start:sample_stop] = \
                chebwin(sample_start-sample_stop, at=100)
        elif self.kernel_type == 'blackman':
            kernel[sample_start:sample_stop] = \
                blackman(sample_start-sample_stop)
        elif self.kernel_type == 'slepian':
            # create a Slepian window with 0.2 bandwidth
            kernel[sample_start:sample_stop] = \
                slepian(sample_start-sample_stop, width=0.2)

        # add modulation
        kernel *= np.exp(2j * np.pi * self.frequency.value * time_step * time_pts)

        # pad or truncate the kernel to match the record length
        if kernel.size < record_length:
            self.aligned_kernel = np.append(kernel, np.zeros(record_length-kernel.size, dtype=np.complex128))
        else:
            self.aligned_kernel = np.resize(kernel, record_length)

        # Integrator reduces and removes axis on output stream
        # update output descriptors
        output_descriptor = DataStreamDescriptor()
        # TODO: handle reduction to single point
        output_descriptor.axes = self.sink.descriptor.axes[:-1]
        output_descriptor._exp_src = self.sink.descriptor._exp_src
        output_descriptor.dtype = np.complex128
        for os in self.source.output_streams:
            os.set_descriptor(output_descriptor)
            os.end_connector.update_descriptors()

    async def process_data(self, data):

        # TODO: handle variable partial records
        if self.pre_int_op:
            data = self.pre_int_op(data)
        filtered = np.inner(np.reshape(data, (-1, len(self.aligned_kernel))), self.aligned_kernel)
        if self.post_int_op:
            filtered = self.post_int_op(filtered)
        # push to ouptut connectors
        for os in self.source.output_streams:
            await os.push(filtered)
