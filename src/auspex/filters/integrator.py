# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from .filter import Filter
from auspex.parameter import Parameter, FloatParameter, IntParameter, BoolParameter
from auspex.stream import DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger

class KernelIntegrator(Filter):

    sink   = InputConnector()
    source = OutputConnector()
    kernel = Parameter()
    bias   = FloatParameter()
    simple_kernel = BoolParameter()
    box_car_start = FloatParameter()
    box_car_stop = FloatParameter()
    frequency = FloatParameter()

    """Integrate with a given kernel. Kernel will be padded/truncated to match record length"""
    def __init__(self, **kwargs):
        super(KernelIntegrator, self).__init__(**kwargs)
        if len(kwargs) > 0:
            self.kernel.value = kwargs['kernel']
            self.bias.value = kwargs['bias']
            self.simple_kernel.value = kwargs['simple_kernel']
            self.box_car_start.value = kwargs['box_car_start']
            self.box_car_stop.value = kwargs['box_car_stop']
            self.frequency.value = kwargs['frequency']

    def update_descriptors(self):
        if self.kernel.value is None:
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
        else:
            kernel = eval(self.kernel.value.encode('unicode_escape'))
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
        output_descriptor.exp_src = self.sink.descriptor.exp_src
        output_descriptor.dtype = self.sink.descriptor.dtype
        for os in self.source.output_streams:
            os.set_descriptor(output_descriptor)
            os.end_connector.update_descriptors()

    async def process_data(self, data):

        # TODO: handle variable partial records
        filtered = np.inner(np.reshape(data, (-1, len(self.aligned_kernel))), self.aligned_kernel)

        # push to ouptut connectors
        for os in self.source.output_streams:
            await os.push(filtered)
