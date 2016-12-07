# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio, concurrent
import numpy as np

from auspex.parameter import Parameter, FloatParameter, IntParameter, BoolParameter
from auspex.stream import DataStreamDescriptor
from auspex.filters.filter import Filter, InputConnector, OutputConnector
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
    def __init__(self, kernel=None, **kwargs):
        super(KernelIntegrator, self).__init__(**kwargs)
        self.kernel.value = kernel

    def update_descriptors(self):
        if self.kernel.value is None:
            raise ValueError("Integrator was passed kernel None")

        logger.debug('Updating KernelIntegrator "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        # pad or truncate the kernel to match the record length
        record_length = self.sink.descriptor.axes[-1].num_points()
        if self.kernel.value.size < record_length:
            self.aligned_kernel = np.append(self.kernel.value, np.zeros(record_length-self.kernel.value.size, dtype=np.complex128))
        else:
            self.aligned_kernel = self.kernel.value.resize(record_length)

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
