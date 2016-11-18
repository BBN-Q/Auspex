# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio, concurrent
import numpy as np

from auspex.parameter import Parameter
from auspex.stream import DataStreamDescriptor
from auspex.filters.filter import Filter, InputConnector, OutputConnector
from auspex.log import logger

class KernelIntegrator(Filter):

    sink   = InputConnector()
    source = OutputConnector()
    kernel = Parameter()

    """Integrate with a given kernel. Kernel will be padded/truncated to match record length"""
    def __init__(self, kernel=None, **kwargs):
        super(KernelIntegrator, self).__init__(**kwargs)
        self.kernel.value = kernel

    def update_descriptors(self):
        if self.kernel.value is None:
            raise ValueError("Integrator was passed kernel None")

        logger.debug('Updating KernelIntegrator "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        #pad or truncate the kernel to match the record length
        record_length = self.sink.descriptor.axes[-1].num_points()

        # This
        if self.kernel.value.size < record_length:
            self.aligned_kernel = np.append(self.kernel.value, np.zeros(record_length-self.kernel.value.size, dtype=np.complex128))
        else:
            self.aligned_kernel = np.resize(self.kernel.value, record_length)
        #zero pad if necessary
        if record_length > len(self.kernel.value):
            self.aligned_kernel[record_length:] = 0.0

        #Integrator reduces and removes axis on output stream
        #update output descriptors
        output_descriptor = DataStreamDescriptor()
        #TODO: handle reduction to single point
        output_descriptor.axes = self.sink.descriptor.axes[:-1]
        for os in self.source.output_streams:
            os.set_descriptor(output_descriptor)
            os.end_connector.update_descriptors()

    async def process_data(self, data):

        #TODO: handle variable partial records
        filtered = np.sum(data * self.aligned_kernel, axis=-1)

        #push to ouptut connectors
        for os in self.source.output_streams:
            await os.push(filtered)
