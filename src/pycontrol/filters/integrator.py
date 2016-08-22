import asyncio, concurrent

import numpy as np

from pycontrol.stream import DataStreamDescriptor
from pycontrol.filters.filter import Filter, InputConnector, OutputConnector
from pycontrol.logging import logger

class KernelIntegrator(Filter):

    sink = InputConnector()
    source = OutputConnector()

    """Integrate with a given kernel. Kernel will be padded/truncated to match record length"""
    def __init__(self, kernel, **kwargs):
        super(KernelIntegrator, self).__init__(**kwargs)
        self.kernel = kernel

    def update_descriptors(self):
        logger.debug('Updating KernelIntegrator "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        #pad or truncate the kernel to match the record length
        record_length = self.sink.descriptor.axes[-1].num_points()

        self.aligned_kernel = np.resize(self.kernel, record_length)
        #zero pad if necessary
        if record_length > len(self.kernel):
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
