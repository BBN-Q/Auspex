from copy import deepcopy
import asyncio, concurrent

import numpy as np
from scipy.signal import firwin, lfilter


from pycontrol.filters.filter import Filter, InputConnector, OutputConnector
from pycontrol.stream import  DataStreamDescriptor
from pycontrol.logging import logger

class Channelizer(Filter):
    """Digital demodulation and filtering to select a particular frequency multiplexed channel"""

    sink = InputConnector()
    source = OutputConnector()

    def __init__(self, freq, cutoff, decimation_factor=1, **kwargs):
        super(Channelizer, self).__init__(**kwargs)
        self.freq = freq
        self.cutoff = cutoff
        self.decimation_factor = decimation_factor

    def update_descriptors(self):
        logger.debug('Updating Channelizer "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        #extract record time sampling
        time_pts = self.sink.descriptor.axes[-1].points
        self.record_length = len(time_pts)
        self.time_step = time_pts[1] - time_pts[0]
        logger.debug("Channelizer time_step = {}".format(self.time_step))

        #store refernece for mix down
        self.reference = np.exp(2j*np.pi * self.freq * self.time_step * np.arange(self.record_length))

        #store filter coefficients
        #TODO: arbitrary 64 tap filter
        if self.decimation_factor > 1:
            self.filter = firwin(64, self.cutoff, window='hamming')
        else:
            self.filter = np.array([1.0])

        #update output descriptors
        decimated_descriptor = DataStreamDescriptor()
        decimated_descriptor.axes = self.sink.descriptor.axes[:]
        decimated_descriptor.axes[-1] = deepcopy(self.sink.descriptor.axes[-1])
        decimated_descriptor.axes[-1].points = self.sink.descriptor.axes[-1].points[self.decimation_factor-1::self.decimation_factor]
        for os in self.source.output_streams:
            os.set_descriptor(decimated_descriptor)
            os.end_connector.update_descriptors()

    async def run(self):

        input_stream = self.sink.input_streams[0]

        while True:

            get_finished_task = asyncio.ensure_future(input_stream.finished())
            get_data_task = asyncio.ensure_future(input_stream.queue.get())

            done, pending = await asyncio.wait((get_finished_task, get_data_task),
                                     return_when=concurrent.futures.FIRST_COMPLETED)

            #since done contains first completed it will be a set of size 1
            if get_finished_task in done and input_stream.queue.empty():
                logger.info('No more data for channelizer "%s"', self.name)
                pending.pop().cancel()
                break

            pending.pop().cancel()
            new_data = done.pop().result()

            #Assume for now we get a integer number of records at a time
            #TODO: handle  and partial records
            num_records = new_data.size // self.record_length

            #mix with reference
            mix_product = self.reference * np.reshape(new_data, (num_records, self.record_length), order="C")

            #filter then decimate
            #TODO: polyphase filterting should provide better performance
            filtered = lfilter(self.filter, 1.0, mix_product)
            filtered = filtered[:, self.decimation_factor-1::self.decimation_factor]

            #push to ouptut connectors
            for os in self.source.output_streams:
                await os.push(filtered.real)
