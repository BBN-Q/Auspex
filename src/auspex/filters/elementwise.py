# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['ElementwiseFilter']

import asyncio, concurrent
import h5py
import itertools
import numpy as np
import os.path
import pickle
import time
import zlib

from auspex.parameter import Parameter, FilenameParameter
from auspex.stream import DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger
import auspex.config as config
from .filter import Filter


class ElementwiseFilter(Filter):
    """Perform elementwise operations on multiple streams:
    e.g. multiply or add all streams element-by-element"""

    sink        = InputConnector()
    source      = OutputConnector()
    filter_name = "GenericElementwise" # To identify subclasses when naming data streams

    def __init__(self, **kwargs):
        super(ElementwiseFilter, self).__init__(**kwargs)
        self.sink.max_input_streams = 100
        self.quince_parameters = []

    def operation(self):
        """Must be overridden with the desired mathematical function"""
        pass

    def unit(self, base_unit):
        """Must be overridden accoriding the desired mathematical function
        e.g. return base_unit + "^{}".format(len(self.sink.input_streams))"""
        pass

    def update_descriptors(self):
        """Must be overridden depending on the desired mathematical function"""
        logger.debug('Updating %s "%s" descriptors based on input descriptor: %s.', self.filter_name, self.name, self.sink.descriptor)

        # Sometimes not all of the input descriptors have been updated... pause here until they are:
        if None in [ss.descriptor for ss in self.sink.input_streams]:
            logger.debug('%s "%s" waiting for all input streams to be updated.', self.filter_name, self.name)
            return

        self.descriptor = self.sink.descriptor.copy()
        self.descriptor.data_name = self.filter_name
        if self.descriptor.unit:
            self.descriptor.unit = self.descriptor.unit + "^{}".format(len(self.sink.input_streams))
        self.source.descriptor = self.descriptor
        self.source.update_descriptors()

    def main(self):
        self.finished_processing = False
        streams = self.sink.input_streams

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to correlator must have matching descriptors.")

        # Buffers for stream data
        stream_data = {s: np.zeros(0, dtype=self.sink.descriptor.dtype) for s in streams}

        # Store whether streams are done
        stream_done = {s: False for s in streams}

        while not self.exit.is_set():
            # Wait for all of the streams to have messages in their queues

            msg_by_stream = {stream: None for stream in streams}
            while any([v is None for v in msg_by_stream.values()]) and not self.exit.is_set():
                for stream in msg_by_stream.keys():
                    if not msg_by_stream[stream]:
                        try:
                            msg_by_stream[stream] = stream.queue.get(True, 0.1)
                        except queue.Empty as e:
                            continue

            # futures = {
            #     asyncio.ensure_future(stream.queue.get()): stream
            #     for stream in streams
            # }

            # # Deal with non-equal number of messages using timeout
            # responses, pending = asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED, timeout=2.0)

            # # Construct the inverse lookup, results in {stream: result}
            # stream_results = {futures[res]: res.result() for res in list(responses)}

            # # Cancel the futures
            # for pend in list(pending):
            #     pend.cancel()

            # Add any new data to the
            for stream, message in msg_by_stream.items():
                message_type = message['type']
                message_data = message['data']
                message_data = message_data if hasattr(message_data, 'size') else np.array([message_data])
                if message_type == 'event':
                    if message['event_type'] == 'done':
                        stream_done[stream] = True
                    elif message['event_type'] == 'refine':
                        logger.warning("ElementwiseFilter doesn't handle refinement yet!")

                elif message_type == 'data':
                    stream_data[stream] = np.concatenate((stream_data[stream], message_data.flatten()))

            if False not in stream_done.values():
                for oc in self.output_connectors.values():
                    for os in oc.output_streams:
                        os.push_event("done")
                logger.debug('%s "%s" is done', self.__class__.__name__, self.name)
                break

            # Now process the data with the elementwise operation
            smallest_length = min([d.size for d in stream_data.values()])
            new_data = [d[:smallest_length] for d in stream_data.values()]
            result = new_data[0]
            for nd in new_data[1:]:
                result = self.operation()(result, nd)
            if result.size > 0:
                self.source.push(result)

            # Add data to carry_data if necessary
            for stream in stream_data.keys():
                if stream_data[stream].size > smallest_length:
                    stream_data[stream] = stream_data[stream][smallest_length:]
                else:
                    stream_data[stream] = np.zeros(0, dtype=self.sink.descriptor.dtype)

            # If we have gotten all our data and process_data has returned, then we are done!
            if all([v.done() for v in self.input_connectors.values()]):
                self.finished_processing = True
