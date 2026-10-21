# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['ElementwiseFilter']

import queue
import itertools
import numpy as np
import os.path
import time

from auspex.stream import InputConnector, OutputConnector
from auspex.log import logger
import auspex.config as config
from .filter import Filter


class ElementwiseFilter(Filter):
    """Perform elementwise operations on multiple streams:
    e.g. multiply or add all streams element-by-element"""

    sink        = InputConnector()
    source      = OutputConnector()
    filter_name = "GenericElementwise" # To identify subclasses when naming data streams

    def __init__(self, filter_name=None, **kwargs):
        super(ElementwiseFilter, self).__init__(filter_name=filter_name, **kwargs)
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
        logger.debug('Updating %s "%s" descriptors based on input descriptor: %s.', self.filter_name, self.filter_name, self.sink.descriptor)

        # Sometimes not all of the input descriptors have been updated... pause here until they are:
        if None in [ss.descriptor for ss in self.sink.input_streams]:
            logger.debug('%s "%s" waiting for all input streams to be updated.', self.filter_name, self.name)
            return

        self.descriptor = self.sink.descriptor.copy()
        if self.filter_name:
            self.descriptor.data_name = self.filter_name
        if self.descriptor.unit:
            self.descriptor.unit = self.descriptor.unit + "^{}".format(len(self.sink.input_streams))
        self.source.descriptor = self.descriptor
        self.source.update_descriptors()

    def main(self):
        self.done.clear()
        streams = self.sink.input_streams

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to correlator must have matching descriptors.")

        # Buffers for stream data
        stream_data = {s: np.zeros(0, dtype=self.sink.descriptor.dtype) for s in streams}

        # Store whether streams are done
        streams_done      = {s: False for s in streams}
        points_per_stream = {s: 0 for s in streams}

        while not self.exit.is_set():

            # Try to pull all messages in the queue. queue.empty() is not reliable, so we
            # ask for forgiveness rather than permission.
            msgs_by_stream = {s: [] for s in streams}

            for stream in streams[::-1]:
                while not self.exit.is_set():
                    try:
                        msgs_by_stream[stream].append(stream.queue.get(False))
                    except queue.Empty as e:
                        time.sleep(0.002)
                        break

            # Process many messages for each stream
            for stream, messages in msgs_by_stream.items():
                for message in messages:
                    message_type = message['type']
                    # message_data = message['data']
                    # message_data = message_data if hasattr(message_data, 'size') else np.array([message_data])
                    if message_type == 'event':
                        if message['event_type'] == 'done':
                            streams_done[stream] = True
                        elif message['event_type'] == 'refine':
                            logger.warning("ElementwiseFilter doesn't handle refinement yet!")
                    elif message_type == 'data':
                        # Add any old data...
                        message_data = stream.pop()
                        if message_data is not None:
                            points_per_stream[stream] += len(message_data)
                            stream_data[stream] = np.concatenate((stream_data[stream], message_data))
                            # logger.info(f"{stream.name}: {message_data} now {stream_data[stream]}")
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

            # If the amount of data processed is equal to the num points in the stream, we are done
            if np.all([streams_done[stream] for stream in streams]):
                self.push_to_all({"type": "event", "event_type": "done", "data": None})
                self.done.set()
                break
