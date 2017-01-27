# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio, concurrent
import itertools
import h5py
import pickle
import zlib
import numpy as np
import os.path
import time

from auspex.parameter import Parameter, FilenameParameter
from auspex.stream import DataStreamDescriptor
from auspex.log import logger
from auspex.filters.filter import Filter, InputConnector, OutputConnector

class Correlator(Filter):
    sink   = InputConnector()
    source = OutputConnector()

    def __init__(self, **kwargs):
        super(Correlator, self).__init__(**kwargs)
        self.sink.max_input_streams = 100
        self.quince_parameters = []

    def update_descriptors(self):
        logger.debug('Updating correlator "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        # Sometimes not all of the input descriptors have been updated... pause here until they are:
        if None in [ss.descriptor for ss in self.sink.input_streams]:
            logger.debug('Correlator "%s" waiting for all input streams to be updated.', self.name)
            return

        descriptor = self.sink.descriptor.copy()
        descriptor.data_name = "Correlator"
        if descriptor.unit:
            descriptor.unit = descriptor.unit + "^{}".format(len(self.sink.input_streams))
        self.source.descriptor = descriptor
        self.source.update_descriptors()

    async def run(self):
        streams = self.sink.input_streams
        stream  = streams[0]

        for s in streams[1:]:
            if not np.all(s.descriptor.tuples() == streams[0].descriptor.tuples()):
                raise ValueError("Multiple streams connected to correlator must have matching descriptors.")

        # Write pointers
        carry_data = [np.zeros(0, dtype=self.sink.descriptor.dtype) for s in streams]

        # Store whether streams are done
        streams_done = [False for s in streams]

        while True:
            # Wait for all of the acquisition to complete
            # Against at least some peoples rational expectations, asyncio.wait doesn't return Futures
            # in the order of the iterable it was passed, but perhaps just in order of completion. So,
            # we construct a dictionary in order that that can be mapped back where we need them:

            futures = {
                asyncio.ensure_future(stream.queue.get()): stream
                for stream in streams
            }

            # Deal with non-equal number of messages using timeout
            responses, pending = await asyncio.wait(futures, timeout=0.1)

            # Construct the inverse lookup
            response_for_stream = {futures[res]: res.result() for res in list(responses)}

            # Add dummy data when tasks didn't complete (this just means they are done already)
            pending = list(pending)
            if len(pending) > 0:
                for pend in pending:
                    response_for_stream[futures[pend]] = {'type': 'data', 'compression': False, 'data': np.array([])}
                    pend.cancel()

            messages = [response_for_stream[stream] for stream in streams]

            # Allow different message types since data may arrive out of order.
            message_types = [m['type'] for m in messages]
            message_data  = [message['data'] for message in messages]
            message_comp  = [message['compression'] for message in messages]
            message_data  = [pickle.loads(zlib.decompress(dat)) if comp == 'zlib' else dat for comp, dat in zip(message_comp, message_data)]
            message_data  = [dat if hasattr(dat, 'size') else np.array([dat]) for dat in message_data]  # Convert single values to arrays

            # Record doneness and set anything done to have a blank array as data.
            if 'event' in message_types:
                for ii in range(len(message_types)):
                    if message_types[ii] == 'event' and message_data[ii] == 'done':
                        streams_done[ii] = True
                        message_data[ii] = np.array([])

                # Propagate doneness along the graph when all streams complete
                if False not in streams_done:
                    for oc in self.output_connectors.values():
                        for os in oc.output_streams:
                            await os.push_event("done")
                    logger.debug('%s "%s" is done', self.__class__.__name__, self.name)
                    break

            if 'data' in message_types:
                for ii in range(1, len(message_data)):
                    message_data[ii] = message_data[ii].flatten()

                # Take care of any carried data
                for ii in range(len(streams)):
                    carry = carry_data[ii]
                    if carry.size > 0:
                        message_data[ii] = np.concatenate((carry, message_data[ii]))

                lengths = [d.size for d in message_data]
                smallest_length = min(lengths)
                product = np.ones(smallest_length)

                # Construct the product up the greatest common index
                for data in message_data:
                    product = product*data[:smallest_length]
                if product.size > 0:
                    await self.source.push(product)

                # Add data to carry_data if necessary
                for ii in range(len(streams)):
                    if lengths[ii] > smallest_length:
                        carry_data[ii] = message_data[ii][smallest_length:]
                    else:
                        carry_data[ii] = np.zeros(0, dtype=self.sink.descriptor.dtype)
