# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import zlib
import pickle
from concurrent.futures import FIRST_COMPLETED

from auspex.stream import DataStream, InputConnector, OutputConnector
from auspex.log import logger

class MetaFilter(type):
    """Meta class to bake the input/output connectors into a Filter class description
    """
    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding connectors to %s", name)
        self._input_connectors  = []
        self._output_connectors = []
        for k,v in dct.items():
            if isinstance(v, InputConnector):
                logger.debug("Found '%s' input connector.", k)
                self._input_connectors.append(k)
            elif isinstance(v, OutputConnector):
                logger.debug("Found '%s' output connector.", k)
                self._output_connectors.append(k)

class Filter(metaclass=MetaFilter):
    """Any node on the graph that takes input streams with optional output streams"""
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.input_connectors = {}
        self.output_connectors = {}

        # For signaling to Quince that something is wrong
        self.out_of_spec = False

        for ic in self._input_connectors:
            a = InputConnector(name=ic, parent=self)
            a.parent = self
            self.input_connectors[ic] = a
            setattr(self, ic, a)
        for oc in self._output_connectors:
            a = OutputConnector(name=oc, parent=self)
            a.parent = self
            self.output_connectors[oc] = a
            setattr(self, oc, a)

    def __repr__(self):
        return "<{}(name={})>".format(self.__class__.__name__, self.name)

    def update_descriptors(self):
        """This method is called whenever the connectivity of the graph changes. This may have implications
        for the internal functioning of the filter, in which case update_descriptors should be overloaded. 
        Any simple changes to the axes within the StreamDescriptors should take place via the class method
        descriptor_map."""
        self.out_of_spec = False

        input_descriptors  = {k: v.descriptor for k,v in self.input_connectors.items()}
        output_descriptors = self.descriptor_map(input_descriptors)
        
        for name, descriptor in output_descriptors.items():
            if name in self.output_connectors:
                self.output_connectors[name].descriptor = descriptor
                self.output_connectors[name].update_descriptors()

    def descriptor_map(self, input_descriptors):
        """Return a dict of the output descriptors."""
        return {'source': v for v in input_descriptors.values()}

    async def on_done(self):
        """To be run when the done signal is received, in case additional steps are
        needed (such as flushing a plot or data)."""
        pass

    async def run(self):
        """
        Generic run method which waits on a single stream and calls `process_data` on any new_data
        """
        logger.debug('Running "%s" run loop', self.name)

        input_stream = getattr(self, self._input_connectors[0]).input_streams[0]

        while True:

            message = await input_stream.queue.get()
            message_type = message['type']
            message_data = message['data']
            message_comp = message['compression']

            if message_comp == 'zlib':
                message_data = pickle.loads(zlib.decompress(message_data))
            # If we receive a message
            if message['type'] == 'event':
                logger.debug('%s "%s" received event "%s"', self.__class__.__name__, self.name, message_data)

                # Propagate along the graph
                for oc in self.output_connectors.values():
                    for os in oc.output_streams:
                        logger.debug('%s "%s" pushed event "%s" to %s, %s', self.__class__.__name__, self.name, message_data, oc, os)
                        await os.queue.put(message)

                # Check to see if we're done
                if message['data'] == 'done':
                    await self.on_done()
                    break

            elif message['type'] == 'data':
                logger.debug('%s "%s" received %d points.', self.__class__.__name__, self.name, message_data.size)
                logger.debug("Now has %d of %d points.", input_stream.points_taken, input_stream.num_points())
                await self.process_data(message_data)

    async def process_data(self, data):
        """Generic pass through.  """
        return data
