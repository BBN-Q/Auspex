# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Filter']

from multiprocessing import Process, Event
import queue
import copy
import numpy as np

from auspex.parameter import Parameter
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
        self._parameters        = []
        self.quince_parameters  = []
        for k,v in dct.items():
            if isinstance(v, InputConnector):
                logger.debug("Found '%s' input connector.", k)
                self._input_connectors.append(k)
            elif isinstance(v, OutputConnector):
                logger.debug("Found '%s' output connector.", k)
                self._output_connectors.append(k)
            elif isinstance(v, Parameter):
                logger.debug("Found '%s' parameter.", k)
                if v.name is None:
                    v.name = k
                self._parameters.append(v)

class Filter(Process, metaclass=MetaFilter):
    """Any node on the graph that takes input streams with optional output streams"""

    def __init__(self, filter_name=None, **kwargs):
        super(Filter, self).__init__()
        self.filter_name = filter_name
        self.input_connectors = {}
        self.output_connectors = {}
        self.parameters = {}
        self.experiment = None # Keep a reference to the parent experiment

        # Event for killing the filter properly
        self.exit = Event()

        # For objectively measuring doneness
        self.finished_processing = False

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
        for param in self._parameters:
            a = copy.deepcopy(param)
            a.parent = self
            self.parameters[param.name] = a
            setattr(self, param.name, a)

    def __repr__(self):
        return "<{} Process (name={})>".format(self.__class__.__name__, self.filter_name)

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

    def on_done(self):
        """To be run when the done signal is received, in case additional steps are
        needed (such as flushing a plot or data)."""
        pass

    def shutdown(self):
        self.exit.set()

    def run(self):
        """
        Generic run method which waits on a single stream and calls `process_data` on any new_data
        """
        logger.debug('Running "%s" run loop', self.filter_name)
        self.finished_processing = False
        input_stream = getattr(self, self._input_connectors[0]).input_streams[0]

        while not self.exit.is_set():
            try:
                message = input_stream.queue.get(True, 0.2)
            except queue.Empty as e:
                continue

            message_type = message['type']
            message_data = message['data']

            # If we receive a message
            if message['type'] == 'event':
                logger.debug('%s "%s" received event "%s"', self.__class__.__name__, self.filter_name, message_data)

                # Propagate along the graph
                for oc in self.output_connectors.values():
                    for os in oc.output_streams:
                        logger.debug('%s "%s" pushed event "%s" to %s, %s', self.__class__.__name__, self.filter_name, message_data, oc, os)
                        os.queue.put(message)

                # Check to see if we're done
                if message['event_type'] == 'done':
                    if not self.finished_processing:
                        logger.warning("Filter {} being asked to finish before being done processing.".format(self.filter_name))
                    self.on_done()
                    break
                elif message['event_type'] == 'refined':
                    self.refine(message_data)

            elif message['type'] == 'data':
                if not hasattr(message_data, 'size'):
                    message_data = np.array([message_data])
                logger.debug('%s "%s" received %d points.', self.__class__.__name__, self.filter_name, message_data.size)
                logger.debug("Now has %d of %d points.", input_stream.points_taken, input_stream.num_points())
                self.process_data(message_data.flatten())

            elif message['type'] == 'data_direct':
                self.process_direct(message_data)

            # If we have gotten all our data and process_data has returned, then we are done!
            if all([v.done() for v in self.input_connectors.values()]):
                self.finished_processing = True

    def process_data(self, data):
        """Process data coming through the filter pipeline"""
        pass

    def process_direct(self, data):
        """Process direct data, ignore things like the data descriptors."""
        pass

    def refine(self, axis):
        """Try to deal with a refinement along the given axes."""
        pass
