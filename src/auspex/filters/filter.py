# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Filter']

import os
import sys
import psutil

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    from threading import Thread as Process
    from threading import Event
    from queue import Queue
else:
    from multiprocessing import Process
    from multiprocessing import Event
    from multiprocessing import Queue

import cProfile
import itertools
import time, datetime
import queue
import copy
import numpy as np

from auspex.parameter import Parameter
from auspex.stream import DataStream, InputConnector, OutputConnector
from auspex.log import logger
import auspex.config

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

    def __init__(self, name=None, **kwargs):
        super(Filter, self).__init__()
        self.filter_name = name
        self.input_connectors = {}
        self.output_connectors = {}
        self.parameters = {}

        # Event for killing the filter properly
        self.exit = Event()
        self.done = Event()

        # Keep track of data throughput
        self.processed = 0

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

        # For sending performance information
        self.last_performance_update = datetime.datetime.now()
        self.beginning = datetime.datetime.now()
        self.perf_queue = None

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
        self.p = psutil.Process(os.getpid())
        if auspex.config.profile:
            if not self.filter_name:
                name = "Unlabeled"
            else:
                name = self.filter_name
            cProfile.runctx('self.main()', globals(), locals(), 'prof-%s-%s.prof' % (self.__class__.__name__, name))
        else:
            self.main()

        # # # Make sure queues are flushed out completely:
        # # logger.info("Closing out queues to prevent hang")
        # # for ic in self.input_connectors.values():
        # #     for ist in ic.input_streams:
        # #         ist.queue.close()
        # for ic in self.input_connectors.values():
        #     for ist in ic.input_streams:
        #         abc = 0
        #         while True:
        #             try:
        #                 ist.queue.get(0.01)
        #                 abc += 1
        #                 logger.info(f"{self}: drained {abc} messages...")
        #             except queue.Empty as e:
        #                 time.sleep(0.002)
        #                 break
        self.done.set()

    def push_to_all(self, message):
        for oc in self.output_connectors.values():
            for ost in oc.output_streams:
                ost.queue.put(message)

    def push_resource_usage(self):
        if self.perf_queue:
            if (datetime.datetime.now() - self.last_performance_update).seconds > 0.1:
                self.perf_queue.put((self.filter_name, datetime.datetime.now()-self.beginning, self.p.cpu_percent(), self.p.memory_info(), self.processed))
                self.last_performance_update = datetime.datetime.now()

    def main(self):
        """
        Generic run method which waits on a single stream and calls `process_data` on any new_data
        """
        logger.debug('Running "%s" run loop', self.filter_name)
        # self.finished_processing.clear()
        input_stream = getattr(self, self._input_connectors[0]).input_streams[0]
        desc = input_stream.descriptor

        stream_done = False
        stream_points = 0

        while not self.exit.is_set():# and not self.finished_processing.is_set():
            # Try to pull all messages in the queue. queue.empty() is not reliable, so we 
            # ask for forgiveness rather than permission.
            messages = []

            while not self.exit.is_set():
                try:
                    messages.append(input_stream.queue.get(False))
                except queue.Empty as e:
                    time.sleep(0.002)
                    break

            self.push_resource_usage()

            for message in messages:
                message_type = message['type']
                message_data = message['data']

                if message['type'] == 'event':
                    logger.debug('%s "%s" received event "%s"', self.__class__.__name__, self.filter_name, message_data)

                    # Propagate along the graph
                    self.push_to_all(message)

                    # Check to see if we're done
                    if message['event_type'] == 'done':
                        logger.debug(f"{self} received done message!")
                        stream_done = True
                        # if not self.finished_processing.is_set():
                        #     logger.warning("Filter {} being asked to finish before being done processing. ({} of {})".format(self.filter_name,stream_points,input_stream.num_points()))
                        # self.exit.set()
                        # self.finished_processing.set()
                        # break
                    elif message['event_type'] == 'refined':
                        self.refine(message_data)
                        continue

                    elif message['event_type'] == 'new_tuples':
                        self.process_new_tuples(input_stream.descriptor, message_data)
                        # break

                elif message['type'] == 'data':
                    if not hasattr(message_data, 'size'):
                        message_data = np.array([message_data])
                    logger.debug('%s "%s" received %d points.', self.__class__.__name__, self.filter_name, message_data.size)
                    logger.debug("Now has %d of %d points.", input_stream.points_taken.value, input_stream.num_points())
                    stream_points += len(message_data.flatten())
                    self.process_data(message_data.flatten())

                elif message['type'] == 'data_direct':
                    self.processed += message_data.nbytes
                    self.process_direct(message_data)

                # if stream_points == input_stream.num_points():
                #     self.finished_processing.set()
                #     break
            
            if stream_done:
                # outputs = self.output_connectors.values()

                # if outputs:
                #     output_status = [v.done() for v in outputs]
                #     print('x dones: %s' % str(output_status))

                #     if not np.all(output_status):
                #         print('------------------->>>>>>>>> NOT ALL DONE YET')

                self.done.set()
                break

            # If we have gotten all our data and process_data has returned, then we are done!
            # if desc.is_adaptive():
            #     if stream_done and np.all([len(desc.visited_tuples) == points_taken[s] for s in streams]):
            #         # self.finished_processing.set()
            #         break
            # else:
            #     if stream_done and np.all([v.done() for v in self.input_connectors.values()]):
            #         self.finished_processing.set()
            #         break      

        # When we've finished, either prematurely or as expected
        # print(self.filter_name, "leaving main loop")
        self.on_done()

    def process_data(self, data):
        """Process data coming through the filter pipeline"""
        pass

    def process_direct(self, data):
        """Process direct data, ignore things like the data descriptors."""
        pass

    def process_new_tuples(self, descriptor, message_data):
        axis_names, sweep_values = message_data
        # All the axis names for this connector
        ic_axis_names = [ax.name for ax in descriptor.axes]
        # The sweep values from sweep axes that are present (axes may have been dropped)
        sweep_values = [sv for an, sv in zip(axis_names, sweep_values) if an in ic_axis_names]
        vals = [a for a in descriptor.data_axis_values()]
        if sweep_values:
            vals  = [[v] for v in sweep_values] + vals

        # Create the outer product of axes
        nested_list    = list(itertools.product(*vals))
        flattened_list = [tuple((val for sublist in line for val in sublist)) for line in nested_list]
        descriptor.visited_tuples = descriptor.visited_tuples + flattened_list

        for oc in self.output_connectors.values():
            oc.push_event("new_tuples", message_data)

        return len(flattened_list)

    def refine(self, refine_data):
        """Try to deal with a refinement along the given axes."""
        axis_name, reset_axis, points = refine_data

        for ic in self.input_connectors.values():
            for desc in [ic.descriptor] + [s.descriptor for s in ic.input_streams]:
                for ax in desc.axes:
                    if ax.name == axis_name:
                        if reset_axis:
                            ax.points = points
                        else:
                            ax.points = np.append(ax.points, points)

        for oc in self.output_connectors.values():
            oc.push_event("refined", refine_data)
