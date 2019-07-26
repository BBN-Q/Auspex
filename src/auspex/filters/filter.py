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
    from multiprocessing import Value, Array

from setproctitle import setproctitle
import cProfile
import itertools
import time, datetime
import queue
import copy
import ctypes
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
        self.qubit_name = ""

        # Event for killing the filter properly
        self.exit = Event()
        self.done = Event()

        # Keep track of data throughput
        self.processed = 0

        # For objectively measuring doneness
        self.finished_processing = Event()
        self.finished_processing.clear()

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

    def configure_with_proxy(self, proxy_obj):
        """For use with bbndb, sets this filter's properties using a FilterProxy object
        taken from the filter database."""

        if proxy_obj.label:
            self.filter_name = proxy_obj.label

        for name, param in self.parameters.items():
            if hasattr(proxy_obj, name):
                if getattr(proxy_obj, name) is not None:
                    param.value = getattr(proxy_obj, name)
            else:
                raise ValueError(f"{proxy_obj} was expected to have parameter {name}")

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

    def on_done(self):
        """To be run when the done signal is received, in case additional steps are
        needed (such as flushing a plot or data)."""
        pass

    def shutdown(self):
        self.exit.set()

    def _parent_process_running(self):
        try:
            os.kill(os.getppid(), 0)
        except OSError:
            return False
        else:
            return True

    def run(self):
        self.p = psutil.Process(os.getpid())
        logger.debug(f"{self} launched with pid {os.getpid()}. ppid {os.getppid()}")
        if auspex.config.profile:
            if not self.filter_name:
                name = "Unlabeled"
            else:
                name = self.filter_name
            cProfile.runctx('self.main()', globals(), locals(), 'prof-%s-%s.prof' % (self.__class__.__name__, name))
        else:
            self.execute_on_run()
            self.main()
        self.done.set()

    def execute_on_run(self):
        pass

    def push_to_all(self, message):
        for oc in self.output_connectors.values():
            for ost in oc.output_streams:
                ost.queue.put(message)
                if message['type'] == 'event' and message["event_type"] == "done":
                    logger.debug(f"Closing out queue {ost.queue}")
                    ost.queue.close()

    def push_resource_usage(self):
        if self.perf_queue and (datetime.datetime.now() - self.last_performance_update).seconds > 1.0:
            perf_info = (str(self), datetime.datetime.now()-self.beginning, self.p.cpu_percent(), self.p.memory_info(), self.processed)
            self.perf_queue.put(perf_info)
            self.last_performance_update = datetime.datetime.now()

    def main(self):
        """
        Generic run method which waits on a single stream and calls `process_data` on any new_data
        """
        # try:

        logger.debug('Running "%s" run loop', self.filter_name)
        setproctitle(f"python auspex filter: {self}")
        input_stream = getattr(self, self._input_connectors[0]).input_streams[0]
        desc = input_stream.descriptor

        stream_done = False
        stream_points = 0

        while not self.exit.is_set():# and not self.finished_processing.is_set():
            # Try to pull all messages in the queue. queue.empty() is not reliable, so we
            # ask for forgiveness rather than permission.
            messages = []

            # Check to see if the parent process still exists:
            if not self._parent_process_running():
                logger.warning(f"{self} with pid {os.getpid()} could not find parent with pid {os.getppid()}. Assuming something has gone wrong. Exiting.")
                break

            while not self.exit.is_set():
                try:
                    messages.append(input_stream.queue.get(False))
                except queue.Empty as e:
                    time.sleep(0.002)
                    break

            self.push_resource_usage()
            for message in messages:
                message_type = message['type']
                if message['type'] == 'event':
                    logger.debug('%s "%s" received event with type "%s"', self.__class__.__name__, message_type)

                    # Check to see if we're done
                    if message['event_type'] == 'done':
                        logger.debug(f"{self} received done message!")
                        stream_done = True
                    else:
                        # Propagate along the graph
                        self.push_to_all(message)

                elif message['type'] == 'data':
                    # if not hasattr(message_data, 'size'):
                    #     message_data = np.array([message_data])
                    message_data = input_stream.pop()
                    if message_data is not None:
                        logger.debug('%s "%s" received %d points.', self.__class__.__name__, self.filter_name, message_data.size)
                        logger.debug("Now has %d of %d points.", input_stream.points_taken.value, input_stream.num_points())
                        stream_points += len(message_data)
                        self.process_data(message_data)
                        self.processed += message_data.nbytes

            if stream_done:
                self.push_to_all({"type": "event", "event_type": "done", "data": None})
                self.done.set()
                break

        # When we've finished, either prematurely or as expected
        self.on_done()

        # except Exception as e:
        #     logger.warning(f"Filter {self} raised exception {e}. Bailing.")
