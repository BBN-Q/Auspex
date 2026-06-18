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
import traceback
import psutil

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    from threading import Event
    from queue import Queue
else:
    from multiprocess import Event
    from multiprocess import Queue
    from multiprocess import Value, Array
    import multiprocess as mp

from setproctitle import setproctitle
import cProfile
import itertools
import time, datetime
import queue
import copy
import ctypes
import numpy as np

import auspex.config
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

class Filter(object, metaclass=MetaFilter):
    """Any node on the graph that takes input streams with optional output streams"""

    def __init__(self, name=None, manager=None, **kwargs):
        super(Filter, self).__init__()
        self.filter_name = name
        self.input_connectors = {}
        self.output_connectors = {}
        self.parameters = {}
        self.qubit_name = ""
        self.manager = manager

        # Event for keeping track of individual filters being done
        self.done = Event()

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
                raise ValueError(f"{proxy_obj} was expected to have parameter {name}")

    def __repr__(self):
        return "<{}(name={})>".format(self.__class__.__name__, self.filter_name)

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

    def final_init(self):
        """Any final configuration that gets run in the main thread before a process is
        spawned on the class run method"""
        pass

    def get_config(self):
        """Return a config dictionary for this object"""
        config = {}
        config['name']      = str(self)
        config['processed'] = 0
        return config

    @classmethod
    def on_done(cls):
        """To be run when the done signal is received, in case additional steps are
        needed (such as flushing a plot or data)."""
        pass

    @classmethod
    def _parent_process_running(cls):
        try:
            os.kill(os.getppid(), 0)
        except OSError:
            return False
        else:
            return True

    @classmethod
    def spawn_fix(cls, ocs, ics):
        for oc in ocs.values():
            for stream in oc.output_streams:
                stream.spawn_fix()
        for ic in ics.values():
            for stream in ic.input_streams:
                stream.spawn_fix()

    @classmethod
    def run(cls, config, exit, done, panic, ocs, ics, profile, perf_queue, output_queue):
        """This is the entry function when filters are launched as Processes.
        config:       configuration dictionary for any necessary parameters at run-time
        exit:         event for killing the filter
        done:         event the filter uses to declare it is done
        panic:        event the filter 
        ocs:          output connectors
        ics:          input connectors
        profile:      push performance information (Boolean)
        perf_queue:   where to push the performance information
        output_queue: queue for returning key, value pairs to the main process
        """
        try:
            p = psutil.Process(os.getpid())
            name = config["name"]
            logger.debug(f"{name} launched with pid {os.getpid()}. ppid {os.getppid()}")
            if profile:
                cProfile.runctx('cls.main(config, exit, done, panic, ocs, ics, perf_queue)', globals(), locals(), 'prof-%s.prof' % name)
            else:
                cls.spawn_fix(ocs, ics)
                cls.execute_on_run(config)
                cls.main(config, exit, done, panic, ocs, ics, perf_queue)
                cls.execute_after_run(config, output_queue)
            done.set()
        except Exception as e:
            just_the_string = traceback.format_exc()
            logger.warning(f"Filter {config['name']} raised exception {e} (traceback follows). Bailing.")
            logger.warning(just_the_string)

            panic.set()
            done.set()

    @classmethod
    def execute_on_run(cls, config):
        pass

    @classmethod
    def execute_after_run(cls, config, output_queue):
        pass

    @classmethod
    def push_to_all(cls, ocs, message):
        for oc in ocs.values():
            for ost in oc.output_streams:
                ost.queue.put(message)
                if message['type'] == 'event' and message["event_type"] == "done":
                    logger.debug(f"Closing out queue {ost.queue}")
                    ost.queue.close()

    @classmethod
    def push_resource_usage(cls, name, perf_queue, beginning, last_performance_update, processed):
        if perf_queue and (datetime.datetime.now() - last_performance_update).seconds > 1.0:
            perf_info = (name, datetime.datetime.now()-beginning, p.cpu_percent(), p.memory_info(), processed)
            perf_queue.put(perf_info)

    @classmethod
    def process_message(cls, config, msg):
        """To be overridden for interesting default behavior"""
        pass

    @classmethod
    def checkin(cls, config):
        """For any filter-specific loop needs"""
        pass

    @classmethod
    def process_data(cls, config, ocs, ics, message_data):
        """Specific per-filter data processing"""
        pass

    @classmethod
    def main(cls, config, exit, done, panic, ocs, ics, perf_queue):
        """
        Generic run method which waits on a single stream and calls `process_data` on any new_data
        """
        name = config["name"]
        setproctitle(f"auspex filt: {name}")

        # Assume we only have a single input stream in this base implementation
        input_stream = list(ics.values())[0].input_streams[0]
        desc = input_stream.descriptor

        stream_done   = False
        stream_points = 0

        # For performance profiling
        beginning = datetime.datetime.now()
        last_performance_update = datetime.datetime.now()

        while not exit.is_set() and not panic.is_set():
            messages = []

            # For any filter-specific loop needs
            cls.checkin(config)

            # Check to see if the parent process still exists:
            if not cls._parent_process_running():
                logger.warning(f"{name} with pid {os.getpid()} could not find parent with pid {os.getppid()}. Assuming something has gone wrong. Exiting.")
                break

            while not exit.is_set():
                try:
                    messages.append(input_stream.queue.get(False))
                except queue.Empty as e:
                    time.sleep(0.002)
                    break
            
            cls.push_resource_usage(name, perf_queue, beginning, last_performance_update, config['processed'])
            last_performance_update = datetime.datetime.now()

            for message in messages:
                message_type = message['type']
                if message['type'] == 'event':
                    logger.debug('%s received event with type "%s"', name, message['event_type'])

                    # Check to see if we're done
                    if message['event_type'] == 'done':
                        logger.debug(f"{name} received done message!")
                        stream_done = True
                    else:
                        # Propagate along the graph
                        cls.push_to_all(ocs, message)
                        cls.process_message(config, message)

                elif message['type'] == 'data':
                    message_data = input_stream.pop()
                    if message_data is not None:
                        logger.debug('%s received %d points.', name, message_data.size)
                        logger.debug("Now has %d of %d points.", input_stream.points_taken.value, input_stream.num_points())
                        stream_points += len(message_data)
                        cls.process_data(config, ocs, ics, message_data)
                        config['processed'] += message_data.nbytes

            if stream_done:
                cls.push_to_all(ocs, {"type": "event", "event_type": "done", "data": None})
                done.set()
                break

        # When we've finished, either prematurely or as expected
        cls.on_done()

