# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['WriteToFile', 'DataBuffer']

import os
import sys

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    import threading as mp
    from threading import Thread as Process
    from threading import Event
    from queue import Queue
else:
    import multiprocess as mp
    from multiprocess import Process, Event
    from multiprocess import Queue

import itertools
import contextlib
import queue
import numpy as np
import os.path
import os, psutil
import time
import datetime
from shutil import copyfile
import cProfile

from .filter import Filter
from auspex.data_format import AuspexDataContainer
from auspex.parameter import Parameter, FilenameParameter, BoolParameter
from auspex.stream import InputConnector, OutputConnector
import auspex.config as config
from auspex.log import logger

class WriteToFile(Filter):
    """Writes data to file using the Auspex container type, which is a simple directory structure
    with subdirectories, binary datafiles, and json meta files that store the axis descriptors
    and other information."""

    sink        = InputConnector()
    filename    = FilenameParameter()
    groupname   = Parameter(default='main')
    datasetname = Parameter(default='data')

    def __init__(self, filename=None, groupname=None, datasetname=None, **kwargs):
        super(WriteToFile, self).__init__(**kwargs)
        if filename: 
            self.filename.value = filename
        if groupname:
            self.groupname.value = groupname
        if datasetname:
            self.datasetname.value = datasetname

        self.ret_queue = None # MP queue For returning data

    def final_init(self):
        assert self.filename.value, "Filename never supplied to writer."
        assert self.groupname.value, "Groupname never supplied to writer."
        assert self.datasetname.value, "Dataset name never supplied to writer."
        self.descriptor = self.sink.input_streams[0].descriptor

    def get_config(self):
        config = super().get_config()
        config['dtype']            = self.descriptor.dtype
        config['w_idx']            = 0
        config['points_taken']     = 0
        config['descriptor']       = self.descriptor
        config['filename']         = self.filename.value
        config['groupname']        = self.groupname.value
        config['datasetname']      = self.datasetname.value
        return config

    @classmethod
    def execute_on_run(cls, config):
        config['container']  = AuspexDataContainer(config['filename'])
        config['group']      = config['container'].new_group(config['groupname'])
        config['mmap']       = config['container'].new_dataset(config['groupname'], config['datasetname'], config['descriptor'])

    def get_data_while_running(self, return_queue):
        """Return data to the main thread or user as requested. Use a MP queue to transmit."""
        assert not self.done.is_set(), Exception("Experiment is over and filter done. Please use get_data")
        self.return_queue.put(np.array(self.mmap))

    def get_data(self):
        assert self.done.is_set(), Exception("Experiment is still running. Please use get_data_while_running")
        container = AuspexDataContainer(self.filename.value)
        return container.open_dataset(self.groupname.value, self.datasetname.value)[:2]

    @classmethod
    def process_data(cls, config, ocs, ics, data):
        # Write the data
        config['mmap'][config['w_idx']:config['w_idx']+data.size] = data
        config['w_idx'] += data.size
        config['points_taken'] = config['w_idx']

class DataBuffer(Filter):
    """Writes data to IO."""

    sink = InputConnector()

    def __init__(self, **kwargs):
        super(DataBuffer, self).__init__(**kwargs)
        if self.manager:
            self._final_buffer = self.manager.Queue()
            self._temp_buffer = self.manager.Queue()
        else:
            self._final_buffer = Queue()
            self._temp_buffer = Queue()
        self._get_buffer = Event()
        self.final_buffer  = None

    def final_init(self):
        self.descriptor   = self.sink.input_streams[0].descriptor

    def get_config(self):
        config = super().get_config()
        config['dtype']            = self.descriptor.dtype
        config['w_idx']            = 0
        config['points_taken']     = 0
        config['expected_points']  = self.descriptor.expected_num_points()
        config['final_buffer']     = self._final_buffer
        config['temp_buffer']      = self._temp_buffer
        config['get_buffer']       = self._get_buffer
        return config

    @classmethod
    def execute_on_run(cls, config):
        config['buff'] = np.empty(config['expected_points'], dtype=config['dtype'])

    @classmethod
    def execute_after_run(cls, config, output_queue):
        if mp.get_start_method() == "spawn":
            output_queue.put(("_final_buffer", config['buff']))
        config['final_buffer'].put(config['buff'])

    @classmethod
    def process_data(cls, config, ocs, ics, data):
        # Write the data
        config['buff'][config['w_idx']:config['w_idx']+data.size] = data
        config['w_idx'] += data.size
        config['points_taken'] = config['w_idx']

        if config['get_buffer'].is_set():
            config['temp_buffer'].put(config['buff'])
            config['get_buffer'].clear()

    def get_data(self):
        if self.done.is_set():
            if self.final_buffer is None:
                self.final_buffer = self._final_buffer.get()
            time.sleep(0.05)
            return np.reshape(self.final_buffer, self.descriptor.dims()), self.descriptor
        else:
            self._get_buffer.set()
            temp_buffer = self._temp_buffer.get()
            time.sleep(0.05)
            return np.reshape(temp_buffer, self.descriptor.dims()), self.descriptor
        
