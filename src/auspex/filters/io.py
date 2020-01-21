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
    import multiprocessing as mp
    from multiprocessing import Process, Event
    from multiprocessing import Queue

from auspex.data_format import AuspexDataContainer

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
from auspex.parameter import Parameter, FilenameParameter, BoolParameter
from auspex.stream import InputConnector, OutputConnector
from auspex.log import logger
import auspex.config as config

class WriteToFile(Filter):
    """Writes data to file using the Auspex container type, which is a simple directory structure
    with subdirectories, binary datafiles, and json meta files that store the axis descriptors
    and other information."""

    sink        = InputConnector()
    filename    = FilenameParameter()
    groupname   = Parameter(default='main')

    def __init__(self, filename=None, groupname=None, datasetname='data', **kwargs):
        super(WriteToFile, self).__init__(**kwargs)
        if filename: 
            self.filename.value = filename
        if groupname:
            self.groupname.value = groupname
        if datasetname:
            self.datasetname = datasetname

        self.ret_queue = None # MP queue For returning data

    def final_init(self):
        assert self.filename.value, "Filename never supplied to writer."
        assert self.groupname.value, "Groupname never supplied to writer."
        assert self.datasetname, "Dataset name never supplied to writer."

        self.descriptor = self.sink.input_streams[0].descriptor
        self.container  = AuspexDataContainer(self.filename.value)
        self.group      = self.container.new_group(self.groupname.value)
        self.mmap       = self.container.new_dataset(self.groupname.value, self.datasetname, self.descriptor)

        self.w_idx = 0
        self.points_taken = 0

    def get_data_while_running(self, return_queue):
        """Return data to the main thread or user as requested. Use a MP queue to transmit."""
        assert not self.done.is_set(), Exception("Experiment is over and filter done. Please use get_data")
        self.return_queue.put(np.array(self.mmap))

    def get_data(self):
        assert self.done.is_set(), Exception("Experiment is still running. Please use get_data_while_running")
        container = AuspexDataContainer(self.filename.value)
        return container.open_dataset(self.groupname.value, self.datasetname)

    def process_data(self, data):
        # Write the data
        self.mmap[self.w_idx:self.w_idx+data.size] = data
        self.w_idx += data.size
        self.points_taken = self.w_idx

class DataBuffer(Filter):
    """Writes data to IO."""

    sink = InputConnector()

    def __init__(self, **kwargs):
        super(DataBuffer, self).__init__(**kwargs)
        self._final_buffer = Queue()
        self._temp_buffer = Queue()
        self._get_buffer = Event()
        self.final_buffer  = None

    def final_init(self):
        self.w_idx        = 0
        self.points_taken = 0
        self.descriptor   = self.sink.input_streams[0].descriptor
        self.buff         = np.empty(self.descriptor.expected_num_points(), dtype=self.descriptor.dtype)

    def checkin(self):
        if self._get_buffer.is_set():
            self._temp_buffer.put(self.buff)
        self._get_buffer.clear()

    def process_data(self, data):
        # Write the data
        self.buff[self.w_idx:self.w_idx+data.size] = data
        self.w_idx += data.size
        self.points_taken = self.w_idx

    def main(self):
        super(DataBuffer, self).main()
        self._final_buffer.put(self.buff)

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
        
