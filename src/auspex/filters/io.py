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
    from queue import Queue
else:
    import multiprocessing as mp
    from multiprocessing import Process
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
import pandas as pd
from shutil import copyfile
import cProfile

from .filter import Filter
from auspex.parameter import Parameter, FilenameParameter, BoolParameter
from auspex.stream import InputConnector, OutputConnector
from auspex.log import logger
import auspex.config as config

from tqdm import tqdm, tqdm_notebook

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
        assert not self.done.is_set(), Exception("Experiment is over and filter done. Please use load_data")
        self.return_queue.put(np.array(self.mmap))

    def load_data(self):
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

    def __init__(self, store_tuples=True, **kwargs):
        super(DataBuffer, self).__init__(**kwargs)
        self.quince_parameters = []
        self.sink.max_input_streams = 100
        self.store_tuples = store_tuples
        self._final_buffers = Queue()
        self.final_buffers = None

        # self.out_queue = Queue() # This seems to work, somewhat surprisingly

    def final_init(self):
        self.w_idxs  = {s: 0 for s in self.sink.input_streams}
        self.descriptor = self.sink.input_streams[0].descriptor

    def main(self):
        self.done.clear()
        streams = self.sink.input_streams

        buffers = {s: np.empty(s.descriptor.expected_num_points(), dtype=s.descriptor.dtype) for s in self.sink.input_streams}

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to DataBuffer must have matching descriptors.")

        # Store whether streams are done
        stream_done = {s: False for s in streams}

        while not self.exit.is_set():
            # Raw messages for stream data
            msgs_by_stream = {s: [] for s in streams}

            # Buffers for stream data
            stream_data = {s: [] for s in streams}

            for stream in streams[::-1]:
                while not self.exit.is_set():
                    try:
                        msgs_by_stream[stream].append(stream.queue.get(False))
                    except queue.Empty as e:
                        time.sleep(0.002)
                        break

            # Add any new data to the buffers
            for stream, messages in msgs_by_stream.items():
                for message in messages:
                    message_type = message['type']
                    message_data = message['data']
                    message_data = message_data if hasattr(message_data, 'size') else np.array([message_data])
                    if message_type == 'event':
                        if message['event_type'] == 'done':
                            stream_done[stream] = True
                        elif message['event_type'] == 'refined':
                            # Single we don't have much structure here we simply
                            # create a new buffer and paste the old buffer into it
                            old_buffer = buffers[stream]
                            new_size   = stream.descriptor.num_points()
                            buffers[stream] = np.empty(stream.descriptor.num_points(), dtype=stream.descriptor.dtype)
                            buffers[stream][:old_buffer.size] = old_buffer

                    elif message_type == 'data':
                        new_dat = message_data.flatten()
                        stream_data[stream].append(new_dat)
                        self.processed += new_dat.nbytes

            self.push_resource_usage()

            for stream in streams:
                datas = stream_data[stream]
                for data in datas:
                    # logger.info(f"stream {s} gets data {data.size}. Buffer size {len(buffers[stream])} Range {self.w_idxs[stream]} to {self.w_idxs[stream]+data.size} ")
                    buffers[stream][self.w_idxs[stream]:self.w_idxs[stream]+data.size] = data
                    self.w_idxs[stream] += data.size

            # If we have gotten all our data and process_data has returned, then we are done!
            if np.all([stream_done[stream] for stream in streams]):
                break

        for s in streams:
            self._final_buffers.put(buffers[s])

        self.done.set()

    def get_rdata(self):
        if self.out_queue:
            data = self.get_data()
            self.out_queue.put(data)

    def get_data(self):
        streams = self.sink.input_streams
        desc = streams[0].descriptor

        # Get the data from the queue if necessary
        if not self.final_buffers:
            self.final_buffers = {s: self._final_buffers.get() for s in streams}

        # Set the dtype for the parameter columns
        if self.store_tuples:
            dtype = desc.axis_data_type(with_metadata=True)
        else:
            dtype = []

        # Extend the dtypes for each data column
        for stream in streams:
            dtype.append((stream.descriptor.data_name, stream.descriptor.dtype))
        data = np.empty(self.final_buffers[streams[0]].size, dtype=dtype)

        if self.store_tuples:
            tuples = desc.tuples(as_structured_array=True)
            for a in desc.axis_names(with_metadata=True):
                data[a] = tuples[a]
        for stream in streams:
            data[stream.descriptor.data_name] = self.final_buffers[stream]
        return data

    def get_descriptor(self):
        return self.sink.input_streams[0].descriptor

# class ProgressBar(Filter):
#     """ Display progress bar(s) on the terminal/notebook.

#     num: number of progress bars to be display, \
#     corresponding to the number of axes (counting from outer most)

#         For running in Jupyter Notebook:
#     Needs to open '_tqdm_notebook.py',\
#     search for 'n = int(s[:npos])'\
#     then replace it with 'n = float(s[:npos])'
#     """
#     sink = InputConnector()
#     def __init__(self, num=0, notebook=False):
#         super(ProgressBar,self).__init__()
#         self.num    = num
#         self.notebook = notebook
#         self.bars   = []
#         self.w_id   = 0

#     def run(self):
#         if config.profile:
#             cProfile.runctx('self.main()', globals(), locals(), 'prof-%s.prof' % self.filter_name)
#         else:
#             self.main()

#     def main(self):
#         self.stream = self.sink.input_streams[0]
#         axes = self.stream.descriptor.axes
#         num_axes = len(axes)
#         totals = [self.stream.descriptor.num_points_through_axis(axis) for axis in range(num_axes)]
#         chunk_sizes = [max(1,self.stream.descriptor.num_points_through_axis(axis+1)) for axis in range(num_axes)]
#         self.num = min(self.num, num_axes)

#         self.bars   = []
#         for i in range(self.num):
#             if self.notebook:
#                 self.bars.append(tqdm_notebook(total=totals[i]/chunk_sizes[i]))
#             else:
#                 self.bars.append(tqdm(total=totals[i]/chunk_sizes[i]))
#         self.w_id   = 0
#         while not self.exit.is_set():
#             if self.stream.done() and self.w_id==self.stream.num_points():
#                 break

#             new_data = np.array(self.stream.queue.get()).flatten()
#             while self.stream.queue.qsize() > 0:
#                 new_data = np.append(new_data, np.array(self.stream.queue.get_nowait()).flatten())
#             self.w_id += new_data.size
#             num_data = self.stream.points_taken.value
#             for i in range(self.num):
#                 if num_data == 0:
#                     if self.notebook:
#                         self.bars[i].sp(close=True)
#                         # Reset the progress bar with a new one
#                         self.bars[i] = tqdm_notebook(total=totals[i]/chunk_sizes[i])
#                     else:
#                         # Reset the progress bar with a new one
#                         self.bars[i].close()
#                         self.bars[i] = tqdm(total=totals[i]/chunk_sizes[i])
#                 pos = int(10*num_data / chunk_sizes[i])/10.0 # One decimal is good enough
#                 if pos > self.bars[i].n:
#                     self.bars[i].update(pos - self.bars[i].n)
#                 num_data = num_data % chunk_sizes[i]
