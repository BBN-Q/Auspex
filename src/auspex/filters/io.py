# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['WriteToHDF5', 'DataBuffer', 'ProgressBar']

import asyncio, concurrent
import itertools
import h5py
import pickle
import zlib
import numpy as np
import os.path
import time
import re
import pandas as pd
from shutil import copyfile
from ruamel.yaml import YAML

from .filter import Filter
from auspex.parameter import Parameter, FilenameParameter, BoolParameter
from auspex.stream import InputConnector, OutputConnector
from auspex.log import logger
import auspex.config as config

from tqdm import tqdm, tqdm_notebook

class WriteToHDF5(Filter):
    """Writes data to file."""

    sink = InputConnector()
    filename = FilenameParameter()
    groupname = Parameter(default='main')
    add_date = BoolParameter(default = False)
    save_settings = BoolParameter(default = True)

    def __init__(self, filename=None, groupname=None, add_date=False, save_settings=True, compress=True, store_tuples=True, exp_log=True, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)
        self.compress = compress
        if filename:
            self.filename.value = filename
        if groupname:
            self.groupname.value = groupname
        self.points_taken = 0
        self.file = None
        self.group = None
        self.store_tuples = store_tuples
        self.create_group = True
        self.up_to_date = False
        self.sink.max_input_streams = 100
        self.add_date.value = add_date
        self.save_settings.value = save_settings
        self.exp_log = exp_log
        self.quince_parameters = [self.filename, self.groupname, self.add_date, self.save_settings]

    def final_init(self):
        if not self.filename.value:
            raise Exception("Filename never supplied to writer.")
        # If self.file is still None, then we need to create
        # the file object. Otherwise, we presume someone has
        # already set it up for us.
        if not self.file:
            self.file = self.new_file()

    def new_filename(self):
        filename = self.filename.value
        basename, ext = os.path.splitext(filename)
        if ext == "":
            logger.debug("Filename for writer {} does not have an extension -- using default '.h5'".format(self.name))
            ext = ".h5"

        dirname = os.path.dirname(os.path.abspath(filename))

        if self.add_date.value:
            date     = time.strftime("%y%m%d")
            dirname  = os.path.join(dirname, date)
            basename = os.path.join(dirname, os.path.basename(basename))

        # Set the file number to the maximum in the current folder + 1
        filenums = []
        if os.path.exists(dirname):
            for f in os.listdir(dirname):
                if ext in f:
                    filenums += [int(re.findall('-(\d{4})\.', f)[0])] if os.path.isfile(os.path.join(dirname, f)) else []

        i = max(filenums) + 1 if filenums else 0
        return "{}-{:04d}{}".format(basename,i,ext)

    def new_file(self):
        """ Open a new data file to write """
        # Close the current file, if any
        if self.file is not None:
            try:
                self.file.close()
            except Exception as e:
                logger.error("Encounter exception: {}".format(e))
                logger.error("Cannot close file '{}'. File may be damaged.".format(self.file.filename))
        # Get new file name
        self.filename.value = self.new_filename()
        head = os.path.dirname(self.filename.value)
        head = os.path.normpath(head)
        dirs = head.split(os.sep)
        # Check if path exists. If not, create new one(s).
        os.makedirs(head, exist_ok=True)
        logger.debug("Create new data file: %s." % self.filename.value)
        # Copy current settings to a folder with the file name
        if self.save_settings.value:
            # just move copies to a new directory
            self.save_yaml()
        if self.exp_log:
            self.write_to_log()
        return h5py.File(self.filename.value, 'w', libver='latest')

    def write_to_log(self):
        """ Record the experiment in a log file """
        logfile = os.path.join(config.LogDir, "experiment_log.tsv")
        if os.path.isfile(logfile):
            lf = pd.read_csv(logfile, sep="\t")
        else:
            logger.info("Experiment log file created.")
            lf = pd.DataFrame(columns = ["Filename", "Date", "Time"])
        lf = lf.append(pd.DataFrame([[self.filename.value, time.strftime("%y%m%d"), time.strftime("%H:%M:%S")]],columns=["Filename", "Date", "Time"]),ignore_index=True)
        lf.to_csv(logfile, sep = "\t", index = False)

    def save_yaml(self):
        """ Save a copy of current experiment settings """
        head = os.path.dirname(self.filename.value)
        fulldir = os.path.splitext(self.filename.value)[0]
        if not os.path.exists(fulldir):
            os.makedirs(fulldir)
            config.dump_meas_file(config.load_meas_file(config.meas_file), os.path.join(fulldir, os.path.split(config.meas_file)[1]), flatten = True)

    def save_yaml_h5(self):
        """ Save a copy of current experiment settings in the h5 metadata"""
        header = self.file.create_group("header")
        # load them dump to get the 'include' information
        header.attrs['settings'] = config.dump_meas_file(config.load_meas_file(config.meas_file), flatten = True)


    async def run(self):
        self.finished_processing = False
        streams    = self.sink.input_streams
        stream     = streams[0]

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to writer must have matching descriptors.")

        desc       = stream.descriptor
        axes       = desc.axes
        params     = desc.params
        axis_names = desc.axis_names(with_metadata=True)

        self.file.attrs['exp_src'] = desc._exp_src
        num_axes   = len(axes)

        if desc.is_adaptive() and not self.store_tuples:
            raise Exception("Cannot omit writing tuples with an adaptive sweep... please enabled store_tuples.")

        if self.store_tuples:
            # All of the combinations for the present values of the sweep parameters only
            tuples          = desc.expected_tuples(with_metadata=True, as_structured_array=True)
        expected_length = desc.expected_num_points()

        compression = 'gzip' if self.compress else None

        # If desired, create the group in which the dataset and axes will reside
        if self.create_group:
            self.group = self.file.create_group(self.groupname.value)
        else:
            self.group = self.file

        self.data_group = self.group.create_group("data")

        # If desired, push experimental metadata into the h5 file
        if self.save_settings.value and 'header' not in self.file.keys(): # only save header once for multiple writers
            self.save_yaml_h5()

        # Create datasets for each stream
        dset_for_streams = {}
        for stream in streams:
            dset = self.data_group.create_dataset(stream.descriptor.data_name, (expected_length,),
                                        dtype=stream.descriptor.dtype,
                                        chunks=True, maxshape=(None,),
                                        compression=compression)
            dset.attrs['is_data'] = True
            dset.attrs['store_tuples'] = self.store_tuples
            dset.attrs['name'] = stream.descriptor.data_name
            dset_for_streams[stream] = dset

        # Write params into attrs
        for k,v in params.items():
            if k not in axis_names:
                self.data_group.attrs[k] = v

        # Create a table for the DataStreamDescriptor
        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        self.descriptor = self.group.create_dataset("descriptor", (len(axes),), dtype=ref_dtype)
        for k,v in desc.metadata.items():
            self.descriptor.attrs[k] = v

        # Create axis data sets for storing the base axes as well as the
        # full set of tuples. For the former we add
        # references to the descriptor.
        tuple_dset_for_axis_name = {}
        for i, a in enumerate(axes):
            if a.unstructured:
                name = "+".join(a.name)
            else:
                name = a.name

            if a.unstructured:
                # Create another reference table to refer to the constituent axes
                unstruc_ref_dset = self.group.create_dataset(name, (len(a.name),), dtype=ref_dtype)
                unstruc_ref_dset.attrs['unstructured'] = True

                for j, (col_name, col_unit) in enumerate(zip(a.name, a.unit)):
                    # Create table to store the axis value independently for each column
                    unstruc_dset = self.group.create_dataset(col_name, (a.num_points(),), dtype=a.dtype)
                    unstruc_ref_dset[j] = unstruc_dset.ref
                    unstruc_dset[:] = a.points[:,j]
                    unstruc_dset.attrs['unit'] = col_unit
                    unstruc_dset.attrs['name'] = col_name

                    # This stores the values taking during the experiment sweeps
                    if self.store_tuples:
                        dset = self.data_group.create_dataset(col_name, (expected_length,), dtype=a.dtype,
                                                             chunks=True, compression=compression, maxshape=(None,) )
                        dset.attrs['unit'] = col_unit
                        dset.attrs['is_data'] = False
                        dset.attrs['name'] = col_name
                        tuple_dset_for_axis_name[col_name] = dset

                self.descriptor[i] = self.group[name].ref
            else:
                # This stores the axis values
                self.group.create_dataset(name, (a.num_points(),), dtype=a.dtype, maxshape=(None,) )
                self.group[name].attrs['unstructured'] = False
                self.group[name][:] = a.points
                self.group[name].attrs['unit'] = "None" if a.unit is None else a.unit
                self.group[name].attrs['name'] = a.name
                self.descriptor[i] = self.group[name].ref

                # This stores the values taking during the experiment sweeps
                if self.store_tuples:
                    dset = self.data_group.create_dataset(name, (expected_length,), dtype=a.dtype,
                                                          chunks=True, compression=compression, maxshape=(None,) )
                    dset.attrs['unit'] = "None" if a.unit is None else a.unit
                    dset.attrs['is_data'] = False
                    dset.attrs['name'] = name
                    tuple_dset_for_axis_name[name] = dset

            # Give the reader some warning about the usefulness of these axes
            self.group[name].attrs['was_refined'] = False

            if a.metadata is not None:
                # Create the axis table for the metadata
                dset = self.group.create_dataset(name + "_metadata", (a.metadata.size,), dtype=np.uint8, maxshape=(None,) )
                dset[:] = a.metadata
                dset = self.group.create_dataset(name + "_metadata_enum", (a.metadata_enum.size,), dtype='S128', maxshape=(None,) )
                dset[:] = np.asarray(a.metadata_enum, dtype='S128')

                # Associate the metadata with the data axis
                self.group[name].attrs['metadata'] = self.group[name + "_metadata"].ref
                self.group[name].attrs['metadata_enum'] = self.group[name + "_metadata_enum"].ref
                self.group[name].attrs['name'] = name + "_metadata"

                # Create the dataset that stores the individual tuple values
                if self.store_tuples:
                    dset = self.data_group.create_dataset(name + "_metadata" , (expected_length,),
                                                          dtype=np.uint8, maxshape=(None,) )
                    dset.attrs['name'] = name + "_metadata"
                    tuple_dset_for_axis_name[name + "_metadata"] = dset

        # Write all the tuples if this isn't adaptive
        if self.store_tuples:
            if not desc.is_adaptive():
                for i, a in enumerate(axis_names):
                    tuple_dset_for_axis_name[a][:] = tuples[a]

        # Write pointer
        w_idx = 0

        while True:
            # Wait for all of the acquisition to complete
            # Against at least some peoples rational expectations, asyncio.wait doesn't return Futures
            # in the order of the iterable it was passed, but perhaps just in order of completion. So,
            # we construct a dictionary in order that that can be mapped back where we need them:
            futures = {
                asyncio.ensure_future(stream.queue.get()): stream
                for stream in streams
            }

            responses, _ = await asyncio.wait(futures)

            # Construct the inverse lookup
            response_for_stream = {futures[res]: res for res in list(responses)}
            messages = [response_for_stream[stream].result() for stream in streams]

            # Ensure we aren't getting different types of messages at the same time.
            message_types = [m['type'] for m in messages]
            try:
                if len(set(message_types)) > 1:
                    raise ValueError("Writer received concurrent messages with different message types {}".format([m['type'] for m in messages]))
            except:
                import ipdb; ipdb.set_trace()

            # Infer the type from the first message
            message_type = messages[0]['type']

            # If we receive a message
            if message_type == 'event':
                logger.debug('%s "%s" received event of type "%s"', self.__class__.__name__, self.name, message_type)
                if messages[0]['event_type'] == 'done':
                    break
                elif messages[0]['event_type'] == 'refined':
                    refined_axis = messages[0]['data']

                    # Resize the data set
                    num_new_points = desc.num_new_points_through_axis(refined_axis)
                    for stream in streams:
                        dset_for_streams[stream].resize((len(dset_for_streams[streams[0]])+num_new_points,))

                    if self.store_tuples:
                        for an in axis_names:
                            tuple_dset_for_axis_name[an].resize((len(tuple_dset_for_axis_name[an])+num_new_points,))

                    # Generally speaking the descriptors are now insufficient to reconstruct
                    # the full set of tuples. The user should know this, so let's mark the
                    # descriptor axes accordingly.
                    self.group[name].attrs['was_refined'] = True


            elif message_type == 'data':
                message_data = [message['data'] for message in messages]
                message_comp = [message['compression'] for message in messages]
                message_data = [pickle.loads(zlib.decompress(dat)) if comp == 'zlib' else dat for comp, dat in zip(message_comp, message_data)]
                message_data = [dat if hasattr(dat, 'size') else np.array([dat]) for dat in message_data]  # Convert single values to arrays

                for ii in range(len(message_data)):
                    if not hasattr(message_data[ii], 'size'):
                        message_data[ii] = np.array([message_data[ii]])
                    message_data[ii] = message_data[ii].flatten()
                    if message_data[ii].size != message_data[0].size:
                        raise ValueError("Writer received data of unequal length.")

                logger.debug('%s "%s" received %d points', self.__class__.__name__, self.name, message_data[0].size)
                logger.debug("Now has %d of %d points.", stream.points_taken, stream.num_points())

                self.up_to_date = (w_idx == dset_for_streams[streams[0]].len())

                # Write the data
                for s, d in zip(streams, message_data):
                    dset_for_streams[s][w_idx:w_idx+d.size] = d

                # Write the coordinate tuples
                if self.store_tuples:
                    if desc.is_adaptive():
                        tuples = desc.tuples()
                        for axis_name in axis_names:
                            tuple_dset_for_axis_name[axis_name][w_idx:w_idx+d.size] = tuples[axis_name][w_idx:w_idx+d.size]

                self.file.flush()
                w_idx += message_data[0].size
                self.points_taken = w_idx

                logger.debug("HDF5: Write index at %d", w_idx)
                logger.debug("HDF5: %s has written %d points", stream.name, w_idx)

            # If we have gotten all our data and process_data has returned, then we are done!
            if np.all([v.done() for v in self.input_connectors.values()]):
                self.finished_processing = True

class DataBuffer(Filter):
    """Writes data to IO."""

    sink = InputConnector()

    def __init__(self, store_tuples=True, **kwargs):
        super(DataBuffer, self).__init__(**kwargs)
        self.quince_parameters = []
        self.sink.max_input_streams = 100
        self.store_tuples = store_tuples

    def final_init(self):
        self.buffers = {s: np.empty(s.descriptor.expected_num_points(), dtype=s.descriptor.dtype) for s in self.sink.input_streams}
        self.w_idxs  = {s: 0 for s in self.sink.input_streams}

    async def run(self):
        self.finished_processing = False
        streams = self.sink.input_streams

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to DataBuffer must have matching descriptors.")

        self.descriptor = streams[0].descriptor

        # Buffers for stream data
        stream_data = {s: np.zeros(0, dtype=self.sink.descriptor.dtype) for s in streams}

        # Store whether streams are done
        stream_done = {s: False for s in streams}

        while True:

            futures = {
                asyncio.ensure_future(stream.queue.get()): stream
                for stream in streams
            }

            # Deal with non-equal number of messages using timeout
            responses, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED, timeout=2.0)

            # Construct the inverse lookup, results in {stream: result}
            stream_results = {futures[res]: res.result() for res in list(responses)}

            # Cancel the futures
            for pend in list(pending):
                pend.cancel()

            # Add any new data to the
            for stream, message in stream_results.items():
                message_type = message['type']
                message_data = message['data']
                message_comp = message['compression']
                message_data = pickle.loads(zlib.decompress(message_data)) if message_comp == 'zlib' else message_data
                message_data = message_data if hasattr(message_data, 'size') else np.array([message_data])
                if message_type == 'event':
                    if message['event_type'] == 'done':
                        stream_done[stream] = True
                    elif message['event_type'] == 'refined':
                        # Single we don't have much structure here we simply
                        # create a new buffer and paste the old buffer into it
                        old_buffer = self.buffers[stream]
                        new_size   = stream.descriptor.num_points()
                        self.buffers[stream] = np.empty(stream.descriptor.num_points(), dtype=stream.descriptor.dtype)
                        self.buffers[stream][:old_buffer.size] = old_buffer

                elif message_type == 'data':
                    stream_data[stream] = message_data.flatten()

            if False not in stream_done.values():
                logger.debug('%s "%s" is done', self.__class__.__name__, self.name)
                break

            for stream in stream_results.keys():
                data = stream_data[stream]

                self.buffers[stream][self.w_idxs[stream]:self.w_idxs[stream]+data.size] = data
                self.w_idxs[stream] += data.size

            # If we have gotten all our data and process_data has returned, then we are done!
            if np.all([v.done() for v in self.input_connectors.values()]):
                self.finished_processing = True

    def get_data(self):
        streams = self.sink.input_streams
        desc = streams[0].descriptor

        # Set the dtype for the parameter columns
        if self.store_tuples:
            dtype = desc.axis_data_type(with_metadata=True)
        else:
            dtype = []

        # Extend the dtypes for each data column
        for stream in streams:
            dtype.append((stream.descriptor.data_name, stream.descriptor.dtype))
        data = np.empty(self.buffers[streams[0]].size, dtype=dtype)

        if self.store_tuples:
            tuples = desc.tuples(as_structured_array=True)
            for a in desc.axis_names(with_metadata=True):
                data[a] = tuples[a]
        for stream in streams:
            data[stream.descriptor.data_name] = self.buffers[stream]
        return data

    def get_descriptor(self):
        return self.sink.input_streams[0].descriptor

class ProgressBar(Filter):
    """ Display progress bar(s) on the terminal/notebook.

    num: number of progress bars to be display, \
    corresponding to the number of axes (counting from outer most)

        For running in Jupyter Notebook:
    Needs to open '_tqdm_notebook.py',\
    search for 'n = int(s[:npos])'\
    then replace it with 'n = float(s[:npos])'
    """
    sink = InputConnector()
    def __init__(self, num=0, notebook=False):
        super(ProgressBar,self).__init__()
        self.num    = num
        self.notebook = notebook
        self.bars   = []
        self.w_id   = 0

    async def run(self):
        self.stream = self.sink.input_streams[0]
        axes = self.stream.descriptor.axes
        num_axes = len(axes)
        totals = [self.stream.descriptor.num_points_through_axis(axis) for axis in range(num_axes)]
        chunk_sizes = [max(1,self.stream.descriptor.num_points_through_axis(axis+1)) for axis in range(num_axes)]
        self.num = min(self.num, num_axes)

        self.bars   = []
        for i in range(self.num):
            if self.notebook:
                self.bars.append(tqdm_notebook(total=totals[i]/chunk_sizes[i]))
            else:
                self.bars.append(tqdm(total=totals[i]/chunk_sizes[i]))
        self.w_id   = 0
        while True:
            if self.stream.done() and self.w_id==self.stream.num_points():
                break

            new_data = np.array(await self.stream.queue.get()).flatten()
            while self.stream.queue.qsize() > 0:
                new_data = np.append(new_data, np.array(self.stream.queue.get_nowait()).flatten())
            self.w_id += new_data.size
            num_data = self.stream.points_taken
            for i in range(self.num):
                if num_data == 0:
                    if self.notebook:
                        self.bars[i].sp(close=True)
                        # Reset the progress bar with a new one
                        self.bars[i] = tqdm_notebook(total=totals[i]/chunk_sizes[i])
                    else:
                        # Reset the progress bar with a new one
                        self.bars[i].close()
                        self.bars[i] = tqdm(total=totals[i]/chunk_sizes[i])
                pos = int(10*num_data / chunk_sizes[i])/10.0 # One decimal is good enough
                if pos > self.bars[i].n:
                    self.bars[i].update(pos - self.bars[i].n)
                num_data = num_data % chunk_sizes[i]
