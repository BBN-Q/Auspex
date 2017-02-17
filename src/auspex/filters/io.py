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

from .filter import Filter
from auspex.parameter import Parameter, FilenameParameter
from auspex.stream import InputConnector, OutputConnector
from auspex.log import logger

from tqdm import tqdm, tqdm_notebook

class WriteToHDF5(Filter):
    """Writes data to file."""

    sink = InputConnector()
    filename = FilenameParameter()
    groupname = Parameter(default='main')

    def __init__(self, filename=None, groupname=None, add_date=False, compress=True, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)
        self.compress = compress
        if filename:
            self.filename.value = filename
        if groupname:
            self.groupname.value = groupname
        self.points_taken = 0
        self.file = None
        self.group = None
        self.create_group = True
        self.up_to_date = False
        self.sink.max_input_streams = 100
        self.add_date = add_date

        self.quince_parameters = [self.filename, self.groupname]

    def final_init(self):
        if not self.filename.value:
            raise Exception("Filename never supplied to writer.")
        # If self.file is still None, then we need to create
        # the file object. Otherwise, we presume someone has
        # already set it up for us.
        if not self.file:
            self.file = self.new_file()

    def new_filename(self):
        # Increment the filename until we find one we want.
        i = 0
        filename = self.filename.value
        ext = filename.find('.h5')
        if ext > -1:
            filename = filename[:ext]
        if self.add_date:
            date = time.strftime("%y%m%d")
            dirname = os.path.dirname(filename)
            basename = os.path.basename(filename)
            fulldir = os.path.join(dirname, date)
            if not os.path.exists(fulldir):
                os.mkdir(fulldir)
            filename = os.path.join(fulldir, basename)
        while os.path.exists("{}-{:04d}.h5".format(filename,i)):
            i += 1
        return "{}-{:04d}.h5".format(filename,i)

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
        fulldir = ''
        for d in dirs:
            fulldir = os.path.join(fulldir, d)
            if not os.path.exists(fulldir):
                logger.debug("Create new directory: {}.".format(fulldir))
                os.mkdir(fulldir)
        logger.debug("Create new data file: %s." % self.filename.value)
        return h5py.File(self.filename.value, 'w', libver='latest')

    async def run(self):
        streams    = self.sink.input_streams
        stream     = streams[0]

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to writer must have matching descriptors.")

        desc       = stream.descriptor
        axes       = desc.axes
        params     = desc.params
        axis_names = desc.axis_names(with_metadata=True)

        self.file.attrs['exp_src'] = desc.exp_src
        num_axes   = len(axes)

        # All of the combinations for the present values of the sweep parameters only
        tuples          = desc.expected_tuples(with_metadata=True, as_structured_array=True)
        expected_length = desc.expected_num_points()

        # If desired, create the group in which the dataset and axes will reside
        if self.create_group:
            self.group = self.file.create_group(self.groupname.value)
        else:
            self.group = self.file

        dtype = desc.axis_data_type(with_metadata=True)

        # Extend the dtypes for each data column
        for stream in streams:
            dtype.append((stream.descriptor.data_name, desc.dtype))

        logger.debug("Data type for HDF5: %s", dtype)
        if self.compress:
            self.data = self.group.create_dataset('data', (expected_length,), dtype=dtype,
                                        chunks=True, maxshape=(None,),
                                        compression='gzip')
            # TODO: use single writer multiple reader when HDF5 libraries are updated in h5py
            # self.file.swmr_mode = True
        else:
            self.data = self.group.create_dataset('data', (expected_length,), dtype=dtype,
                                        chunks=True, maxshape=(None,))
            # TODO: use single writer multiple reader when HDF5 libraries are updated in h5py
            # self.file.swmr_mode = True

        # Write params into attrs
        for k,v in params.items():
            if k not in axis_names:
                self.data.attrs[k] = v

        # Create a table for the DataStreamDescriptor
        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        self.descriptor = self.group.create_dataset("descriptor", (len(axes),), dtype=ref_dtype)
        for k,v in desc.metadata.items():
            self.descriptor.attrs[k] = v

        # Associated axis dimensions with the data and add
        # references to the descriptor.
        for i, a in enumerate(axes):
            if a.unstructured:
                name = "+".join(a.name)
            else:
                name = a.name

            dtype = a.data_type(with_metadata=True)
            self.group.create_dataset(name, (a.num_points(),), dtype=dtype, maxshape=(None,) )
            self.group[name].attrs['was_refined'] = False

            if a.unstructured:
                for j, (col_name, col_unit) in enumerate(zip(a.name, a.unit)):
                    self.group[name][col_name,:] = a.points[:,j]
                    self.group[name].attrs['unit_'+col_name] = col_unit
            else:
                self.group[name][:] = a.points
                self.group[name].attrs['unit'] = "None" if a.unit is None else a.unit

            if a.metadata:
                self.group[name + "_metadata"] = np.string_(a.metadata)
                self.group[name][name + "_metadata",:] = np.string_(a.metadata)

            self.data.dims.create_scale(self.group[name], name)
            self.data.dims[0].attach_scale(self.group[name])
            self.descriptor[i] = self.group[name].ref

        # Write all the tuples if this isn't adaptive
        if not desc.is_adaptive():
            for i, a in enumerate(axis_names):
                self.data[a,:] = tuples[a]

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
                logger.debug('%s "%s" received event "%s"', self.__class__.__name__, self.name, message_data)
                if messages[0]['event_type'] == 'done':
                    break
                elif messages[0]['event_type'] == 'refined':
                    refined_axis = messages[0]['data']
                    
                    # Resize the data set
                    num_new_points = desc.num_new_points_through_axis(refined_axis)
                    self.data.resize((len(self.data)+num_new_points,))

                    # Generally speaking the descriptors are now insufficient to reconstruct
                    # the full set of tuples. The user should know this, so let's mark the
                    # descriptor axes accordingly.
                    self.group[name].attrs['was_refined'] = True

            
            elif message_type == 'data':
                message_data = [message['data'] for message in messages]
                message_comp = [message['compression'] for message in messages]
                message_data = [pickle.loads(zlib.decompress(dat)) if comp == 'zlib' else dat for comp, dat in zip(message_comp, message_data)]
                message_data = [dat if hasattr(dat, 'size') else np.array([dat]) for dat in message_data]  # Convert single values to arrays

                for ii in range(1, len(message_data)):
                    if not hasattr(message_data[ii], 'size'):
                        message_data[ii] = np.array([message_data[ii]])
                    message_data[ii] = message_data[ii].flatten()
                    if message_data[ii].size != message_data[0].size:
                        raise ValueError("Writer received data of unequal length.")

                logger.debug('%s "%s" received %d points', self.__class__.__name__, self.name, message_data[0].size)
                logger.debug("Now has %d of %d points.", stream.points_taken, stream.num_points())

                try:
                    self.up_to_date = (w_idx == self.data.len())
                except:
                    import ipdb; ipdb.set_trace()

                # Write the data
                for s, d in zip(streams, message_data):
                    self.data[s.descriptor.data_name, w_idx:w_idx+d.size] = d
                
                # Write the coordinate tuples
                if desc.is_adaptive():
                    tuples = desc.tuples()
                    for axis_name in axis_names:
                        self.data[axis_name, w_idx:w_idx+d.size] = tuples[axis_name][w_idx:w_idx+d.size]

                self.file.flush()
                w_idx += message_data[0].size
                self.points_taken = w_idx

                logger.debug("HDF5: Write index at %d", w_idx)
                logger.debug("HDF5: %s has written %d points", stream.name, w_idx)

class DataBuffer(Filter):
    """Writes data to file."""

    sink = InputConnector()

    def __init__(self, **kwargs):
        super(DataBuffer, self).__init__(**kwargs)
        self.quince_parameters = []
        self.sink.max_input_streams = 100

    def final_init(self):
        self.buffers = {s: np.empty(s.descriptor.expected_num_points(), dtype=s.descriptor.dtype) for s in self.sink.input_streams}
        self.w_idxs  = {s: 0 for s in self.sink.input_streams}

    async def run(self):
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

    def get_data(self):
        streams = self.sink.input_streams
        desc = streams[0].descriptor
        # Set the dtype for the parameter columns
        dtype = desc.axis_data_type(with_metadata=True)

        # Extend the dtypes for each data column
        for stream in streams:
            dtype.append((stream.descriptor.data_name, stream.descriptor.dtype))
        data = np.empty(self.buffers[streams[0]].size, dtype=dtype)

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
