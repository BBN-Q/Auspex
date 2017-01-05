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
import numpy as np
import os.path

from auspex.parameter import Parameter, FilenameParameter
from auspex.stream import DataStreamDescriptor
from auspex.log import logger
from auspex.filters.filter import Filter, InputConnector, OutputConnector
from tqdm import tqdm, tqdm_notebook

class WriteToHDF5(Filter):
    """Writes data to file."""

    sink = InputConnector()
    filename = FilenameParameter()
    groupname = Parameter(default='main')

    def __init__(self, filename=None, groupname=None, compress=True, **kwargs):
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
        stream     = self.sink.input_streams[0]
        desc       = stream.descriptor
        axes       = stream.descriptor.axes
        params     = stream.descriptor.params
        axis_names = desc.axis_names(with_metadata=True)

        self.file.attrs['exp_src'] = stream.descriptor.exp_src
        num_axes   = len(axes)

        # All of the combinations for the present values of the sweep parameters only
        tuples     = stream.descriptor.tuples(with_metadata=True, as_structured_array=True)

        # If desired, create the group in which the dataset and axes will reside
        if self.create_group:
            self.group = self.file.create_group(self.groupname.value)
        else:
            self.group = self.file

        dtype = desc.axis_data_type(with_metadata=True)
        dtype.append((desc.data_name, desc.dtype))
        logger.debug("Data type for HDF5: %s", dtype)
        if self.compress:
            self.data = self.group.create_dataset('data', (len(tuples),), dtype=dtype,
                                        chunks=True, maxshape=(None,),
                                        compression='gzip')
            # TODO: update when HDF version changes...
            # self.file.swmr_mode = True
        else:
            self.data = self.group.create_dataset('data', (len(tuples),), dtype=dtype,
                                        chunks=True, maxshape=(None,))
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

        # Write the initial coordinate tuples
        for i, a in enumerate(axis_names):
            self.data[a,:] = tuples[a]

        # Write pointer
        w_idx = 0

        while True:

            message = await stream.queue.get()
            message_type = message['type']
            message_data = message['data']
            message_comp = message['compression']

            if message_comp == 'zlib':
                message_data = pickle.loads(zlib.decompress(message_data))
            # If we receive a message
            if message['type'] == 'event':
                logger.debug('%s "%s" received event "%s"', self.__class__.__name__, self.name, message_data)
                if message['data'] == 'done':
                    break
            elif message['type'] == 'data':
                if not hasattr(message_data, 'size'):
                    message_data = np.array([message_data])
                message_data = message_data.flatten()

                logger.debug('%s "%s" received %d points', self.__class__.__name__, self.name, message_data.size)
                logger.debug("Now has %d of %d points.", stream.points_taken, stream.num_points())

                try:
                    self.up_to_date = (w_idx == self.data.len())
                except:
                    import ipdb; ipdb.set_trace()

                # Resize if necessary, also get the new set of sweep tuples since the axes must have changed
                if w_idx + message_data.size > self.data.len():
                    # Get new data size
                    num_points = stream.descriptor.num_points()
                    self.data.resize((num_points,))
                    logger.debug("HDF5 stream was resized to %d points", w_idx + message_data.size)

                    # Get and write new coordinate tuples to the main
                    # data set as well as the individual axis tables.
                    tuples = stream.descriptor.tuples(with_metadata=True, as_structured_array=True)
                    for axis_name in axis_names:
                        self.data[axis_name, w_idx:num_points] = tuples[axis_name][w_idx:num_points]
                    for i, a in enumerate(axes):
                        if a.unstructured:
                            name = "+".join(a.name)
                            if a.num_points() > self.group[name].len():
                                self.group[name].resize((a.num_points(),))
                                for j, col_name in enumerate(a.name):
                                    self.group[name][col_name,:] = a.points[:,j]

                self.data[desc.data_name, w_idx:w_idx+message_data.size] = message_data

                self.file.flush()
                w_idx += message_data.size
                self.points_taken = w_idx

                logger.debug("HDF5: Write index at %d", w_idx)
                logger.debug("HDF5: %s has written %d points", stream.name, w_idx)

        # try:
        #     self.file.close()
        #     self.file = None
        # except:
        #     # This doesn't seem to happen when we don't used named columns
        #     logger.debug("Ignoring 'dictionary changed sized during iteration' error.")

class DataBuffer(Filter):
    """Writes data to file."""

    sink = InputConnector()

    def __init__(self, **kwargs):
        super(DataBuffer, self).__init__(**kwargs)
        self.quince_parameters = []

    def final_init(self):
        self.descriptor = self.sink.input_streams[0].descriptor
        self.buffer = np.empty(self.descriptor.num_points(), dtype=self.sink.input_streams[0].descriptor.dtype)
        self.w_idx = 0

    async def process_data(self, data):
        if self.w_idx + data.size > self.buffer.size:
            # Create a new buffer and paste the old buffer into it
            old_buffer = self.buffer
            new_size = self.descriptor.num_points()
            self.buffer = np.empty(num_points(), dtype=self.descriptor.dtype)
            self.buffer[:old_buffer.size] = old_buffer

        self.buffer[self.w_idx:self.w_idx+data.size] = data
        self.w_idx += data.size

    def get_data(self):
        dtype = self.descriptor.axis_data_type(with_metadata=True)
        dtype.append((self.descriptor.data_name, self.descriptor.dtype))
        data = np.empty(self.buffer.size, dtype=dtype)

        tuples = self.descriptor.tuples(with_metadata=True, as_structured_array=True)
        for a in self.descriptor.axis_names(with_metadata=True):
            data[a] = tuples[a]
        data[self.descriptor.data_name] = self.buffer
        return data

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
