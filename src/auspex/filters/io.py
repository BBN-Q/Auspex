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

from auspex.stream import DataStreamDescriptor
from auspex.log import logger
from auspex.filters.filter import Filter, InputConnector, OutputConnector
from tqdm import tqdm, tqdm_notebook

class WriteToHDF5(Filter):
    """Writes data to file."""

    sink = InputConnector()
    def __init__(self, filename, compress=True, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)
        self.compress = compress
        self.filename = filename
        self.points_taken = 0
        self.file = None

    def final_init(self):
        self.file = self.new_file()

    def new_filename(self):
        # Increment the filename until we find one we want.
        i = 0
        ext = self.filename.find('.h5')
        if ext > -1:
            filename = self.filename[:ext]
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
        self.filename = self.new_filename()
        head = os.path.dirname(self.filename)
        head = os.path.normpath(head)
        dirs = head.split(os.sep)
        # Check if path exists. If not, create new one(s).
        fulldir = ''
        for d in dirs:
            fulldir = os.path.join(fulldir, d)
            if not os.path.exists(fulldir):
                logger.debug("Create new directory: {}.".format(fulldir))
                os.mkdir(fulldir)
        logger.debug("Create new data file: %s." %self.filename)
        return h5py.File(self.filename, 'w')

    async def run(self):
        stream     = self.sink.input_streams[0]
        desc       = stream.descriptor
        axes       = stream.descriptor.axes
        params     = stream.descriptor.params
        axis_names = desc.axis_names()

        params['exp_src'] = stream.descriptor.exp_src
        num_axes   = len(axes)

        # All of the combinations for the present values of the sweep parameters only
        tuples     = np.array(stream.descriptor.tuples())

        # Create a 2D dataset with a 1D data column
        dtype = [(a, 'f') for a in axis_names]
        logger.debug("Data type for HDF5: %s", dtype)
        dtype.append((desc.data_name, 'f'))
        if self.compress:
            data = self.file.create_dataset('data', (len(tuples),), dtype=dtype,
                                        chunks=True, maxshape=(None,),
                                        compression='gzip')
        else:
            data = self.file.create_dataset('data', (len(tuples),), dtype=dtype,
                                        chunks=True, maxshape=(None,))

        # Write params into attrs
        for k,v in params.items():
            if k not in axis_names:
                data.attrs[k] = v

        # Include the fixed rectilinear axes if we have rectilinear sweeps
        if True not in [a.unstructured for a in axes]:
            for i, a in enumerate(axes):
                self.file[a.name] = a.points
                data.dims.create_scale(self.file[a.name], a.name)
                data.dims[0].attach_scale(self.file[a.name])
                data[a.name] = tuples[:,i]
        # Write the initial batch of coordinate tuples

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

                logger.debug('%s "%s" received %d points.', self.__class__.__name__, self.name, message_data.size)
                logger.debug("Now has %d of %d points.", stream.points_taken, stream.num_points())

                # Resize if necessary, also get the new set of sweep tuples since the axes must have changed
                if w_idx + message_data.size >= data.len():
                    logger.debug("HDF5 stream was resized to %d points", w_idx + message_data.size)
                    data.resize((w_idx + message_data.size,))
                    tuples = np.append(tuples, np.array(stream.descriptor.tuples()), axis=0)

                    # Write to table
                    for i, axis_name in enumerate(axis_names):
                        logger.debug("Setting %s to %s", axis_name, tuples[w_idx:w_idx+message_data.size, i])
                        data[axis_name, w_idx:w_idx+message_data.size] = tuples[w_idx:w_idx+message_data.size, i]

                data[desc.data_name, w_idx:w_idx+message_data.size] = message_data

                self.file.flush()
                w_idx += message_data.size
                self.points_taken = w_idx

                logger.debug("HDF5: Write index at %d", w_idx)
                logger.debug("HDF5: %s has written %d points", stream.name, w_idx)

        try:
            self.file.close()
        except:
            # This doesn't seem to happen when we don't used named columns
            logger.debug("Ignoring 'dictionary changed sized during iteration' error.")

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
