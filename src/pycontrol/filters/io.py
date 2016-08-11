import asyncio, concurrent
import itertools
import h5py
import numpy as np
import os.path

from pycontrol.stream import DataStreamDescriptor
from pycontrol.logging import logger
from pycontrol.filters.filter import Filter, InputConnector, OutputConnector
from tqdm import tqdm, tqdm_notebook

class WriteToHDF5(Filter):
    """Writes data to file."""

    data = InputConnector()
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
        filename = self.new_filename()
        head = os.path.dirname(filename)
        head = os.path.normpath(head)
        dirs = head.split(os.sep)
        # Check if path exists. If not, create new one(s).
        fulldir = ''
        for d in dirs:
            fulldir = os.path.join(fulldir, d)
            if not os.path.exists(fulldir):
                logger.debug("Create new directory: {}.".format(fulldir))
                os.mkdir(fulldir)
        logger.debug("Create new data file: %s." %filename)
        return h5py.File(filename, 'w')

    async def run(self):
        stream     = self.data.input_streams[0]
        desc       = stream.descriptor
        axes       = stream.descriptor.axes
        params     = stream.descriptor.params

        params['exp_src'] = stream.descriptor.exp_src
        num_axes   = len(axes)
        
        # All of the combinations for the present values of the sweep parameters only
        tuples     = np.array(stream.descriptor.tuples())
        logger.debug("Tuples for the current sweep are %s", tuples)

        # Create a 2D dataset with a 1D data column
        dtype = [(a.name, 'f') for a in axes]
        dtype.append((stream.start_connector.name, 'f'))
        if self.compress:
            data = self.file.create_dataset('data', (len(tuples),), dtype=dtype, 
                                        chunks=True, maxshape=(None,),
                                        compression='gzip')
        else:
            data = self.file.create_dataset('data', (len(tuples),), dtype=dtype, 
                                        chunks=True, maxshape=(None,))

        # Write pointer
        w_idx = 0 

        while True:
            logger.debug("HDF5 awaiting data")
            done, pending = await asyncio.wait((stream.finished(), stream.queue.get()), return_when=concurrent.futures.FIRST_COMPLETED)
            new_data = list(done)[0].result()
            if type(new_data)==bool:
                if new_data:
                    break
                else:
                    logger.debug("Printer %s awaiting data", self.name)
                    new_data = np.array(await stream.queue.get()).flatten()

            logger.debug("HDF5 stream has %d points", stream.points_taken)
            logger.debug("HDF5: %s got data %s of length %d", stream.name, new_data, new_data.size)

            # Resize if necessary, also get the new set of sweep tuples since the axes must have changed
            if w_idx + new_data.size > data.len():
                logger.debug("HDF5 stream was resized to %d points", w_idx + new_data.size)
                data.resize((w_idx + new_data.size,))
                tuples = np.array(stream.descriptor.tuples())

            # Write to table
            for i, axis in enumerate(axes):
                data[axis.name, w_idx:w_idx+new_data.size] = tuples[w_idx:w_idx+new_data.size, i]

            data[stream.start_connector.name, w_idx:w_idx+new_data.size] = new_data

            self.file.flush()
            w_idx += new_data.size
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
    data = InputConnector()
    def __init__(self, num=0, notebook=False):
        super(ProgressBar,self).__init__()
        self.num    = num
        self.notebook = notebook
        self.bars   = []
        self.w_id   = 0

    async def run(self):
        self.stream = self.data.input_streams[0]
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