import asyncio
import itertools
import h5py
import numpy as np
import os.path

from pycontrol.stream import DataStreamDescriptor
from pycontrol.logging import logger
from pycontrol.filters.filter import Filter, InputConnector, OutputConnector
from tqdm import tqdm, tqdm_notebook

class WriteToHDF5_old(Filter):
    """Writes data to file."""

    data = InputConnector()
    def __init__(self, filename, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)

        # Increment the filename until we find one we want.
        i = 0
        ext = filename.find('.h5')
        if ext > -1:
            filename = filename[:ext]
        while os.path.exists("{}-{:04d}.h5".format(filename,i)):
            i += 1
        self.filename = "{}-{:04d}.h5".format(filename,i)

    def final_init(self):
        self.file = h5py.File(self.filename, 'a')

    async def run(self):
        stream     = self.data.input_streams[0]
        axes       = stream.descriptor.axes
        params     = stream.descriptor.params
        data_dims  = stream.descriptor.data_dims()
        params['exp_src'] = stream.descriptor.exp_src
        num_axes   = len(axes)
        chunk_size = axes[-1].num_points()

        # Increment the dataset name until we find one we want.
        ind = 0
        while "data-{:04d}".format(ind) in self.file.keys():
            ind += 1
        dataset_name = "data-{:04d}".format(ind)
        logger.debug("Creating dataset with name %s and axes %s", dataset_name, axes)
        if 'axes' not in self.file.keys():
            self.file.create_group('axes')
        self.file.create_dataset(dataset_name, data_dims, dtype='f', compression="gzip")
        data = self.file[dataset_name]

        axis_names = []
        # Go through and create axis dimensions
        for i, axis in enumerate(axes):

            points = np.array(axis.points)
            data.dims[i].label = axis.name

            if axis.unstructured:
                # Attach a dimension for each coordinates of the unstructured axis
                for j, cn in enumerate(axis.coord_names):
                    logger.debug("Appending coordinates %s to axis %s", cn, points[:,j])
                    new_axis_name = cn + '-' + dataset_name
                    self.file['axes'][new_axis_name] = points[:,j]
                    data.dims.create_scale(self.file['axes'][new_axis_name], cn)
                    data.dims[i].attach_scale(self.file['axes'][new_axis_name])
                    logger.debug("HDF5: adding axis %s to dim %d", axis.name, i)
                    axis_names.append(cn)
            else:
                logger.debug("HDF5: adding axis %s to dim %d", axis.name, i)
                new_axis_name =  axis.name + '-' + dataset_name
                self.file['axes'][new_axis_name] = points
                data.dims.create_scale(self.file['axes'][new_axis_name], axis.name)
                data.dims[i].attach_scale(self.file['axes'][new_axis_name])
                axis_names.append(axis.name)

        # Write params into attrs
        for k,v in params.items():
            if k not in axis_names:
                data.attrs[k] = v

        # Flush the buffers into disk
        self.file.flush()

        r_idx = 0
        w_idx = 0
        temp = np.empty(stream.num_points())
        while True:
            if stream.done() and w_idx == stream.num_points():
                break

            logger.debug("HDF5 awaiting data")
            new_data = np.array(await stream.queue.get()).flatten()
            while stream.queue.qsize() > 0:
                new_data = np.append(new_data, np.array(stream.queue.get_nowait()).flatten())

            logger.debug("HDF5 stream has %d points", stream.points_taken)
            logger.debug("HDF5: %s got data %s of length %d", stream.name, new_data, new_data.size)

            temp[r_idx:r_idx+new_data.size] = new_data
            logger.debug("HDF5: Data buffer is %s", temp)
            r_idx += new_data.size
            logger.debug("HDF5: Read index at %d", r_idx)

            num_chunks = int(r_idx/chunk_size)
            logger.debug("HDF5: got enough points for %d rows.", num_chunks)

            for i in range(num_chunks):
                coord = list(np.unravel_index(w_idx, data_dims))
                coord[-1] = slice(None, None, None)
                data[tuple(coord)] = temp[i*chunk_size:(i+1)*chunk_size]
                w_idx += chunk_size

            logger.debug("HDF5: Write index at %d", w_idx)

            # import ipdb; ipdb.set_trace()
            if num_chunks > 0:
                extra = r_idx - num_chunks*chunk_size
                temp[0:extra] = temp[num_chunks*chunk_size:num_chunks*chunk_size + extra]
                r_idx = extra

            # Force flush data into disk
            self.file.flush()
            logger.debug("HDF5: %s has written %d points", stream.name, w_idx)

        self.file.close()


class WriteToHDF5(Filter):
    """Writes data to file."""

    data = InputConnector()
    def __init__(self, filename, dsplit=None, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)
        self.filename = self.get_filename(filename)
        self.dsplit = dsplit
        self.points_taken = 0

    def final_init(self):
        self.file = h5py.File(self.filename, 'a')

    def get_filename(self,filename):
        # Increment the filename until we find one we want.
        i = 0
        ext = filename.find('.h5')
        if ext > -1:
            filename = filename[:ext]
        while os.path.exists("{}-{:04d}.h5".format(filename,i)):
            i += 1
        return "{}-{:04d}.h5".format(filename,i)

    def new_file(self):
        filename = self.filename[:-8]
        self.filename = self.get_filename(filename)
        self.final_init()

    def new_dataset(self, data_dims):
        # Increment the dataset name until we find one we want.
        ind = 0
        while "data-{:04d}".format(ind) in self.file.keys():
            ind += 1
        dset_name =  "data-{:04d}".format(ind)
        logger.debug("Creating dataset with name %s and dimensions %s", dset_name, data_dims)
        return self.file.create_dataset(dataset_name, data_dims, dtype='f', compression="gzip")

    def get_axis_index(self, axis):
        # Convert named axis to an index
        if isinstance(axis, int):
            return axis # User is responsible if axis is out of range?

        if isinstance(axis, str):
            names = [a.name for a in self.data.input_streams[0].descriptor.axes]
            if axis not in names:
                raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(axis, self.data.input_streams[0].descriptor))
            logger.debug("Axis %s corresponds to numerical axis %d", self._axis, self.axis_num)
            return names.index(axis)

    async def run(self):
        stream     = self.data.input_streams[0]
        axes       = stream.descriptor.axes
        params     = stream.descriptor.params
        
        data_dims  = stream.descriptor.data_dims()
        params['exp_src'] = stream.descriptor.exp_src
        num_axes   = len(axes)
        chunk_size = axes[-1].num_points()

        dsplit     = self.get_axis_index(self.dsplit)
        axes_in    = axes[dsplit+1:]
        axes_out   = axes[0:dsplit+1]
        dims_in    = data_dims[dsplit+1:]
        dims_out   = data_dims[0:dsplit+1]
        dset_num   = 1 if len(dims_out)==0 else itertools.product(*[dim for dim in dims_out])
        dset_numpoints = stream.num_points_through_axis(dsplit+1)

        if 'axes' not in self.file.keys():
            self.file.create_group('axes')
        # Create an index table
        id_table = self.file.create_dataset('index', (len(dims_out)+2,dset_num), dtype='str')
        id_table.dims[0].label = 'ID'
        id_table.dims[1].label = 'Dataset Name'
        for i in range(len(dims_out)):
            id_table.dims[i+2].label = axes_out[i].name

        r_idx = 0
        w_idx = 0
        dset_id = 0
        dset_complete = True
        temp = np.empty(stream.num_points())
        while True:
            if stream.done or self.points_taken == stream.num_points():
                break

            if dset_complete:
                data = self.new_dataset(dims_in)
                axis_names = []
                # Go through and create axis dimensions
                for i, axis in enumerate(axes_in):
                    points = np.array(axis.points)
                    data.dims[i].label = axis.name
                    logger.debug("HDF5: adding axis %s to dim %d", axis.name, i)
                    new_axis_name =  axis.name + '-' + dataset_name
                    self.file['axes'][new_axis_name] = points
                    data.dims.create_scale(self.file['axes'][new_axis_name], axis.name)
                    data.dims[i].attach_scale(self.file['axes'][new_axis_name])
                    axis_names.append(axis.name)
                # Write params into attrs
                for k,v in params.items():
                    if k not in axis_names:
                        data.attrs[k] = v
                # Update index table
                # id_table[dset_id] = [] #???

                # Flush the buffers into disk
                self.file.flush()
                dset_complete = False
                dset_id += 1

            logger.debug("HDF5 awaiting data")
            new_data = np.array(await stream.queue.get()).flatten()
            while stream.queue.qsize() > 0:
                new_data = np.append(new_data, np.array(stream.queue.get_nowait()).flatten())

            logger.debug("HDF5 stream has %d points", stream.points_taken)
            logger.debug("HDF5: %s got data %s of length %d", stream.name, new_data, new_data.size)

            temp[r_idx:r_idx+new_data.size] = new_data
            logger.debug("HDF5: Data buffer is %s", temp)
            r_idx += new_data.size
            logger.debug("HDF5: Read index at %d", r_idx)

            num_chunks = int(r_idx/chunk_size)
            logger.debug("HDF5: got enough points for %d rows.", num_chunks)

            for i in range(num_chunks):
                coord = list(np.unravel_index(w_idx, data_dims))
                coord[-1] = slice(None, None, None)
                data[tuple(coord)] = temp[i*chunk_size:(i+1)*chunk_size]
                w_idx += chunk_size

            logger.debug("HDF5: Write index at %d", w_idx)

            # import ipdb; ipdb.set_trace()
            if num_chunks > 0:
                extra = r_idx - num_chunks*chunk_size
                temp[0:extra] = temp[num_chunks*chunk_size:num_chunks*chunk_size + extra]
                r_idx = extra

            # Force flush data into disk
            self.file.flush()
            logger.debug("HDF5: %s has written %d points", stream.name, w_idx)
            if w_idx >= dset_numpoints-1:
                dset_complete = True
                self.points_taken += w_idx
                w_idx = 0

        self.file.close()


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