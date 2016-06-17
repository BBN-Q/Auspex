import asyncio
import operator
import functools
import h5py
import numpy as np
import os.path
import glob

from pycontrol.stream import DataStreamDescriptor
from .filter import Filter, InputConnector, OutputConnector

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: \t%(message)s')
logger.setLevel(logging.INFO)

class WriteToHDF5(Filter):
    """Writes data to file."""

    data = InputConnector()
    def __init__(self, filename, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)

        # Increment the filename until we find one we want.
        i = 0
        while os.path.exists("{:04d}-{}".format(i,filename)):
            i += 1
        self.filename = "{:04d}-{}".format(i,filename)

    async def run(self):

        self.file = h5py.File(self.filename, 'w')

        stream     = self.data.input_streams[0]
        axes       = stream.descriptor.axes
        data_dims  = stream.descriptor.data_dims()
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
        # import ipdb; ipdb.set_trace()

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
            else:
                logger.debug("HDF5: adding axis %s to dim %d", axis.name, i)
                new_axis_name =  axis.name + '-' + dataset_name
                self.file['axes'][new_axis_name] = points
                data.dims.create_scale(self.file['axes'][new_axis_name], axis.name)
                data.dims[i].attach_scale(self.file['axes'][new_axis_name])

        r_idx = 0
        w_idx = 0

        temp = np.empty(stream.num_points())

        while True:
            if stream.done():
                break

            logger.debug("HDF5 awaiting data")
            new_data = np.array(await stream.queue.get()).flatten()
            logger.debug("HDF5 stream has %d points", stream.points_taken)
            logger.debug("HDF5: %s got data %s of length %d", stream.name, new_data, new_data.size)

            temp[r_idx:r_idx+new_data.size] = new_data
            r_idx += new_data.size

            num_chunks = int(new_data.size/chunk_size)
            logger.debug("HDF5: got enough points for %d rows.", num_chunks)

            for i in range(num_chunks):
                coord = list(np.unravel_index(w_idx, data_dims))
                coord[-1] = slice(None, None, None)
                data[tuple(coord)] = temp[i*chunk_size:(i+1)*chunk_size]
                w_idx += chunk_size

            extra = r_idx - num_chunks*chunk_size
            temp[0:extra] = temp[num_chunks*chunk_size:num_chunks*chunk_size + extra]
            r_idx = extra

            logger.debug("HDF5: %s has written %d points", stream.name, w_idx)

        self.file.close()
