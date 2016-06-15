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
        self.file = h5py.File(self.filename, 'w')

    async def run(self):

        stream     = self.data.input_streams[0]
        axes       = stream.descriptor.axes
        data_dims  = stream.descriptor.data_dims()
        num_axes   = len(axes)
        chunk_size = axes[-1].num_points()

        self.file.create_dataset('data', data_dims, dtype='f', compression="gzip")
        data = self.file['data']

        # Go through and create axis dimensions
        for i, axis in enumerate(axes):

            points = np.array(axis.points)
            data.dims[i].label = axis.name

            if axis.unstructured:
                # Attach a dimension for each coordinates of the unstructured axis
                for j, cn in enumerate(axis.coord_names):
                    self.file[cn] = points[:,j]
                    data.dims.create_scale(self.file[cn], cn)
                    data.dims[i].attach_scale(self.file[cn])
                    logger.debug("HDF5: adding axis %s to dim %d", axis.name, i)
            else:
                logger.debug("HDF5: adding axis %s to dim %d", axis.name, i)
                self.file[axis.name] = points
                data.dims.create_scale(self.file[axis.name], axis.name)
                data.dims[i].attach_scale(self.file[axis.name])

        r_idx = 0
        w_idx = 0

        # Establish the files
        # logger.debug("Creating table for stream %s with dims %s", s.name, descr_dims)
        # self.file['data'].create_dataset(s.name, (num_points,), dtype='f')
        # self.file['data'][s.name].attrs['stream_dims'] = descr_dims
            # TODO: add other datatypes, such as complex

        temp = np.empty(stream.num_points())

        while True:
            if stream.done():
                break

            new_data = np.array(await stream.queue.get()).flatten()
            logger.debug("HDF5: %s got data %s", stream.name, new_data)
            
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
            # coord_start = np.array(np.unravel_index(idx, data_dims))
            # coord_end   = np.array(np.unravel_index(idx+new_data.size, data_dims))
            # if coord_start[-1] == 0 and coord_end[-1] == 0:
            #     logger.debug("HDF5: directly writing chunk.")
            #     # We are writing whole rows at a time!
            #     slicing = []
            #     # Work up slices for the first N-1 axes
            #     for si, ei in coord_start[:-1], coord_end[:-1]:
            #         if ei == si:
            #             slicing.append(ei)
            #         elif ei > si:
            #             slicing.append(slice(si,ei,None))
            #         else:
            #             logger.debug("HDF5: Can't work with %s, %s for some reason.", ei,si)
            #     # Add in the last axis
            #     slicing.append(slice(None, None, None))
            #     slicing = tuple(slicing)
            #     logger.debug("HDF5: coords are %s, %s.", coord_start, coord_end)
            #     logger.debug("HDF5: directly writing chunk with slice %s.", slicing)
            #     data[slicing] = new_data
            # else:
                # Off the grid. 
                # logger.debug("HDF5: data received in annoying increment. Avoid this if possible.")


            # logger.debug("HDF5: going from %s to %s: %s.", coord_start, coord_end, coord_diff)

            # num_chunks = int(new_data.size/chunk_size)
            # for i in range(num_chunks):
            #     # logger.debug("\twriting to coords %s", coords) 
            #     # data[*(coords[:-1]),:] = 
            #     idx += chunk_size

            

                # TODO: use futures so we don't block here

                    