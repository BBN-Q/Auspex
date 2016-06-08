import asyncio
import operator
import functools
import h5py
import numpy as np
import os.path
import glob

from pycontrol.stream import DataStreamDescriptor
from .filter import Filter, InputConnector, OutputConnector

class WriteToHDF5(Filter):
    """Writes data to file."""

    data = InputConnector() # Can take multiple inputs

    def __init__(self, filename, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)
        # self.max_input_streams = 100 # We'll never do more than this, right?
        
        # Increment the filename until we find one we want.
        i = 0
        while os.path.exists("{:04d}-{}".format(i,filename)):
            i += 1
        self.filename = "{:04d}-{}".format(i,filename)
        self.file = h5py.File(self.filename, 'w')
        


    async def wait_for_data(self, stream):
        new_data = np.array(await stream.queue.get()).flatten()
        print("Writer stream {} got data {}".format(stream.name, new_data))
        return new_data.size

    async def run(self):

        self.file.create_group('data')
        num_axes = len(self.data.input_streams[0].descriptor.axes)
        num_points = self.data.input_streams[0].num_points()
        self.file.create_dataset("sweep_coords", (num_points,), dtype='f')

        # TODO: creates attributes for group including date, machine, software versions, etc.

        points_written = [0 for s in self.data.input_streams]
        points_per_stream = [s.num_points() for s in self.data.input_streams]

        # Establish the files
        for s in self.data.input_streams:
            descr_dims  = s.descriptor.data_dims(fortran=False)
            num_points = s.num_points()
            print("Creating table for stream {} with dims {}".format(s.name, descr_dims))
            self.file['data'].create_dataset(s.name, (num_points,), dtype='f')
            self.file['data'][s.name].attrs['stream_dims'] = descr_dims
            # TODO: add other datatypes, such as complex

        while True:
            if False not in list(map(operator.eq, points_written, points_per_stream)):
                break

            for i, s in enumerate(self.data.input_streams):
                points_written[i] += await self.wait_for_data(s)
                print("Writer stream {} got {} points".format(s.name, points_written[i]))

                # TODO: use futures so we don't block here

                    