import asyncio
import numpy as np
from pycontrol.stream import DataStreamDescriptor
from .filter import Filter, InputConnector, OutputConnector

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: \t%(message)s')
logger.setLevel(logging.INFO)

class Average(Filter):
    """Takes data and collapses along the specified axis."""

    data = InputConnector()
    partial_average = OutputConnector()
    final_average = OutputConnector()

    def __init__(self, axis=0, **kwargs):
        super(Average, self).__init__(**kwargs)
        self.axis = axis # Can be string on numerical index
        self.points_before_final_average   = None
        self.points_before_partial_average = None
        self.sum_so_far = None
        self.num_averages = None

    @property
    def axis(self):
        return self._axis
    @axis.setter
    def axis(self, value):
        self._axis = value
        if self.data.descriptor is not None:
            self.update_descriptors() 

    def update_descriptors(self):
        logger.debug("Updating averager descriptors based on input descriptor: %s.", self.data.descriptor)
        descriptor_in = self.data.descriptor
        names = [a.name for a in descriptor_in.axes]

        # Convert named axes to an index
        if isinstance(self.axis, str):
            if self.axis not in names:
                raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self.axis, self.descriptor_in))
            self.axis = names.index(self.axis)
            
        logger.debug("Averaging over axis #%d: %s", self.axis, names[self.axis])

        new_axes = descriptor_in.axes[:]
        self.num_averages = new_axes.pop(self.axis).num_points()
        logger.debug("Number of partial averages is %d", self.num_averages)
        descriptor_out = DataStreamDescriptor()
        descriptor_out.axes = new_axes


        self.points_before_partial_average = descriptor_in.num_points_through_axis(self.axis+1)
        self.points_before_final_average   = descriptor_in.num_points_through_axis(self.axis)

        self.data_dims = descriptor_in.data_dims() 
        self.avg_dims = self.data_dims[self.axis+1:]
        logger.debug("Data dimensions are %s", self.data_dims)
        logger.debug("Averaging dimensions are %s", self.avg_dims)

        if self.avg_dims == []:
            logger.debug("Performing scalar average!")
            self.points_before_final_average = descriptor_in.num_points()
            self.sum_so_far = 0.0
        else:
            self.sum_so_far = np.zeros(self.avg_dims)

        self.partial_average.descriptor = descriptor_in
        self.final_average.descriptor = descriptor_out

        for stream in self.partial_average.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, descriptor_in)
            stream.set_descriptor(descriptor_in)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, descriptor_in)
            stream.end_connector.update_descriptors()

        for stream in self.final_average.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, descriptor_out)
            stream.set_descriptor(descriptor_out)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, descriptor_out)
            stream.end_connector.update_descriptors()

    async def run(self):
        logger.debug("Running averager async loop")
        logger.debug("Points before partial average: %s.", self.points_before_partial_average)
        logger.debug("Points before final average: %s.", self.points_before_final_average)
        # import ipdb; ipdb.set_trace()

        if self.points_before_final_average is None:
            raise Exception("Average has not been initialized. Run 'update_descriptors'")

        idx = 0
        completed_averages = 0

        # We only need to accumulate up to the averaging axis
        # BUT we may get something longer at any given time!
        temp = np.empty(self.data.input_streams[0].num_points())
        logger.debug("Established averager buffer of size %d", self.data.input_streams[0].num_points())

        while True:
            if self.data.input_streams[0].done():
                # We've stopped receiving new input, make sure we've flushed the output streams
                if len(self.partial_average.output_streams + self.final_average.output_streams) > 0:
                    if False not in [os.done() for os in self.partial_average.output_streams + self.final_average.output_streams]:
                        break

            new_data = await self.data.input_streams[0].queue.get()
            logger.debug("%s got data %s", self.name, new_data)
            logger.debug("Now has %d of %d points.", self.data.input_streams[0].points_taken, self.data.input_streams[0].num_points())
            
            # todo: handle unflattened data separately
            if len(new_data.shape) > 1:
                new_data = new_data.flatten()

            temp[idx:idx+len(new_data)] = new_data
            idx += len(new_data)

            # Grab all of the partial data
            num_partials = int(idx/self.points_before_partial_average)
            if completed_averages + num_partials > self.num_averages:
                num_partials = self.num_averages - completed_averages

            logger.debug("Just got enough points for %d partial averages.", num_partials)
            for i in range(num_partials):
                # print("Adding to sum")
                b = i*self.points_before_partial_average
                e = b + self.points_before_partial_average
                self.sum_so_far += np.reshape(temp[b:e], self.avg_dims)
                # print("Sum is now {}".format(self.sum_so_far))

            completed_averages += num_partials
            logger.debug("Now has %d of %d averages.", completed_averages, self.num_averages)
            

            # Shift any extra data back to the beginnig of the array
            extra = idx - num_partials*self.points_before_partial_average
            temp[0:extra] = temp[num_partials*self.points_before_partial_average:num_partials*self.points_before_partial_average + extra]
            idx = extra

            if num_partials > 0:
                logger.debug("Now has %d of %d averages.", completed_averages, self.num_averages)
                for output_stream in self.partial_average.output_streams:
                    await output_stream.push(self.sum_so_far/completed_averages)

            if completed_averages == self.num_averages:
                for output_stream in self.final_average.output_streams:
                    await output_stream.push(self.sum_so_far/self.num_averages)
                self.sum_so_far = np.zeros(self.avg_dims)
                completed_averages = 0
            
