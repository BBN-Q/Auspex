import asyncio
import numpy as np
from pycontrol.stream import DataStreamDescriptor
from pycontrol.filters.filter import Filter, InputConnector, OutputConnector
from pycontrol.logging import logger

class Average(Filter):
    """Takes data and collapses along the specified axis."""

    data = InputConnector()
    partial_average = OutputConnector()
    final_average = OutputConnector()

    def __init__(self, axis, **kwargs):
        super(Average, self).__init__(**kwargs)
        self._axis = axis
        self.points_before_final_average   = None
        self.points_before_partial_average = None
        self.sum_so_far = None
        self.num_averages = None

    @property
    def axis(self):
        return self._axis
    @axis.setter
    def axis(self, value):
        if isinstance(value, str):
            self._axis = value
            if self.data.descriptor is not None:
                self.update_descriptors()
        else:
            raise ValueError("Must specify averaging axis as string.")

    def update_descriptors(self):
        logger.debug('Updating averager "%s" descriptors based on input descriptor: %s.', self.name, self.data.descriptor)
        descriptor_in = self.data.descriptor
        names = [a.name for a in descriptor_in.axes]

        # Convert named axes to an index
        if self._axis not in names:
            raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self._axis, self.descriptor_in))
        self.axis_num = names.index(self._axis)
        logger.debug("Axis %s corresponds to numerical axis %d", self._axis, self.axis_num)

        logger.debug("Averaging over axis #%d: %s", self.axis_num, self._axis)

        self.data_dims = descriptor_in.data_dims()
        if self.axis_num == len(descriptor_in.axes) - 1:
            logger.debug("Performing scalar average!")
            self.points_before_partial_average = 1
            self.avg_dims = [1]
        else:
            self.points_before_partial_average = descriptor_in.num_points_through_axis(self.axis_num+1)
            self.avg_dims = self.data_dims[self.axis_num+1:]
        self.points_before_final_average   = descriptor_in.num_points_through_axis(self.axis_num)
        logger.debug("Points before partial average: %s.", self.points_before_partial_average)
        logger.debug("Points before final average: %s.", self.points_before_final_average)
        logger.debug("Data dimensions are %s", self.data_dims)
        logger.debug("Averaging dimensions are %s", self.avg_dims)

        new_axes = descriptor_in.axes[:]
        self.num_averages = new_axes.pop(self.axis_num).num_points()
        logger.debug("Number of partial averages is %d", self.num_averages)
        descriptor_out = DataStreamDescriptor()
        descriptor_out.axes = new_axes

        self.sum_so_far = np.zeros(self.avg_dims)
        self.partial_average.descriptor = descriptor_out
        self.final_average.descriptor = descriptor_out

        for stream in self.partial_average.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, descriptor_in)
            stream.set_descriptor(descriptor_out)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, descriptor_in)
            stream.end_connector.update_descriptors()

        for stream in self.final_average.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, descriptor_out)
            stream.set_descriptor(descriptor_out)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, descriptor_out)
            stream.end_connector.update_descriptors()

    async def run(self):
        logger.debug('Running "%s" averager async loop', self.name)

        if self.points_before_final_average is None:
            raise Exception("Average has not been initialized. Run 'update_descriptors'")

        completed_averages = 0

        # We only need to accumulate up to the averaging axis
        # BUT we may get something longer at any given time!

        temp = np.empty(self.data.input_streams[0].num_points())
        logger.debug("Established averager buffer of size %d", self.data.input_streams[0].num_points())

        carry = np.zeros(0)

        while True:
            if self.data.input_streams[0].done():
                # We've stopped receiving new input, make sure we've flushed the output streams
                if len(self.final_average.output_streams) > 0:
                    if all([os.done() for os in self.final_average.output_streams]):
                        logger.debug("Averager %s done", self.name)
                        break

            new_data = await self.data.input_streams[0].queue.get()
            logger.debug("%s got data %s", self.name, new_data)
            logger.debug("Now has %d of %d points.", self.data.input_streams[0].points_taken, self.data.input_streams[0].num_points())

            # todo: handle unflattened data separately
            if len(new_data.shape) > 1:
                new_data = new_data.flatten()

            if carry.size > 0:
                new_data = np.concatenate(1, (carry, new_data))

            idx = 0
            while idx < new_data.size:
                #check whether we have enough data to fill an averaging frame
                if new_data.size - idx >= self.points_before_partial_average:
                    #reshape the data to an averaging frame
                    self.sum_so_far += np.reshape(new_data[idx:idx+self.points_before_partial_average], self.avg_dims)
                    idx += self.points_before_partial_average
                    completed_averages += 1
                #otherwise add it to the carry
                else:
                    carry = new_data[idx:]

                #if we have finished averaging emit
                if completed_averages == self.num_averages:
                    for os in self.final_average.output_streams:
                        await os.push(self.sum_so_far/self.num_averages)
                    self.sum_so_far = 0.0
                    completed_averages = 0
