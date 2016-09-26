# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio, concurrent

import numpy as np
from pycontrol.stream import DataStreamDescriptor, DataAxis
from pycontrol.filters.filter import Filter, InputConnector, OutputConnector
from pycontrol.log import logger

class Averager(Filter):
    """Takes data and collapses along the specified axis."""

    sink = InputConnector()
    partial_average = OutputConnector()
    final_average = OutputConnector()

    def __init__(self, axis, **kwargs):
        super(Averager, self).__init__(**kwargs)
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
            if self.sink.descriptor is not None:
                self.update_descriptors()
        else:
            raise ValueError("Must specify averaging axis as string.")

    def update_descriptors(self):
        logger.debug('Updating averager "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)
        descriptor_in = self.sink.descriptor
        names = [a.name for a in descriptor_in.axes]

        # Convert named axes to an index
        if self._axis not in names:
            raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self._axis, descriptor_in))
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

        # Define final axis descriptor
        # final_axes = descriptor_in.axes[:]
        descriptor_final = descriptor_in.copy()
        self.num_averages = descriptor_final.pop_axis(self.axis).num_points()
        logger.debug("Number of partial averages is %d", self.num_averages)

        # Define partial axis descriptor
        # partial_axes = descriptor_in.axes[:]
        descriptor_partial = descriptor_in.copy()
        descriptor_partial.pop_axis(self.axis)
        # descriptor_partial.axes = partial_axes
        # partial_axes.pop(self.axis_num)
        descriptor_partial.add_axis(DataAxis("Partial Averages", list(range(self.num_averages))))

        self.sum_so_far = np.zeros(self.avg_dims)
        self.partial_average.descriptor = descriptor_partial
        self.final_average.descriptor = descriptor_final

        for stream in self.partial_average.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, descriptor_partial)
            stream.set_descriptor(descriptor_partial)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, descriptor_partial)
            stream.end_connector.update_descriptors()

        for stream in self.final_average.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, descriptor_final)
            stream.set_descriptor(descriptor_final)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, descriptor_final)
            stream.end_connector.update_descriptors()

    def final_init(self):
        if self.points_before_final_average is None:
            raise Exception("Average has not been initialized. Run 'update_descriptors'")

        self.completed_averages = 0

        # We only need to accumulate up to the averaging axis
        # BUT we may get something longer at any given time!
        self.carry = np.zeros(0)

    async def process_data(self, data):

        # TODO: handle complex data
        data = data.real

        # TODO: handle unflattened data separately
        if len(data.shape) > 1:
            data = data.flatten()
        #handle single points
        elif not isinstance(data, np.ndarray) and (data.size == 1):
            data = np.array([data])

        if self.carry.size > 0:
            data = np.concatenate((self.carry, data))
            self.carry = np.zeros(0)

        idx = 0
        while idx < data.size:
            #check whether we have enough data to fill an averaging frame
            if data.size - idx >= self.points_before_partial_average:
                #reshape the data to an averaging frame
                self.sum_so_far += np.reshape(data[idx:idx+self.points_before_partial_average], self.avg_dims)
                idx += self.points_before_partial_average
                self.completed_averages += 1

                # Emit a partial average since we've accumulated enough data
                for os in self.partial_average.output_streams:
                    await os.push(self.sum_so_far/self.completed_averages)

            #otherwise add it to the carry
            else:
                self.carry = data[idx:]
                break

            #if we have finished averaging emit
            if self.completed_averages == self.num_averages:
                for os in self.final_average.output_streams:
                    await os.push(self.sum_so_far/self.num_averages)
                self.sum_so_far[:] = 0.0
                self.completed_averages = 0


class MultiAverage(Filter):
    """Takes data and collapses along the specified axes.
    Current approach: Store all data up to the outer most averaging axis \
    then collapse using numpy.mean. Not memory efficient.

    Future dev: Store (and accumulate) data of non-averaging axes.
    """

    data = InputConnector()
    average = OutputConnector()

    def __init__(self, axes, **kwargs):
        super(MultiAverage, self).__init__(**kwargs)
        self._axes = self.get_axes(axes)
        self.average_points = None

    @property
    def axes(self):
        return self._axes
    @axes.setter
    def axes(self, value):
        self._axes = self.get_axes(value)
        if self.sink.descriptor is not None:
            self.update_descriptors()

    def get_axes(self,value):
        """ axes must be a string or list of strings """
        if isinstance(value, str):
            return [value]
        value = list(value)
        if isinstance(value[0],str):
            return value
        raise ValueError("Must specify averaging axes as a string (one axis) or a list of strings.")

    def update_descriptors(self):
        logger.debug('Updating averager "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)
        descriptor_in = self.sink.descriptor
        names = [a.name for a in descriptor_in.axes]

        # Convert named axes to an index
        self.axes_num = []
        for axis in self._axes:
            if axis not in names:
                raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(axis, self.descriptor_in))
            self.axes_num.append(names.index(axis))
            logger.debug("Axis %s corresponds to numerical axis %d", axis, self.axes_num[-1])

        logger.debug("Averaging over axes #%s: %s", self.axes_num, self._axes)

        if len(self.axes_num)==0:
            logger.debug("Nothing to average. Work as a Passthrough filter.")
            self.average_points = 1
            self.avg_dims = [1]
        else:
            out_axis = min(self.axes_num)
            self.average_points = descriptor_in.num_points_through_axis(out_axis)
            self.data_dims = descriptor_in.data_dims()
            self.avg_dims = self.data_dims[out_axis:]

        logger.debug("Points before final average: %s.", self.average_points)
        logger.debug("Data dimensions are %s", self.data_dims)

        new_axes = descriptor_in.axes[:]
        self.avg_axes = []
        for axis in sorted(self.axes_num, reverse=True):
            new_axes.pop(axis)
            self.avg_axes.append(axis - out_axis)
        self.avg_axes = tuple(self.avg_axes)
        # self.num_averages = new_axes.num_points()
        # logger.debug("Number of partial averages is %d", self.num_averages)
        descriptor_out = DataStreamDescriptor()
        descriptor_out.axes = new_axes
        logger.debug("New axes after averaging: %s" %new_axes)

        self.average.descriptor = descriptor_out

        for stream in self.average.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, descriptor_out)
            stream.set_descriptor(descriptor_out)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, descriptor_out)
            stream.end_connector.update_descriptors()

    async def run(self):
        logger.debug('Running "%s" averager async loop', self.name)

        if self.average_points is None:
            raise Exception("Average has not been initialized. Run 'update_descriptors'")

        # completed_averages = 0

        # We only need to accumulate up to the outer most averaging axis
        # BUT we may get something longer at any given time!

        logger.debug("Established averager buffer of size %d", self.sink.input_streams[0].num_points())

        carry = np.zeros(0)

        while True:
            if self.sink.input_streams[0].done():
                # We've stopped receiving new input, make sure we've flushed the output streams
                if len(self.average.output_streams) > 0:
                    if all([os.done() for os in self.average.output_streams]):
                        logger.debug("Averager %s done", self.name)
                        break
                else:
                    logger.debug("Found no output stream. Averager %s done.", self.name)
                    break

            new_data = await self.sink.input_streams[0].queue.get()
            logger.debug("%s got data %s", self.name, new_data)
            logger.debug("Now has %d of %d points.", self.sink.input_streams[0].points_taken, self.sink.input_streams[0].num_points())

            # todo: handle unflattened data separately
            if len(new_data.shape) > 1:
                new_data = new_data.flatten()

            if carry.size > 0:
                new_data = np.concatenate((carry, new_data),axis=0)

            idx = 0

            # fill an averaging frame as long as possible
            while new_data.size - idx >= self.average_points:
                #reshape the data to an averaging frame
                data_tmp = np.reshape(new_data[idx:idx+self.average_points], self.avg_dims)
                idx += self.average_points

                for os in self.average.output_streams:
                    await os.push(np.mean(data_tmp, axis=self.avg_axes))

            # add the remnant to the carry
            carry = new_data[idx:]
