# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio, concurrent
import time

import numpy as np
from auspex.parameter import Parameter
from auspex.stream import DataStreamDescriptor, DataAxis
from auspex.filters.filter import Filter, InputConnector, OutputConnector
from auspex.log import logger

class Averager(Filter):
    """Takes data and collapses along the specified axis."""

    sink            = InputConnector()
    partial_average = OutputConnector()
    final_average   = OutputConnector()
    final_variance  = OutputConnector()
    axis            = Parameter(allowed_values=["round_robins", "segments", "time"])

    def __init__(self, averaging_axis=None, **kwargs):
        super(Averager, self).__init__(**kwargs)
        self.axis.value = averaging_axis
        self.points_before_final_average   = None
        self.points_before_partial_average = None
        self.sum_so_far = None
        self.num_averages = None

        self.quince_parameters = [self.axis]

        # Rate limiting for partial averages
        self.last_update     = time.time()
        self.update_interval = 0.5

    def update_descriptors(self):
        logger.debug('Updating averager "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)
        descriptor_in = self.sink.descriptor
        names = [a.name for a in descriptor_in.axes]

        self.axis.allowed_values = names

        if self.axis.value is None:
            self.axis.value = descriptor_in.axes[0].name

        # Convert named axes to an index
        if self.axis.value not in names:
            raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self.axis.value, descriptor_in))
        self.axis_num = names.index(self.axis.value)
        logger.debug("Axis %s corresponds to numerical axis %d", self.axis.value, self.axis_num)

        logger.debug("Averaging over axis #%d: %s", self.axis_num, self.axis.value)

        self.data_dims = descriptor_in.data_dims()
        if self.axis_num == len(descriptor_in.axes) - 1:
            logger.debug("Performing scalar average!")
            self.points_before_partial_average = 1
            self.avg_dims = [1]
        else:
            self.points_before_partial_average = descriptor_in.num_points_through_axis(self.axis_num+1)
            self.avg_dims = self.data_dims[self.axis_num+1:]

        # If we get multiple final average simultaneously
        self.reshape_dims = self.data_dims[self.axis_num:]
        if self.axis_num > 0:
            self.reshape_dims = [-1] + self.reshape_dims
        self.mean_axis = self.axis_num - len(self.data_dims)

        self.points_before_final_average   = descriptor_in.num_points_through_axis(self.axis_num)
        logger.debug("Points before partial average: %s.", self.points_before_partial_average)
        logger.debug("Points before final average: %s.", self.points_before_final_average)
        logger.debug("Data dimensions are %s", self.data_dims)
        logger.debug("Averaging dimensions are %s", self.avg_dims)

        # Define final axis descriptor
        descriptor = descriptor_in.copy()
        self.num_averages = descriptor.pop_axis(self.axis.value).num_points()
        logger.debug("Number of partial averages is %d", self.num_averages)

        self.sum_so_far                 = np.zeros(self.avg_dims, dtype=descriptor.dtype)
        self.current_avg_frame          = np.zeros(self.points_before_final_average, dtype=descriptor.dtype)
        self.partial_average.descriptor = descriptor
        self.final_average.descriptor   = descriptor

        for stream in self.partial_average.output_streams + self.final_average.output_streams:
            stream.set_descriptor(descriptor)
            stream.end_connector.update_descriptors()

        # Define variance axis descriptor
        descriptor = descriptor_in.copy()
        descriptor.pop_axis(self.axis.value)
        if descriptor.unit:
            descriptor.unit = descriptor.unit + "^2"
        descriptor.metadata["num_averages"] = self.num_averages
        self.final_variance.descriptor = descriptor

        for stream in self.final_variance.output_streams:
            stream.set_descriptor(descriptor)
            stream.end_connector.update_descriptors()

    def final_init(self):
        if self.points_before_final_average is None:
            raise Exception("Average has not been initialized. Run 'update_descriptors'")

        self.completed_averages = 0

        # We only need to accumulate up to the averaging axis
        # BUT we may get something longer at any given time!
        self.carry = np.zeros(0, dtype=self.final_average.descriptor.dtype)

    async def process_data(self, data):

        # TODO: handle unflattened data separately
        if len(data.shape) > 1:
            data = data.flatten()
        #handle single points
        elif not isinstance(data, np.ndarray) and (data.size == 1):
            data = np.array([data])

        if self.carry.size > 0:
            data = np.concatenate((self.carry, data))
            self.carry = np.zeros(0, dtype=self.final_average.descriptor.dtype)

        idx       = 0
        idx_frame = 0
        while idx < data.size:
            #check whether we have enough data to fill an averaging frame
            if data.size - idx >= self.points_before_final_average:
                # How many chunks can we process at once?
                num_chunks = int((data.size - idx)/self.points_before_final_average)
                new_points = num_chunks*self.points_before_final_average
                reshaped   = data[idx:idx+new_points].reshape(self.reshape_dims)
                averaged   = reshaped.mean(axis=self.mean_axis)
                idx       += new_points
 
                for os in self.final_average.output_streams:
                    await os.push(averaged)

                for os in self.final_variance.output_streams:
                    await os.push(reshaped.var(axis=self.mean_axis, ddof=1)) # N-1 in the denominator 

                for os in self.partial_average.output_streams:
                    await os.push(averaged)

            # Maybe we can fill a partial frame
            elif data.size - idx >= self.points_before_partial_average:
                # How many chunks can we process at once?
                num_chunks       = int((data.size - idx)/self.points_before_partial_average)
                new_points       = num_chunks*self.points_before_partial_average

                # Find the appropriate dimensions for the partial
                partial_reshape_dims = self.reshape_dims[:]
                partial_reshape_dims[self.mean_axis] = -1
                partial_reshape_dims = partial_reshape_dims[self.mean_axis:]

                reshaped         = data[idx:idx+new_points].reshape(partial_reshape_dims)
                summed           = reshaped.sum(axis=self.mean_axis)
                self.sum_so_far += summed
                idx             += new_points
                
                self.current_avg_frame[idx_frame:idx_frame+new_points] = data[idx:idx+new_points]
                idx_frame       += new_points

                self.completed_averages += num_chunks

                # If we now have enoough for the final average, push to both partial and final...
                if self.completed_averages == self.num_averages:
                    reshaped = self.current_avg_frame.reshape(self.reshape_dims)
                    for os in self.final_average.output_streams + self.partial_average.output_streams:
                        await os.push(reshaped.means(axis=self.mean_axis))
                    for os in self.final_variance.output_streams:
                        await os.push(reshaped.var(axis=self.mean_axis, ddof=1)) # N-1 in the denominator 
                    self.sum_so_far[:]      = 0.0
                    self.current_frame[:]   = 0.0
                    self.completed_averages = 0
                    self.idx_frame          = 0 
                else:
                    # Emit a partial average since we've accumulated enough data
                    if (time.time() - self.last_update >= self.update_interval):
                        for os in self.partial_average.output_streams:
                            await os.push(self.sum_so_far/self.completed_averages)
                        self.last_update = time.time()

            # otherwise just add it to the carry
            else:
                self.carry = data[idx:]
                break
