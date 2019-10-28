# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Averager']

import time
import itertools
import numpy as np

from .filter import Filter
from auspex.log import logger
from auspex.parameter import Parameter, FloatParameter
from auspex.stream import InputConnector, OutputConnector, DataStreamDescriptor, DataAxis

def view_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    http://stackoverflow.com/questions/37079175/how-to-remove-a-column-from-a-structured-numpy-array-without-copying-it
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b

def remove_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to remove.

    Returns a view of the array `a` (not a copy).
    http://stackoverflow.com/questions/37079175/how-to-remove-a-column-from-a-structured-numpy-array-without-copying-it
    """
    dt = a.dtype
    keep_names = [name for name in dt.names if name not in names]
    return view_fields(a, keep_names)


class Averager(Filter):
    """Takes data and collapses along the specified axis."""

    sink            = InputConnector()
    partial_average = OutputConnector()
    source          = OutputConnector()
    final_variance  = OutputConnector()
    final_counts    = OutputConnector()
    axis            = Parameter()
    threshold       = FloatParameter()

    def __init__(self, axis=None, threshold=0.5, **kwargs):
        super(Averager, self).__init__(**kwargs)
        self.axis.value = axis
        self.threshold.value = threshold
        self.points_before_final_average   = None
        self.points_before_partial_average = None
        self.sum_so_far = None
        self.num_averages = None
        self.passthrough = False

        # Rate limiting for partial averages
        self.last_update     = time.time()
        self.update_interval = 0.5

    def update_descriptors(self):
        logger.debug('Updating averager "%s" descriptors based on input descriptor: %s.', self.filter_name, self.sink.descriptor)
        descriptor_in = self.sink.descriptor
        names = [a.name for a in descriptor_in.axes]

        self.axis.allowed_values = names

        if self.axis.value is None:
            self.axis.value = descriptor_in.axes[0].name

        # Convert named axes to an index
        if self.axis.value not in names:
            raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self.axis.value, descriptor_in))
        self.axis_num = descriptor_in.axis_num(self.axis.value)
        logger.debug("Averaging over axis #%d: %s", self.axis_num, self.axis.value)

        self.data_dims = descriptor_in.data_dims()
        # If we only have a single point along this axis, then just pass the data straight through
        if self.data_dims[self.axis_num] == 1:
            logger.debug("Averaging over a singleton axis")
            self.passthrough = True

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

        if len(descriptor.axes) == 0:
            # We will be left with only a single point here!
            descriptor.add_axis(DataAxis("result", [0]))

        self.sum_so_far                 = np.zeros(self.avg_dims, dtype=descriptor.dtype)
        self.current_avg_frame          = np.zeros(self.points_before_final_average, dtype=descriptor.dtype)
        self.partial_average.descriptor = descriptor
        self.source.descriptor          = descriptor
        self.excited_counts             = np.zeros(self.data_dims, dtype=np.int64)

        # We can update the visited_tuples upfront if none
        # of the sweeps are adaptive...
        desc_out_dtype = descriptor_in.axis_data_type(with_metadata=True, excluding_axis=self.axis.value)
        if not descriptor_in.is_adaptive():
            vals           = [a.points_with_metadata() for a in descriptor_in.axes if a.name != self.axis.value]
            nested_list    = list(itertools.product(*vals))
            flattened_list = [tuple((val for sublist in line for val in sublist)) for line in nested_list]
            descriptor.visited_tuples = np.core.records.fromrecords(flattened_list, dtype=desc_out_dtype)
        else:
            descriptor.visited_tuples = np.empty((0), dtype=desc_out_dtype)

        for stream in self.partial_average.output_streams:
            stream.set_descriptor(descriptor)
            stream.descriptor.buffer_mult_factor = 20
            stream.end_connector.update_descriptors()

        for stream in self.source.output_streams:
            stream.set_descriptor(descriptor)
            stream.end_connector.update_descriptors()

        # Define variance axis descriptor
        descriptor_var = descriptor_in.copy()
        descriptor_var.data_name = "Variance"
        descriptor_var.pop_axis(self.axis.value)
        if descriptor_var.unit:
            descriptor_var.unit = descriptor_var.unit + "^2"
        descriptor_var.metadata["num_averages"] = self.num_averages
        self.final_variance.descriptor= descriptor_var

        # Define counts axis descriptor
        descriptor_count = descriptor_in.copy()
        descriptor_count.data_name = "Counts"
        descriptor_count.dtype = np.float64
        descriptor_count.pop_axis(self.axis.value)
        descriptor_count.add_axis(DataAxis("state", [0,1]),position=0)
        if descriptor_count.unit:
            descriptor_count.unit = "counts"
        descriptor_count.metadata["num_counts"] = self.num_averages
        self.final_counts.descriptor = descriptor_count

        if not descriptor_in.is_adaptive():
            descriptor_var.visited_tuples = np.core.records.fromrecords(flattened_list, dtype=desc_out_dtype)
        else:
            descriptor_var.visited_tuples = np.empty((0), dtype=desc_out_dtype)

        for stream in self.final_variance.output_streams:
            stream.set_descriptor(descriptor_var)
            stream.end_connector.update_descriptors()

        for stream in self.final_counts.output_streams:
            stream.set_descriptor(descriptor_count)
            stream.end_connector.update_descriptors()

    def final_init(self):
        if self.points_before_final_average is None:
            raise Exception("Average has not been initialized. Run 'update_descriptors'")

        self.completed_averages = 0
        self.idx_frame          = 0
        self.idx_global         = 0
        # We only need to accumulate up to the averaging axis
        # BUT we may get something longer at any given time!
        self.carry = np.zeros(0, dtype=self.source.descriptor.dtype)

    def process_data(self, data):

        if self.passthrough:
            for os in self.source.output_streams:
                os.push(data)
            for os in self.final_variance.output_streams:
                os.push(data*0.0)
            for os in self.partial_average.output_streams:
                os.push(data)
            return

        # TODO: handle unflattened data separately
        if len(data.shape) > 1:
            data = data.flatten()
        #handle single points
        elif not isinstance(data, np.ndarray) and (data.size == 1):
            data = np.array([data])

        if self.carry.size > 0:
            data = np.concatenate((self.carry, data))
            self.carry = np.zeros(0, dtype=self.source.descriptor.dtype)

        idx       = 0
        while idx < data.size:
            #check whether we have enough data to fill an averaging frame
            if data.size - idx >= self.points_before_final_average:
                #logger.debug("Have {} points, enough for final avg.".format(data.size))
                # How many chunks can we process at once?
                num_chunks = int((data.size - idx)/self.points_before_final_average)
                new_points = num_chunks*self.points_before_final_average
                reshaped   = data[idx:idx+new_points].reshape(self.reshape_dims)
                averaged   = reshaped.mean(axis=self.mean_axis)
                idx       += new_points

                # do state assignment
                excited_states = (np.real(reshaped) > self.threshold.value).sum(axis=self.mean_axis)
                ground_states = self.num_averages - excited_states

                if self.sink.descriptor.is_adaptive():
                    new_tuples = self.sink.descriptor.tuples()[self.idx_global:self.idx_global + new_points]
                    new_tuples_stripped = remove_fields(new_tuples, self.axis.value)
                    take_axis = -1 if self.axis_num > 0 else 0
                    reduced_tuples = new_tuples_stripped.reshape(self.reshape_dims).take((0,), axis=take_axis)
                    self.idx_global += new_points

                # Add to Visited tuples
                if self.sink.descriptor.is_adaptive():
                    for os in self.source.output_streams + self.final_variance.output_streams + self.partial_average.output_streams:
                        os.descriptor.visited_tuples = np.append(os.descriptor.visited_tuples, reduced_tuples)

                for os in self.source.output_streams:
                    os.push(averaged)

                for os in self.final_variance.output_streams:
                    os.push(reshaped.var(axis=self.mean_axis, ddof=1)) # N-1 in the denominator

                for os in self.partial_average.output_streams:
                    os.push(averaged)

                for os in self.final_counts.output_streams:
                    os.push(ground_states)
                    os.push(excited_states)

            # Maybe we can fill a partial frame
            elif data.size - idx >= self.points_before_partial_average:
                # logger.info("Have {} points, enough for partial avg.".format(data.size))
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

                self.current_avg_frame[self.idx_frame:self.idx_frame+new_points] = data[idx:idx+new_points]
                idx             += new_points
                self.idx_frame  += new_points

                self.completed_averages += num_chunks

                # If we now have enoough for the final average, push to both partial and final...
                if self.completed_averages == self.num_averages:
                    reshaped = self.current_avg_frame.reshape(partial_reshape_dims)
                    for os in self.source.output_streams + self.partial_average.output_streams:
                        os.push(reshaped.mean(axis=self.mean_axis))
                    for os in self.final_variance.output_streams:
                        os.push(np.real(reshaped).var(axis=self.mean_axis, ddof=1)+1j*np.imag(reshaped).var(axis=self.mean_axis, ddof=1)) # N-1 in the denominator

                    # do state assignment
                    excited_states = (np.real(reshaped) < self.threshold.value).sum(axis=self.mean_axis)
                    ground_states  = self.num_averages - excited_states
                    for os in self.final_counts.output_streams:
                        os.push(ground_states)
                        os.push(excited_states)

                    self.sum_so_far[:]        = 0.0
                    self.current_avg_frame[:] = 0.0
                    self.completed_averages   = 0
                    self.idx_frame            = 0
                else:
                    # Emit a partial average since we've accumulated enough data
                    if (time.time() - self.last_update >= self.update_interval):
                        for os in self.partial_average.output_streams:
                            os.push(self.sum_so_far/self.completed_averages)
                        self.last_update = time.time()

            # otherwise just add it to the carry
            else:
                self.carry = data[idx:]
                break
