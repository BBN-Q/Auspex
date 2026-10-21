# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Framer']

import time
import itertools
import numpy as np

from .filter import Filter
from auspex.log import logger
from auspex.parameter import Parameter
from auspex.stream import InputConnector, OutputConnector

class Framer(Filter):
    """Mete out data in increments defined by the specified axis."""

    sink   = InputConnector()
    source = OutputConnector()
    axis   = Parameter()

    def __init__(self, axis=None, **kwargs):
        super(Framer, self).__init__(**kwargs)
        self.axis.value = axis
        self.points_before_final_average   = None
        self.points_before_partial_average = None
        self.sum_so_far = None
        self.num_averages = None

        self.quince_parameters = [self.axis]

    def final_init(self):
        descriptor_in = self.sink.descriptor
        names = [a.name for a in descriptor_in.axes]

        self.axis.allowed_values = names

        if self.axis.value is None:
            self.axis.value = descriptor_in.axes[0].name

        # Convert named axes to an index
        if self.axis.value not in names:
            raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self.axis.value, descriptor_in))
        self.axis_num = descriptor_in.axis_num(self.axis.value)
        logger.debug("Framing on axis #%d: %s", self.axis_num, self.axis.value)

        # Find how many points we want to spit out at a time
        self.data_dims = descriptor_in.data_dims()
        if self.axis_num == len(descriptor_in.axes) - 1:
            raise Exception("Framer has refused to frame along single points.")
        else:
            self.frame_points = descriptor_in.num_points_through_axis(self.axis_num+1)

        logger.debug("Points before emitting frame: %s.", self.frame_points)

        # For storing carryover if getting uneven buffers
        self.idx = 0
        self.carry = np.zeros(0, dtype=self.sink.descriptor.dtype)

    def process_data(self, data):
        # Append any data carried from the last run
        if self.carry.size > 0:
            data = np.concatenate((self.carry, data))

        # This is the largest number of frames we can emit for the time being
        num_frames = data.size // self.frame_points

        # This is the carryover that we'll store until next round.
        # If nothing is left then reset the carryover.
        remaining_points = data.size % self.frame_points
        if remaining_points > 0:
            if num_frames > 0:
                self.carry = data[-remaining_points:]
                data = data[:-remaining_points]
            else:
                self.carry = data
        else:
            self.carry = np.zeros(0, dtype=self.sink.descriptor.dtype)

        if num_frames > 0:
            for i in range(num_frames):
                for os in self.source.output_streams:
                    os.push(data[i*self.frame_points:(i+1)*self.frame_points])