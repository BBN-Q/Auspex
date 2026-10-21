# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import itertools
import numpy as np

from auspex.parameter import ParameterGroup, FloatParameter, IntParameter, Parameter
from auspex.stream import DataStream, DataAxis, SweepAxis, DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger

class Sweeper(object):
    """ Control center of sweep axes """
    def __init__(self):
        self.axes = []
        logger.debug("Generate Sweeper.")

    def swept_parameters(self):
        swept_axes = []
        for a in self.axes:
            if a.unstructured:
                swept_axes.extend(a.parameter)
            else:
                swept_axes.append(a.parameter)
        return swept_axes

    def add_sweep(self, axis):
        self.axes.append(axis)
        logger.debug("Add sweep axis to sweeper object: {}".format(axis))

    def update(self):
        """ Update the levels """
        imax = len(self.axes)-1
        if imax < 0:
            logger.debug("There are no sweep axis, only data axes.")
            return None, None
        else:
            i=0
            while i<imax and self.axes[i].step==0:
                i += 1
            # Need to update parameters from outer --> inner axis
            for j in range(i,-1,-1):
                self.axes[j].update()

        # At this point all of the updates should have happened
        # return the current coordinates of the sweep. Return the
        # reversed list since we store "innermost" axes last.
        values = []
        names  = []
        for a in self.axes[::-1]:
            names.append(a.name)
            if a.metadata is not None:
                if type(a.value) in [np.ndarray, list]:
                    values.append(tuple(list(a.value) + [a.metadata_value]))
                else:
                    values.append((a.value, a.metadata_value))
            else:
                if type(a.value) in [np.ndarray, list]:
                    values.append(tuple(a.value))
                else:
                    values.append((a.value,))
        return values, names

    def is_adaptive(self):
        return True in [a.refine_func is not None for a in self.axes]

    def check_for_refinement(self, output_connectors_dict):
        refined_axes = []
        for a in self.axes:
            if a.check_for_refinement(output_connectors_dict):
                refined_axes.append(a.name)
                break
        if len(refined_axes) > 1:
            raise Exception("More than one axis trying to refine simultaneously. This cannot be tolerated.")

    def done(self):
        return np.all([a.done for a in self.axes])

    def __repr__(self):
        return "Sweeper"
