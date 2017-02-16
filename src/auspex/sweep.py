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

    def add_sweep(self, axis):
        self.axes.append(axis)
        logger.debug("Add sweep axis to sweeper object: {}".format(axis))

    async def update(self):
        """ Update the levels """
        logger.debug("Sweeper updates values.")
        imax = len(self.axes)-1
        if imax < 0:
            logger.debug("There are no sweep axis, only data axes.")
            return None
        else:
            i=0
            while i<imax and self.axes[i].step==0:
                i += 1
            # Need to update parameters from outer --> inner axis
            for j in range(i,-1,-1):
                await self.axes[j].update()
        
        # At this point all of the updates should have happened
        # return the current coordinates of the sweep. Return the 
        # reversed list since we store "innermost" axes last.
        values = []
        for a in self.axes[::-1]:
            if a.metadata:
                if type(a.value) in [np.ndarray, list]:
                    values.append(tuple(list(a.value) + [a.metadata_value])) 
                else:
                    values.append((a.value, a.metadata_value))
            else:
                if type(a.value) in [np.ndarray, list]:
                    values.append(tuple(a.value)) 
                else:
                    values.append((a.value,))
        return values

    async def check_for_refinement(self):
        refined_axes = []
        for a in self.axes:
            if await a.check_for_refinement():
                refined_axes.append(a.name)
        if len(refined_axes) > 1:
            raise Exception("More than one axis trying to refine simultaneously. This cannot be tolerated.")
        elif len(refined_axes) == 1:
            return refined_axes[0]
        else:
            return None

    def done(self):
        return np.all([a.done for a in self.axes])

    def __repr__(self):
        return "Sweeper"
