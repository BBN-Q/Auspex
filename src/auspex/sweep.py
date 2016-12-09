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
            return True
        else:
            i=0
            while i<imax and self.axes[i].step==0:
                i += 1
            # Need to update parameters from outer --> inner axis
            for j in range(i,-1,-1):
                await self.axes[j].update()

    def done(self):
        return np.all([a.done for a in self.axes])

    def __repr__(self):
        return "Sweeper"
