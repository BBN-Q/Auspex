import itertools
import numpy as np

from pycontrol.parameter import ParameterGroup, FloatParameter, IntParameter, Parameter
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, InputConnector, OutputConnector
from pycontrol.logging import logger

# class SweptParameter(object):
#     """Data structure for a swept Parameters, contains the Parameter
#     object rather than subclassing it since we just need to keep track
#     of some values"""
#     def __init__(self, parameter, values):
#         self.parameter = parameter
#         self.associated_axes = []
#         self.update_values(values)
#         self.push = self.parameter.push

#     def update_values(self, values):
#         self.values = values
#         self.length = len(values)
#         for axis in self.associated_axes:
#             axis.points = self.values

#     def add_values(self, values):
#         self.values.extend(values)
#         self.length = len(self.values)
#         for axis in self.associated_axes:
#             axis.points = self.values

#     @property
#     def value(self):
#         return self.parameter.value
#     @value.setter
#     def value(self, value):
#         self.parameter.value = value

#     def __repr__(self):
#         return "<SweptParameter: {}>".format(self.parameter.name)

# class SweptParameterGroup(object):
#     """For unstructured (meshed) coordinate tuples. The actual values
#     are stored locally as _values, and we acces each tuple by indexing
#     into that array."""
#     def __init__(self, parameters, values):
#         self.parameters = parameters
#         self.associated_axes = []
#         self.update_values(values)

#     def push(self):
#         # Values here will just be the index
#         for p in self.parameters:
#             p.push()

#     def update_values(self, values):
#         self._values = values
#         self.length = len(values)
#         self.values = list(range(self.length)) # Dummy index list for sweeper
#         for axis in self.associated_axes:
#             axis.points = self._values

#     def add_values(self, values):
#         self._values.extend(values)
#         self.length = len(self._values)
#         for axis in self.associated_axes:
#             axis.points = self._values

#     @property
#     def value(self):
#         return [p.value for p in self.parameters]
#     @value.setter
#     def value(self, index):
#         for i, p in enumerate(self.parameters):
#             p.value = self._values[index][i]

#     def __repr__(self):
#         return "<SweptParameterGroup: {}>".format([p.name for p in self.parameters])

class SweepAxis(DataAxis):
    """ Structure for swept axis, separate from DataAxis.
    Can be an unstructured axis, in which case parameter is actually a list of parameters. """
    def __init__(self, parameter, points = [], refine_func=None, refine_args=[]):
        super(SweepAxis, self).__init__("Name", points)
        self.unstructured = hasattr(parameter, '__iter__')
        self.parameter    = parameter
        if self.unstructured:
            self.unit  = [p.unit for p in parameter]
            self.value = [None for p in parameter]
        else:
            self.unit = parameter.unit
            self.value     = None
        
        self.refine_func = refine_func
        self.refine_args = refine_args
        self.step        = 0
        self.done        = False

        if self.unstructured and len(parameter) != len(points[0]):
            raise ValueError("Parameter value tuples must be the same length as the number of parameters.")

        logger.debug("Created {}".format(self.__repr__()))

    def update(self):
        """ Update value after each run.
        If refine_func is None, loop through the list of points.
        """
        if self.step < self.num_points():
            self.value = self.points[self.step]
            logger.debug("Sweep Axis '{}' at step {} takes value: {}.".format(self.name,
                                                                               self.step,self.value))
            self.push()
            self.step += 1
            self.done = False
        if self.step==self.num_points():
            # Check to see if we need to perform any refinements
            if self.refine_func is not None:
                if self.refine_func(self, *self.refine_args):
                    # Refine_func should return true if we have more refinements...
                    self.value = self.points[self.step]
                    self.push()
                    self.step += 1
                    self.done = False
                else:
                    self.step = 0
                    self.done = True
                    logger.debug("Sweep Axis '{}' complete.".format(self.name))
            else:
                self.step = 0
                self.done = True
        else:
            self.done = False
            logger.debug("Sweep Axis '{}' complete.".format(self.name))

    def push(self):
        """ Push parameter value(s) """
        if self.unstructured:
            for p, v in zip(self.parameter, self.value):
                p.value = v
                p.push()
        else:
            self.parameter.value = self.value
            self.parameter.push()

    def __repr__(self):
        return "<SweepAxis(name={},length={},unit={},value={}>".format(self.name,self.num_points(),self.unit,self.value)

class Sweeper(object):
    """ Control center of sweep axes """
    def __init__(self):
        self.axes = []
        logger.debug("Generate Sweeper.")

    def add_sweep(self, axis):
        self.axes.append(axis)
        logger.debug("Add sweep axis: {}".format(axis))

    def update(self):
        """ Update the levels """
        logger.debug("Sweeper updates values.")
        imax = len(self.axes)-1
        i=0
        while i<imax and self.axes[i].step==0:
            i += 1
        # Need to update parameters from outer --> inner axis
        for j in range(i,-1,-1):
            self.axes[j].update()
        return np.all([a.done for a in self.axes])

    def __repr__(self):
        return "Sweeper"
