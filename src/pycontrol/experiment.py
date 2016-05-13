import logging
import asyncio
import inspect
import time
from functools import reduce

import numpy as np
import scipy as sp
import pandas as pd
import h5py

from .instruments.instrument import Instrument

logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s - %(levelname)s: \t%(asctime)s: \t%(message)s')
logger.setLevel(logging.INFO)

class Quantity(object):
    """Physical quantity to be measured."""
    def __init__(self, name=None, unit=None):
        super(Quantity, self).__init__()
        self.name   = name
        self.unit   = unit
        self.method = None
        self._value = None
        self.delay_before = 0
        self.delay_after = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def assign_method(self, method):
        logger.debug("Setting method of Quantity %s to %s" % (self.name, str(method)) )
        self.method = method

    def measure(self):
        logger.debug("%s Being asked to measure" % self.name)
        if self.delay_before is not None:
            time.sleep(self.delay_before)

        try:
            self._value = self.method()
        except:
            self._value = None
            print("Unable to measure %s." % self.name)

        if self.delay_after is not None:
            time.sleep(self.delay_after)

    def __str__(self):
        result = ""
        result += "%s" % str(self._value)
        if self.unit:
            result += " %s" % self.unit
        return result

    def __repr__(self):
        result = "<Quantity(name='%s'" % self.name
        result += ",value=%s" % repr(self._value)
        if self.unit:
            result += ",unit='%s'" % self.unit
        return result + ")>"

class DataAxis(object):
    """An axes in a data stream"""
    def __init__(self, label, points, unit=None):
        super(DataAxis, self).__init__()
        self.label  = label
        self.points = points
        self.unit   = unit
    def __repr__(self):
        return "<DataAxis(label={}, points={}, unit={})>".format(
            self.label, self.points, self.unit)

class DataStreamDescriptor(object):
    """Axis information"""
    def __init__(self):
        super(DataStreamDescriptor, self).__init__()
        self.axes = []

    def add_axis(self, axis):
        self.axes.append(axis)

    def num_dims(self):
        return len(self.axes)

    def num_points(self):
        return reduce(lambda x,y: x*y, [len(a.points) for a in self.axes])

    def __repr__(self):
        return "<DataStreamDescriptor(num_dims={}, num_points={})>".format(
            self.num_dims(), self.num_points())

class DataStream(object):
    """A stream of data"""
    def __init__(self):
        super(DataStream, self).__init__()
        self.queue = asyncio.Queue()
        self.points_taken = 0
        self.descriptor = None

    def set_descriptor(self, descriptor):
        self.descriptor = descriptor

    def num_points(self):
        return self.descriptor.num_points()

    def percent_complete(self):
        return 100.0*self.points_taken/self.num_points()

    def done(self):
        return self.points_taken >= self.num_points()

    def __repr__(self):
        return "<DataStream(completion={}%, descriptor={})>".format(
            self.percent_complete(), self.descriptor)

    async def push(self, data):
        self.points_taken += len(data)
        await self.queue.put(data)

class Trace(Quantity):
    """Holds a data array rather than a singe point."""
    def __init__(self, *args, **kwargs):
        super(Trace, self).__init__(*args, **kwargs)
        self._value = []
    def __repr__(self):
        result = "<Trace(name='%s'" % self.name
        result += ",value=%s" % repr(self._value)
        result += ",length=%i" % len(self._value)
        if self.unit:
            result += ",unit='%s'" % self.unit
        return result + ")>"

class Parameter(object):
    """ Encapsulates the information for an experiment parameter"""

    def __init__(self, name=None, unit=None, default=None, abstract=False):
        self.name     = name
        self._value   = default
        self.unit     = unit
        self.default  = default
        self.method   = None
        self.abstract = abstract # Is this something we can actually push?

        # Hooks to be called before or after updating a sweep parameter
        self.pre_push_hooks = []
        self.post_push_hooks = []

    def add_pre_push_hook(self, hook):
        self.pre_push_hooks.append(hook)

    def add_post_push_hook(self, hook):
        self.post_push_hooks.append(hook)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __str__(self):
        result = ""
        result += "%s" % str(self.value)
        if self.unit:
            result += " %s" % self.unit
        return result

    def __repr__(self):
        result = "<Parameter(name='%s'" % self.name
        result += ",value=%s" % repr(self.value)
        if self.unit:
            result += ",unit='%s'" % self.unit
        return result + ")>"

    def assign_method(self, method):
        logger.debug("Setting method of Parameter %s to %s" % (self.name, str(method)) )
        self.method = method

    def push(self):
        if not self.abstract:
            self.method(self._value)

class FloatParameter(Parameter):

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        try:
            self._value = float(value)
        except ValueError:
            raise ValueError("FloatParameter given non-float value of "
                             "type '%s'" % type(value))

    def __repr__(self):
        result = super(FloatParameter, self).__repr__()
        return result.replace("<Parameter", "<FloatParameter", 1)

class IntParameter(Parameter):

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        try:
            self._value = int(value)
        except ValueError:
            raise ValueError("IntParameter given non-int value of "
                             "type '%s'" % type(value))

    def __repr__(self):
        result = super(IntParameter, self).__repr__()
        return result.replace("<Parameter", "<IntParameter", 1)

class MetaProcedure(type):
    """Meta class to bake the instrument objects into a class description
    """

    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding controls to %s", name)
        self._parameters  = {}
        self._quantities  = {}
        self._instruments = {}
        self._traces      = {}

        for k,v in dct.items():
            if isinstance(v, Instrument):
                logger.debug("Found '%s' instrument", k)
                self._instruments[k] = v
            elif isinstance(v, Parameter):
                logger.debug("Found '%s' parameter", k)
                if v.name is None:
                    v.name = k
                self._parameters[k] = v
            elif isinstance(v, Quantity):
                logger.debug("Found '%s' quantity", k)
                if v.name is None:
                    v.name = k
                self._quantities[k] = v
            elif isinstance(v, Trace):
                logger.debug("Found '%s' trace", k)
                if v.name is None:
                    v.name = k
                self._traces[k] = v

class Procedure(metaclass=MetaProcedure):
    """The measurement loop to be run for each set of sweep parameters."""
    def __init__(self):
        super(Procedure, self).__init__()

    def init_instruments(self):
        """Gets run before a sweep starts"""
        pass

    def shutdown_instruments(self):
        """Gets run after a sweep ends, or when the program is terminated."""
        pass

    def run(self):
        """The actual measurement that gets run for each set of values in a sweep."""
        pass
