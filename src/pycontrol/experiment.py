import logging
import inspect
import time

import numpy as np
import scipy as sp
import pandas as pd
import h5py

from .instruments.instrument import Instrument
from .streams.stream import DataStream, DataAxis, DataStreamDescriptor

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

class SweptParameter(object):
    """Data structure for a swept Parameters, contains the Parameter
    object rather than subclassing it since we just need to keep track
    of some values"""
    def __init__(self, parameter, values):
        self.parameter = parameter
        self.values = values
        self.length = len(values)
        self.indices = range(self.length)

    @property
    def value(self):
        return self.parameter.value
    @value.setter
    def value(self, value):
        self.parameter.value = value

class MetaExperiment(type):
    """Meta class to bake the instrument objects into a class description
    """

    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding controls to %s", name)
        self._parameters     = {}
        self._quantities     = {}
        self._instruments    = {}
        self._traces         = {}
        self._output_streams = {}

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
            elif isinstance(v, DataStream):
                logger.debug("Found '%s' DataStream", k)
                if v.name is None:
                    v.name = k
                self._output_streams[k] = v

class Experiment(metaclass=MetaExperiment):
    """The measurement loop to be run for each set of sweep parameters."""
    def __init__(self):
        super(Experiment, self).__init__()

        # Iterable that yields sweep values
        self._sweep_generator = None

        # Container for patameters that will be swept
        self._swept_parameters = []

    def init_instruments(self):
        """Gets run before a sweep starts"""
        pass

    def shutdown_instruments(self):
        """Gets run after a sweep ends, or when the program is terminated."""
        pass

    # The method below is for custom control
    # Use cases: branching control, feeback loops

    def run(self):
        """A completely arbitrary specification of sweeping/data taking"""
        pass

    # The methods below are for "automatic" sweeping and running
    # Use cases: nested loops sweeping isntrument parameters

    def run_sweeps(self):
        """A run method that is restricted to pre-defined sweeps."""
        pass

    def add_sweep(self, param, sweep_list):
        p = SweptParameter(param, sweep_list)
        self._swept_parameters.append(p)
        self.generate_sweep()
        axis = DataAxis(param.name, sweep_list)
        for k, ds in self._data_streams.items():
            ds.descriptor.add_axis(axis)

    def generate_sweep(self):
        self._sweep_generator = itertools.product(*[sp.values for sp in self._swept_parameters])
        self._index_generator = itertools.product(*[sp.indices for sp in self._swept_parameters])

