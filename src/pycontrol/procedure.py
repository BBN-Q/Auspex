import logging

import numpy as np
import scipy as sp
import pandas as pd
import h5py

import inspect
import time

from .instruments.instrument import Instrument

class Quantity(object):
    """Physical quantity to be measured."""
    def __init__(self, name, unit=None):
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
        logging.debug("Setting method of Quantity %s to %s" % (self.name, str(method)) )
        self.method = method

    def measure(self):
        logging.debug("%s Being asked to measure" % self.name)
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

    def __init__(self, name, unit=None, default=None, abstract=False):
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
        logging.debug("Setting method of Parameter %s to %s" % (self.name, str(method)) )
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

class Procedure(object):
    """The measurement loop to be run for each set of sweep parameters."""
    def __init__(self):
        super(Procedure, self).__init__()
        self._gather_parameters()
        self._gather_quantities()
        self._gather_instruments()

    def _gather_parameters(self):
        """ Collects all the Parameter objects for this procedure and stores\
        them in a dictionary.
        """
        self._parameters  = {}
        for item in dir(self):
            parameter = getattr(self, item)
            if isinstance(parameter, Parameter):
                self._parameters[item] = parameter

    def _gather_quantities(self):
        """ Collects all the Quantity objects for this procedure and stores\
        them in a dictionary.
        """
        self._quantities  = {}
        for item in dir(self):
            quantity = getattr(self, item)
            if isinstance(quantity, Quantity):
                self._quantities[item] = quantity

    def _gather_instruments(self):
        """ Collects all the Quantity objects for this procedure and stores\
        them in a dictionary.
        """
        self._instruments = {}
        for item in dir(self):
            inst = getattr(self, item)
            if isinstance(inst, Instrument):
                self._instruments[item] = inst

    def init_instruments(self):
        """Gets run before a sweep starts"""
        pass

    def shutdown_instruments(self):
        """Gets run after a sweep ends, or when the program is terminated."""
        pass

    def run(self):
        """The actual measurement that gets run for each set of values in a sweep."""
        pass
