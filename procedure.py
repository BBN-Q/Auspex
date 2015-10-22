from __future__ import print_function, division
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd
import h5py

import inspect
import time

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

    def __init__(self, name, unit=None, default=None):
        self.name     = name
        self._value   = default
        self.unit     = unit
        self.default  = default
        self.method   = None
        self.changed  = True

        # Special routines to execute
        self._pre_push_routines = []
        self._post_push_routines = []
        
    def check_if_changed(self, value):
        if value is None:
            self.changed = True
        else:
            if value != self._value:
                logging.debug("In '{:s}', value was {:s}, will be {:s}.".format(self.name, str(self._value), str(value)) )
                self.changed = True
            else:
                logging.debug("In '{:s}', value was {:s}, will be {:s}.".format(self.name, str(self._value), str(value)) )
                self.changed = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.check_if_changed(value)
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

    def add_pre_push_routine(self, routine):
        self._pre_push_routines.append(routine)

    def add_post_push_routine(self, routine):
        self._post_push_routines.append(routine)

    def push(self):
        if self.changed:
            logging.debug("Telling '{:s}' to call set method, since the value has changed.".format(self.name))
            if self._value is not None:
                [ppr() for ppr in self._pre_push_routines]
            self.method(self._value)
            if self._value is not None:
                [ppr() for ppr in self._post_push_routines]
        self.changed = False

class FloatParameter(Parameter):
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        self.check_if_changed(value)
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
        self.check_if_changed(value)
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
        self._parameters = {}
        self._quantities = {}

        self._gather_parameters()
        self._gather_quantities()

    def _gather_parameters(self):
        """ Collects all the Parameter objects for this procedure and stores\
        them in a dictionary.
        """
        for item in dir(self):
            parameter = getattr(self, item)
            if isinstance(parameter, Parameter):
                self._parameters[item] = parameter
    
    def _gather_quantities(self):
        """ Collects all the Quantity objects for this procedure and stores\
        them in a dictionary.
        """
        for item in dir(self):
            quantity = getattr(self, item)
            if isinstance(quantity, Quantity):
                self._quantities[item] = quantity

    def run(self):
        # for param in self._parameters:
        #     self._parameters[param].push()
        for quant in self._quantities:
            self._quantities[quant].measure()
