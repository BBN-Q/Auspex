# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.log import logger

class Parameter(object):
    """ Encapsulates the information for an experiment parameter"""

    def __init__(self, name=None, unit=None, default=None):
        self.name     = name
        self._value   = default
        self.unit     = unit
        self.default  = default
        self.method   = None

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
        if self.method is not None:
            # logger.debug("Calling pre_push_hooks of Parameter %s with value %s" % (self.name, self._value) )
            for pph in self.pre_push_hooks:
                pph()
            # logger.debug("Calling method of Parameter %s with value %s" % (self.name, self._value) )
            self.method(self._value)
            # logger.debug("Calling post_push_hooks of Parameter %s with value %s" % (self.name, self._value) )
            for pph in self.post_push_hooks:
                pph()

class ParameterGroup(Parameter):
    """ An array of Parameters """
    def __init__(self, params, name=None):
        if name is None:
            names = '('
            for param in params:
                names += param.name
            self.name = names + ')'
        else:
            self.name = name

        self.parameters = params
        self._value   = [param.value for param in params]
        self.default  = [param.defaul for param in params]
        self.method   = [param.method for param in params]

        units = '('
        for param in params:
            if param.unit is None:
                units += 'None'
            else:
                units += param.unit
        self.unit     = units + ')'

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, values):
        self._value = values
        for param, value in zip(self.parameters, values):
            param.value = value

    def assign_method(self, methods):
        for param, method in zip(self.parameters,methods):
            param.assign_method(method)

    def push(self):
        for param in self.parameters:
            param.push()

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
