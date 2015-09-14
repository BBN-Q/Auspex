from __future__ import print_function, division
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd

import time

class Command(object):
    """Wraps a particular device command set based on getter and setter strings. The optional
    value_map keyword argument allows specification of a dictionary map between python values
    such as True and False and the strange 'on' and 'off' type of values frequently used
    by instruments. Translation occurs via the provided 'convert_set' and 'convert_get' methods."""
    def __init__(self, name, set_string=None, get_string=None, value_map=None, type=float, range=None, allowed_values=None):
        """Initialize the class with optional set and get string corresponding to instrument
        commands. Also a map containing pairs of e.g. {python_value1: instr_value1, python_value2: instr_value2}"""
        super(Command, self).__init__()
        self.name = name
        self.set_string = set_string
        self.get_string = get_string

        self.doc = ""

        self.python_to_instr = None
        self.instr_to_python = None

        if value_map is not None:
            self.python_to_instr = value_map
            self.instr_to_python = {v: k for k, v in value_map.items()}

        # We neeed to do something or other
        if set_string is None and get_string is None:
            raise ValueError("Neither a setter nor a getter was specified.")

    def convert_set(self, set_value_python):
        """Convert the python value to a value understood by the instrument."""
        if self.python_to_instr is None:
            return set_value_python
        else:
            return self.python_to_instr[set_value_python]

    def convert_get(self, get_value_instrument):
        """Convert the instrument's returned values to something conveniently accessed
        through python."""
        if self.python_to_instr is None:
            return get_value_instrument
        else:
            return self.instr_to_python[get_value_instrument]

class Interface(object):
    """Currently just a dummy standing in for a PyVISA instrument."""
    def __init__(self):
        super(Interface, self).__init__()
    def write(self, value):
        logging.debug("Writing '%s'" % value)
    def query(self, value):
        logging.debug("Querying '%s'" % value)
        if value == ":output?;":
            return "on"
        return np.random.random()
    def values(self, query):
        logging.debug("Returning values %s" % query)
        return np.random.random()

def add_command(instr, name, cmd):
    def fget(self):
        val = self.interface.query(cmd.get_string)
        return cmd.convert_get(val)

    def fset(self, val):
        set_value = cmd.convert_set(val)
        self.interface.write(cmd.set_string % set_value)

    #Add getter and setter methods for passing around
    if cmd.get_string:
        setattr(instr, "get_" + name, fget)
    if cmd.set_string:
        setattr(instr, "set_" + name, fset)
    #Using None prevents deletion or setting/getting unsettable/gettable attributes
    setattr(instr, name, property(fget if cmd.get_string else None, fset if cmd.set_string else None, None, cmd.doc))

class MetaInstrument(type):
    """Meta class to create instrument classes with controls turned into descriptors.
    """
    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logging.debug("Adding controls to %s", name)
        for k,v in dct.items():
            if isinstance(v, Command):
                logging.debug("Adding '%s' command", k)
                add_command(self, k, v)

class Instrument(object):
    __metaclass__ = MetaInstrument

    def __init__(self, name, resource_name):
        self.name = name
        self.resource_name = resource_name

        self.interface = Interface()
