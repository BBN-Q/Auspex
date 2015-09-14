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
    def __init__(self, name, set_string=None, get_string=None, value_map=None, type=float, value_range=None, allowed_values=None):
        """Initialize the class with optional set and get string corresponding to instrument
        commands. Also a map containing pairs of e.g. {python_value1: instr_value1, python_value2: instr_value2}"""
        super(Command, self).__init__()
        self.name = name
        self.set_string = set_string
        self.get_string = get_string

        self.python_to_instr = None
        self.instr_to_python = None

        if value_map is not None:
            self.python_to_instr = value_map
            self.instr_to_python = {v: k for k, v in value_map.items()}

        if allowed_values is None:
            self.allowed_values = None
        else:
            self.allowed_values = allowed_values

        if value_range is None:
            self.value_range = None
        else:
            self.value_range = (min(value_range), max(value_range))

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
        logging.debug("Writing %s" % value)
    def query(self, value):
        logging.debug("Querying %s" % value)
        if value == ":output?;":
            return "on"
        return np.random.random()
    def values(self, query):
        logging.debug("Returning values %s" % query)
        return np.random.random()    

class Instrument(object):
    """This provides all of the actual device functionality, and contains the interface class
    that allows for communication for the physial instrument. When subclassing Instrument, calling
    the __init__ method of this base class will parse the class attributes and convert any Command
    objects such as to provide convenient get_xx and set_xx setter/getter methods as well
    as python @properties therof."""

    parsed_commands = False

    def __init__(self, name, resource_name, check_errors_on_get=False, check_errors_on_set=False):
        super(Instrument, self).__init__()
        self.name = name
        self.resource_name = resource_name

        self._commands = {} 

        self.check_errors_on_get = check_errors_on_get
        self.check_errors_on_set = check_errors_on_set
        self.interface = Interface()

        # Parse class attributes, making sure not to do so multiple times
        # if we have multiple instances of the same instrument type.
        if not self.parsed_commands:
            self.parse_commands()
            setattr(self.__class__, 'parsed_commands', True)

    def parse_commands(self):
        """Go through the class attributes and process any Command classes of subclasses in order to 
        produce setter and getters methods, as appropriate, as well as defining property-style access
        to those meethods."""
        logging.debug("Parsing commands in %s" % self.name)
        for item in dir(self):
            command = getattr(self, item)
            if isinstance(command, Command):
                logging.debug("Processing command %s", command.name)
                self._commands[item] = command

                if command.get_string is not None:
                    # Using the default argument is a hacky way to create a local copy
                    # of the command object.
                    def fget(self, command=command):
                        value = self.interface.query(command.get_string)
                        return command.convert_get(value)

                    setattr(self.__class__, item, property(fget))
                    setattr(self.__class__, 'get_'+item, fget)

                if command.set_string is not None :
                    # Using the default argument is a hacky way to create a local copy
                    # of the command object. We can't create a setter only property.
                    def fset(self, value, command=command):
                        if command.range is not None:
                            if (value < command.range[0]) or (value > command.range[1]):
                                raise ValueError("Outside of the allowable range specified for instrument '%s'." % self.name)
                        if allowed_values is not None:
                            if value is not in self.allowed_values:
                                raise ValueError("Not in the allowable set of values specified for instrument '%s': %s" % (self.name, self.allowed_values) )
                        set_value = command.convert_set(value)
                        self.interface.write(command.set_string % set_value)
                    setattr(self.__class__, 'set_'+item, fset)

                if command.set_string is not None and command.get_string is not None:
                    setattr(self.__class__, item, property(fget, fset))

    def check_errors(self):
        pass
