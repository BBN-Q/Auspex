# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0


import numpy as np
import scipy as sp
import pandas as pd

import visa

import os
import time

from pycontrol.logging import logger

class StringCommand(object):
    """Wraps a particular device command set based on getter and setter strings. The optional
    value_map keyword argument allows specification of a dictionary map between python values
    such as True and False and the strange 'on' and 'off' type of values frequently used
    by instruments. Translation occurs via the provided 'convert_set' and 'convert_get' methods."""
    formatter = '{}'

    def __init__(self, name=None, set_string=None, get_string=None, scpi_string=None, value_map=None, value_range=None,
                 allowed_values=None, aliases=None, set_delay=0.0, get_delay=0.0, additional_args=None):
        """Initialize the class with optional set and get string corresponding to instrument
        commands. Also a map containing pairs of e.g. {python_value1: instr_value1, python_value2: instr_value2, ...}."""

        super(StringCommand, self).__init__()
        self.aliases = aliases
        self.set_delay = set_delay
        self.get_delay = get_delay

        if scpi_string:
            # Construct get and set strings using this base scpi_string
            self.get_string = scpi_string+"?;"
            self.set_string = scpi_string+" "+self.formatter
        else:
            self.set_string = set_string
            self.get_string = get_string

        self.doc = ""

        self.python_to_instr = None
        self.instr_to_python = None

        if value_range is None:
            self.value_range = None
        else:
            self.value_range = (min(value_range), max(value_range))

        self.allowed_values = allowed_values
        self.additional_args = additional_args

        if value_map is not None:
            self.python_to_instr = value_map
            self.instr_to_python = {v: k for k, v in value_map.items()}

            if self.value_range is not None:
                raise Exception("Cannot specify both value_range and value_map as they are redundant.")

            if self.allowed_values is not None:
                raise Exception("Cannot specify both value_map and allowed_values as they are redundant.")
            else:
                self.allowed_values=list(self.python_to_instr.keys())

            logger.debug("Constructed map and inverse map for command values:\n--- %s\n--- %s'" % (self.python_to_instr, self.instr_to_python))

        # We neeed to do something or other
        if self.set_string is None and self.get_string is None:
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

class FloatCommand(StringCommand):
    formatter = '{:E}'
    def convert_get(self, get_value_instrument):
        """Convert the instrument's returned values to something conveniently accessed
        through python."""
        if self.python_to_instr is None:
            return float(get_value_instrument)
        else:
            return float(self.instr_to_python[get_value_instrument])

class IntCommand(StringCommand):
    formatter = '{:d}'
    def convert_get(self, get_value_instrument):
        """Convert the instrument's returned values to something conveniently accessed
        through python."""
        if self.python_to_instr is None:
            return int(get_value_instrument)
        else:
            return int(self.instr_to_python[get_value_instrument])

class RampCommand(FloatCommand):
    """For quantities that are to be ramped from present to desired value. These will always be floats..."""
    def __init__(self, increment, pause=0.0, **kwargs):
        super(RampCommand, self).__init__(**kwargs)
        self.increment = increment
        self.pause = pause

class Interface(object):
    """Currently just a dummy standing in for a PyVISA instrument."""
    def __init__(self):
        super(Interface, self).__init__()
    def write(self, value):
        logger.debug("Writing '%s'" % value)
    def query(self, value):
        logger.debug("Querying '%s'" % value)
        if value == ":output?;":
            return "on"
        return np.random.random()
    def values(self, query):
        logger.debug("Returning values %s" % query)
        return np.random.random()

class VisaInterface(Interface):
    """PyVISA interface for communicating with instruments."""
    def __init__(self, resource_name):
        super(VisaInterface, self).__init__()
        try:
            if os.name == "nt":
                visa_loc = 'C:\\windows\\system32\\visa64.dll'
                rm = visa.ResourceManager(visa_loc)
            else:
                rm = visa.ResourceManager("@py")
            self._resource = rm.open_resource(resource_name)
        except:
            raise Exception("Unable to create the resource '%s'" % resource_name)
    def values(self, query_string):
        return self._resource.query_ascii_values(query_string, container=np.array)
    def value(self, query_string):
        return self._resource.query_ascii_values(query_string)
    def write(self, write_string):
        self._resource.write(write_string)
    def write_raw(self, raw_string):
        self._resource.write_raw(raw_string)
    def read(self):
        return self._resource.read()
    def read_raw(self):
        return self._resource.read_raw()
    def query(self, query_string):
        return self._resource.query(query_string)
    def write_binary_values(self, query_string, values, **kwargs):
        return self._resource.write_binary_values(query_string, values, **kwargs)
    def query_binary_values(self, query_string, container=np.array, datatype=u'h',
                is_big_endian=False):
        return self._resource.query_binary_values(query_string, container=container, datatype=datatype,
                is_big_endian=is_big_endian)

    # IEEE Mandated SCPI commands
    def CLS(self):
        self._resource.write("*CLS") # Clear Status Command
    def ESE(self):
        return self._resource.query("*ESE?") # Standard Event Status Enable Query
    def ESR(self):
        return self._resource.write("*ESR?") # Standard Event Status Register Query
    def IDN(self):
        return self._resource.query("*IDN?") # Identification Query
    def OPC(self):
        return self._resource.query("*OPC?") # Operation Complete Command
    def RST(self):
        self._resource.write("*RST") # Reset Command
    def SRE(self):
        return self._resource.query("*SRE?") # Service Request Enable Query
    def STB(self):
        return self._resource.query("*STB?") # Read Status Byte Query
    def TST(self):
        return self._resource.query("*TST?") # Self-Test Query
    def WAI(self):
        self._resource.write("*WAI") # Wait-to-Continue Command

def add_command(instr, name, cmd):
    """Helper function for parsing Instrument attributes and turning them into
    setters and getters."""
    def fget(self, **kwargs):
        val = self.interface.query( cmd.get_string.format( **kwargs ) )
        time.sleep(cmd.get_delay)
        return cmd.convert_get(val)

    def fset(self, val, **kwargs):
        if cmd.value_range is not None:
            if (val < cmd.value_range[0]) or (val > cmd.value_range[1]):
                err_msg = "The value {} is outside of the allowable range {} specified for instrument '{}'.".format(val, cmd.value_range, self.name)
                raise ValueError(err_msg)

        if cmd.allowed_values is not None:
            if not val in cmd.allowed_values:
                err_msg = "The value {} is not in the allowable set of values specified for instrument '{}': {}".format(val, self.name, cmd.allowed_values)
                raise ValueError(err_msg)

        if isinstance(cmd, RampCommand):
            # Ramp from one value to another, making sure we actually take some steps
            start_value = float(self.interface.query(cmd.get_string))
            approx_steps = int(abs(val-start_value)/cmd.increment)
            if approx_steps == 0:
                values = [val]
            else:
                values = np.linspace(start_value, val, approx_steps+2)
            for v in values:
                self.interface.write(cmd.set_string.format(v))
                time.sleep(cmd.pause)
        else:
            # Go straight to the desired value
            set_value = cmd.convert_set(val)
            # logger.debug("Formatting '%s' with string '%s'" % (cmd.set_string, set_value))
            # logger.debug("The result of the formatting is %s" % cmd.set_string.format(set_value, **{k: str(v) for k,v in kwargs.items()}))
            self.interface.write(cmd.set_string.format(set_value, **kwargs))
            time.sleep(cmd.set_delay)
    # Add getter and setter methods for passing around
    if cmd.additional_args is None:
        # We add properties in this case since not additional arguments are required
        # Using None prevents deletion or setting/getting unsettable/gettable attributes
        setattr(instr, name, property(fget if cmd.get_string else None, fset if cmd.set_string else None, None, cmd.doc))

    # In this case we can't create a property given additional arguments
    if cmd.get_string:
        setattr(instr, "get_" + name, fget)

    if cmd.set_string:
        setattr(instr, "set_" + name, fset)

class MetaInstrument(type):
    """Meta class to create instrument classes with controls turned into descriptors.
    """
    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding controls to %s", name)
        for k,v in dct.items():
            if isinstance(v, StringCommand):
                logger.debug("Adding '%s' command", k)
                add_command(self, k, v)
                if v.aliases is not None:
                    for a in v.aliases:
                        logger.debug("------> Adding alias '%s'" % a)
                        add_command(self, a, v)

class Instrument(metaclass=MetaInstrument):
    """This provides all of the actual device functionality, and contains the interface class
    that allows for communication for the physial instrument. When subclassing Instrument, calling
    the __init__ method of this base class will parse the class attributes and convert any Command
    objects such as to provide convenient get_xx and set_xx setter/getter methods as well
    as python @properties therof."""

    __isfrozen = False

    def __init__(self, resource_name, name=None, interface_type=None, check_errors_on_get=False, check_errors_on_set=False):
        super(Instrument, self).__init__()
        self.name = name
        self.resource_name = resource_name

        self.check_errors_on_get = check_errors_on_get
        self.check_errors_on_set = check_errors_on_set

        if interface_type is None:
            # Load the dummy interface, unless we see that GPIB is in the resource string
            if any([x in resource_name for x in ["GPIB", "USB", "SOCKET", "hislip", "inst0"]]):
                interface_type = "VISA"

        if interface_type is None:
            self.interface = Interface()
        elif interface_type == "VISA":
            if "SOCKET" in resource_name or "hislip" in resource_name or "inst0" in resource_name:
                ## assume single NIC for now
                resource_name = "TCPIP0::" + resource_name
            self.interface = VisaInterface(resource_name)
        else:
            raise ValueError("That interface type is not yet recognized.")

        self._freeze()

    def set_all(self, settings_dict):
        """Accept a settings dictionary and attempt to set all of the instrument
        parameters using the key/value pairs."""
        for name, value in settings_dict.items():
            if hasattr(self, name):
                setattr(self, name, value)

    # We want to lock the class dictionary
    # This solution from http://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "{} has a frozen class. Cannot access attribute {}".format(self, key) )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def _unfreeze(self):
        self.__isfrozen = False

    def __del__(self):
        #close the VISA resource
        if hasattr(self.interface, "_resource"):
            self.interface._resource.close()

    def __repr__(self):
        name = "Mystery Instrument" if self.name == "" else self.name
        return "{} @ {}".format(name, self.resource_name)

    def check_errors(self):
        pass
