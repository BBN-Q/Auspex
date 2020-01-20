__all__ = ['Instrument']

import numpy as np
import os
import time
import socket
from unittest.mock import MagicMock

from auspex.log import logger
from .interface import Interface, VisaInterface, PrologixInterface

#Helper function to check for IPv4 address
#See http://stackoverflow.com/a/11264379
def is_valid_ipv4(ipv4_address):
    try:
        socket.inet_aton(ipv4_address)
        if ipv4_address.count(".") != 3:
            logger.warning("User-provided IP {} is a valid IP address but does" +
                " not appear to be in human-readable format.".format(ipv4_address))
            return False
        return True
    except socket.error:
        return False
    except:
        raise

class Command(object):
    """Store the arguments and keywords, so that we may later dispatch
    depending on the instrument type."""
    formatter = '{}'

    def __init__(self, *args, **kwargs):
        self.args   = args
        self.kwargs = kwargs

    def parse(self):
        for a in ['aliases', 'set_delay', 'get_delay',
                  'value_map', 'value_range',
                  'allowed_values']:
            if a in self.kwargs:
                setattr(self, a, self.kwargs.pop(a))
            else:
                setattr(self, a, None) # Default to None

        if 'doc' in self.kwargs:
            self.doc = self.kwargs.pop('doc')
        else:
            self.doc = ""

        if self.value_range is not None:
            self.value_range = (min(self.value_range), max(self.value_range))

        self.python_to_instr = None # Dict StringCommandmapping from python values to instrument values
        self.instr_to_python = None # Dict mapping from instrument values to python values

        if self.value_map is not None:
            self.python_to_instr = self.value_map
            self.instr_to_python = {v: k for k, v in self.value_map.items()}

            if self.value_range is not None:
                raise Exception("Cannot specify both value_range and value_map as they are redundant.")

            if self.allowed_values is not None:
                raise Exception("Cannot specify both value_map and allowed_values as they are redundant.")
            else:
                self.allowed_values=list(self.python_to_instr.keys())

            logger.debug("Constructed map and inverse map for command values:\n--- %s\n--- %s'" % (self.python_to_instr, self.instr_to_python))

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

class SCPICommand(Command):
    def parse(self):
        super(SCPICommand, self).parse()

        for a in ['scpi_string', 'get_string', 'set_string', 'additional_args']:
            if a in self.kwargs:
                setattr(self, a, self.kwargs.pop(a))
            else:
                setattr(self, a, None) # Default to None

        # SCPI specific additions
        if self.scpi_string is not None:
            # Construct get and set strings using this base scpi_string
            self.get_string = self.scpi_string+"?;"
            self.set_string = self.scpi_string+" "+self.formatter

        # We need to do something or other
        if self.set_string is None and self.get_string is None:
            raise ValueError("Neither a setter nor a getter was specified.")

class StringCommand(Command):
    formatter = '{:s}'
    def convert_get(self, get_value_instrument):
        """Convert the instrument's returned values to something conveniently accessed
        through python."""
        if self.python_to_instr is None:
            return str(get_value_instrument)
        else:
            return str(self.instr_to_python[get_value_instrument])

class BoolCommand(StringCommand):
    def convert_get(self, get_value_instrument):
        """Convert the instrument's returned values to something conveniently accessed
        through python."""
        if self.python_to_instr is None:
            return bool(get_value_instrument)
        else:
            return bool(self.instr_to_python[get_value_instrument])

class FloatCommand(Command):
    formatter = '{:E}'
    def convert_get(self, get_value_instrument):
        """Convert the instrument's returned values to something conveniently accessed
        through python."""
        if self.python_to_instr is None:
            return float(get_value_instrument)
        else:
            return float(self.instr_to_python[get_value_instrument])

class IntCommand(Command):
    formatter = '{:d}'
    def convert_get(self, get_value_instrument):
        """Convert the instrument's returned values to something conveniently accessed
        through python."""
        if self.python_to_instr is None:
            return int(float(get_value_instrument))
        else:
            return int(float(self.instr_to_python[get_value_instrument]))

class RampCommand(FloatCommand):
    """For quantities that are to be ramped from present to desired value. These will always be floats..."""
    def parse(self):
        super(RampCommand, self).parse()
        if 'increment' in self.kwargs:
            self.increment = self.kwargs.pop('increment')
        else:
            raise Exception("RampCommand requires a ramp increment")
        if 'pause' in self.kwargs:
            self.pause = self.kwargs.pop('pause')
        else:
            self.pause = 0.0

class SCPIStringCommand(SCPICommand, StringCommand): pass
class SCPIBoolCommand(SCPICommand, BoolCommand): pass
class SCPIFloatCommand(SCPICommand, FloatCommand): pass
class SCPIIntCommand(SCPICommand, IntCommand): pass
class SCPIRampCommand(SCPICommand, RampCommand): pass

class ReceiverChannel(object): pass

class MetaInstrument(type):
    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)

        # What sort of instrument are we?
        if len(bases) > 0 and bases[0].__name__ is not 'Instrument':
            instr_type = bases[0].__name__.replace('Instrument','')
            logger.debug("Adding Commands to %s", name)
            logger.debug("We are metaprogramming a %s instrument.", instr_type)

            for k,v in dct.items():
                if isinstance(v, Command):
                    logger.debug("Adding '%s' command", k)
                    if instr_type == "SCPI":
                        nv = add_command_SCPI(self, k, v)
                        if nv.aliases is not None:
                            for a in nv.aliases:
                                logger.debug("------> Adding alias '%s'" % a)
                                add_command_SCPI(self, a, v)
                    elif instr_type == "CLib":
                        nv = add_command_CLib(self, k, v)
                        if nv.aliases is not None:
                            for a in nv.aliases:
                                logger.debug("------> Adding alias '%s'" % a)
                                add_command_CLib(self, a, v)


class Instrument(metaclass=MetaInstrument):
    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name

    def disconnect(self):
        pass

    def configure_with_proxy(self, proxy):
        self.configure_with_dict(dict((col, getattr(proxy, col)) for col in proxy.__table__.columns.keys()))

    def configure_with_dict(self, settings_dict):
        """Accept a sdettings dictionary and attempt to set all of the instrument
        parameters using the key/value pairs."""
        for name, value in settings_dict.items():
            if name not in ["id", "label", "model", "address", "channel_db_id", "standalone"]:
                if "_id" in name:
                    continue
                # Python is insane, and attempts to run a property's getter
                # when queried by hasattr. Avoid this behavior with the
                # "ask for forgiveness" paradigm.
                try:
                    setattr(self, name, value)
                except (AttributeError, TypeError) as e:
                    logger.info("Instrument {} property: {} could not be set to {}.".format(self.name,name,value))
                    pass

class CLibInstrument(Instrument): pass

class SCPIInstrument(Instrument):

    __isfrozen = False

    def __init__(self, resource_name=None, name="Yet-to-be-named SCPI Instrument"):
        self.name            = name
        self.resource_name   = resource_name
        self.proxy_obj       = None
        if not hasattr(self, "instrument_type"):
            self.instrument_type = None # This can be AWG, Digitizer, etc.
        self.interface       = None
        self._freeze()

    def connect(self, resource_name=None, interface_type=None):
        """Either connect to the resource name specified during initialization, or specify
        a new resource name here."""

        self._unfreeze()
        if resource_name is None and self.resource_name is None:
            raise Exception("Must supply a resource name to 'connect' if the instrument was initialized without one.")
        elif resource_name is not None:
            self.resource_name = resource_name
        self.full_resource_name = self.resource_name

        if interface_type is None:
            # Load the dummy interface, unless we see that GPIB is in the resource string
            if any([x in self.resource_name for x in ["GPIB", "USB", "SOCKET", "hislip", "inst0", "COM", "ASRL", "TCPIP"]]):
                interface_type = "VISA"

        try:
            if interface_type is None:
                logger.debug("Instrument {} is using a generic instrument " +
                    "interface as none was provided.".format(self.name))
                self.interface = Interface()
            elif interface_type == "VISA":
                if "GPIB" in self.full_resource_name:
                    pass
                elif "USB" in self.full_resource_name:
                    pass
                elif any(is_valid_ipv4(substr) for substr in self.full_resource_name.split("::")) and "TCPIP" not in self.full_resource_name:
                    # assume single NIC for now
                    self.full_resource_name = "TCPIP0::" + self.full_resource_name
                self.interface = VisaInterface(self.full_resource_name)
                logger.debug("A pyVISA interface {} was created for instrument {}.".format(str(self.interface._resource), self.name))
            elif interface_type == "Prologix":
                self.interface = PrologixInterface(self.full_resource_name)

            else:
                raise ValueError("That interface type is not yet recognized.")
        except:
            logger.error("Could not initialize interface for %s.", self.full_resource_name)
            self.interface = MagicMock()
        self._freeze()

    def disconnect(self):
        self.interface.close()

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
        if hasattr(self, 'interface') and hasattr(self.interface, "_resource"):
            logger.debug("VISA Interface for {} @ {} closed.".format(self.name, self.resource_name))
            self.interface._resource.close()

    def __repr__(self):
        return "{} @ {}".format(self.name, self.resource_name)


def add_command_SCPI(instr, name, cmd):
    """Helper function for parsing Instrument attributes and turning them into
    setters and getters for SCPI style commands."""
    # Replace with the relevant SCPI command variant
    new_cmd = globals()['SCPI'+cmd.__class__.__name__](*cmd.args, **cmd.kwargs)
    new_cmd.parse()

    def fget(self, **kwargs):
        val = self.interface.query( new_cmd.get_string.format( **kwargs ) )
        if new_cmd.get_delay is not None:
            time.sleep(new_cmd.get_delay)
        return new_cmd.convert_get(val)

    def fset(self, val, **kwargs):
        if new_cmd.value_range is not None:
            if (val < new_cmd.value_range[0]) or (val > new_cmd.value_range[1]):
                err_msg = "The value {} is outside of the allowable range {} specified for instrument '{}'.".format(val, new_cmd.value_range, self.name)
                raise ValueError(err_msg)

        if new_cmd.allowed_values is not None:
            if not val in new_cmd.allowed_values:
                err_msg = "The value {} is not in the allowable set of values specified for instrument '{}': {}".format(val, self.name, new_cmd.allowed_values)
                raise ValueError(err_msg)

        if isinstance(cmd, RampCommand):
            if 'increment' in kwargs:
                new_cmd.increment = kwargs['increment']
            if 'pause' in kwargs:
                new_cmd.pause = kwargs['pause']
            # Ramp from one value to another, making sure we actually take some steps
            start_value = float(self.interface.query(new_cmd.get_string))
            approx_steps = int(abs(val-start_value)/new_cmd.increment)
            if approx_steps == 0:
                values = [val]
            else:
                values = np.linspace(start_value, val, approx_steps+2)
            for v in values:
                self.interface.write(new_cmd.set_string.format(v))
                time.sleep(new_cmd.pause)
        else:
            # Go straight to the desired value
            set_value = new_cmd.convert_set(val)
            self.interface.write(new_cmd.set_string.format(set_value, **kwargs))
            if new_cmd.set_delay is not None:
                time.sleep(new_cmd.set_delay)

    # Add getter and setter methods for passing around
    if new_cmd.additional_args is None:
        # We add properties in this case since not additional arguments are required
        # Using None prevents deletion or setting/getting unsettable/gettable attributes
        setattr(instr, name, property(fget if new_cmd.get_string else None, fset if new_cmd.set_string else None, None, new_cmd.doc))

    # In this case we can't create a property given additional arguments
    if new_cmd.get_string:
        setattr(instr, "get_" + name, fget)
        setattr(getattr(instr, "get_" + name), "__doc__", new_cmd.doc)

    if new_cmd.set_string:
        setattr(instr, "set_" + name, fset)
        setattr(getattr(instr, "set_" + name), "__doc__", new_cmd.doc)

    return new_cmd

def add_command_CLib(instr, name, cmd):
    """The hope is that this will eventually be used to automate writing instrument
    drivers that interface with C Libraries."""
    cmd.parse()
    return cmd
