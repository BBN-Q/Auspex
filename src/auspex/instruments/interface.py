import os
import visa
import numpy as np
from pycontrol.log import logger

class Interface(object):
    """Currently just a dummy interface for testing."""
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
    def close(self):
        pass

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
    def close(self):
        self._resource.close()

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
