# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['MLBF']

from .instrument import Instrument, is_valid_ipv4
from auspex.log import logger
import numpy as np
import socket

class UDPInstrument(Instrument):
    def __init__(self,resource_name=None,name="UDP Instrument",instrument_type="UDP"):
        """ Control for UDP Instruments

        resource_name: Network IP address and port, e.g. 100.100.10.10:30303
        """
        self.resource_name = resource_name
        self.name = name
        self.instrument_type = instrument_type
        self.interface_type = "UDP"
        self._sock = None

    def connect(self,resource_name=None):
        """ Connect via UDP socket

        resource_name: Network IP address and port, e.g. 100.100.10.10:30303
        """
        if resource_name is not None:
            self.resource_name = resource_name
        if self.resource_name is None:
            logger.error("Failed setting up connection to %s. IP Address and Port are not provided." %self.name)
            return False
        try:
            logger.debug("UDP connect to %s at %s" %(self.name,self.resource_name))
            if is_valid_ipv4(self.resource_name[:self.resource_name.index(':')]):
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.connect((self.resource_name[:self.resource_name.index(':')],int(self.resource_name[self.resource_name.index(':')+1:])))
            else:
                logger.error("Invalid IP address %s" %self.resource_name[:resource_name.index(':')])

        except Exception as ex:
            self._sock = None
            logger.error("Failed setting up connection to %s. Exception: %s" %(self.resource_name,ex))
            return False

    def disconnect(self):
        if self._sock is not None:
            self._sock.close()
            logger.info("Disconnected %s from %s" %(self.name,self.resource_name))
        else:
            logger.warning("No connection is established. Do nothing.")

#    def query(self,command):
#        return self._sock.send(str(command).encode("ascii"))

    def write(self,command):
        return self._sock.send(str(command).encode("ascii"))

    def close(self):
        self.disconnect()

class MLBF(UDPInstrument):
    """SCPI instrument driver for Micro Lambda Programmable YIG filter.

    Properties:
        frequency: Set Center Frequency of the Notch Filter, 4-15 GHz.
    """
    instrument_type = "Microwave Filter"

    def __init__(self, resource_name=None, *args, **kwargs):
        """MLB YIG Filter"""
        super(MLBF, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None):
        """Connect to the Filter via a specified physical interface."""
        super(MLBF, self).connect(resource_name)

    @property
    def frequency(self):
        return

    @frequency.setter
    def frequency(self, f):
        """Set Frequency in MHz"""
        self.write("F%f"%f)
