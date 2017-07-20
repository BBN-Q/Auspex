# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['PrologixSocketResource']

import os
import numpy as np
import socket
import functools
from auspex.log import logger
from pyvisa.util import _converters, from_ascii_block, to_ascii_block, to_ieee_block, from_binary_block

class PrologixError(Exception):
    """Error interacting with the Prologix GPIB-ETHERNET controller."""

class PrologixSocketResource(object):
    """A resource representing a GPIB instrument controlled through a PrologixError
    GPIB-ETHERNET controller. Mimics the functionality of a pyVISA resource object.

    See http://prologix.biz/gpib-ethernet-controller.html for more details
    and a utility that will discover all prologix instruments on the network.

    Attributes:
        timeout: Timeout duration for TCP comms. Default 5s.
        write_termination: Character added to each outgoing message.
        read_termination: Character which denotes the end of a reponse message.
        idn_string: GPIB identification command. Defaults to '*IDN?'
        bufsize: Maximum amount of data to be received in one call, in bytes.
    """
    def __init__(self, ipaddr=None, gpib=None):
        super(PrologixSocketResource, self).__init__()
        if ipaddr is not None:
            self.ipaddr = ipaddr
        if gpib is not None:
            self.gpib = gpib
        self.sock = None
        self._timeout = 5
        self.read_termination = "\r\n"
        self.write_termination = "\r\n"
        self.idn_string = "*IDN?"
        self.bufsize = 4096

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = timeout
        if self.sock is not None:
            self.sock.settimeout(self._timeout)

    def connect(self, ipaddr=None, gpib=None):
        """Connect to a GPIB device through a Prologix GPIB-ETHERNET controller.
        box.

        Args:
            ipaddr: The IP address of the Prologix GPIB-ETHERNET.
            gpib: The GPIB address of the instrument to be controlled.
        Returns:
            None.
        """
        if ipaddr is not None:
            self.ipaddr = ipaddr
        if gpib is not None:
            self.gpib = gpib
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM,
                socket.IPPROTO_TCP)
            self.sock.settimeout(self._timeout)
            self.sock.connect((self.ipaddr, 1234)) #Prologix communicates on port 1234
        except socket.error as err:
            logger.error("Cannot open socket to Prologix at {0}: {1}".format(self.ipaddr,
                err.msg))
            raise PrologixError(self.ipaddr) from err
        self.sock.send(b"++ver\r\n")
        whoami = self.sock.recv(128).decode()
        if "Prologix" not in whoami:
            logger.error("The device at {0} does not appear to be a Prologix; got {1}.".format(self.ipaddr, whoami))
            raise PrologixError(whoami)
        self.sock.send(b"++mode 1\r\n") #set to controller mode
        self.sock.send(b"++auto 1\r\n") #enable read-after-write
        self._addr()
        self.sock.send(b"++clr\r\n")
        idn = self.query(self.idn_string)
        if idn is '':
            logger.error(("Did not receive response to GPIB command {0} " +
                "from GPIB device {1} on Prologix at {2}.").format(self.idn_string,
                self.gpib, self.ipaddr))
            raise PrologixError(idn)
        else:
            logger.debug(("Succesfully connected to device {0} at GPIB port {1} on" +
                " Prologix controller at {2}.").format(idn, self.gpib, self.ipaddr))

    def close(self):
        """Close the connection to the Prologix."""
        if self.sock is not None:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()

    def _addr(self):
        """Set PROLOGIX to address of instrument we want to control."""
        self.sock.send(('++addr %d\n' % self.gpib).encode())

    def read(self):
        """Read an ASCII value from the instrument.

        Args:
            None.
        Returns:
            The instrument data with termination character stripped.
        """
        self._addr()
        ans = self.sock.recv(self.bufsize).decode()
        return ans.rstrip(self.read_termination)

    def query(self, command):
        """Query instrument with ASCII command then read response.

        Args:
            command: Message to be sent to instrument.
        Returns:
            The instrument data with termination character stripped.
        """
        self._addr()
        self.sock.send((command + self.write_termination).encode())
        ans = self.sock.recv(self.bufsize).decode()
        return ans.rstrip(self.read_termination)

    def write(self, command):
        """Write a string message to device in ASCII format.

        Args:
            command: The message to be sent.
        Returns:
            The number of bytes in the message.
        """
        self._addr()
        self.sock.send((command + self.write_termination).encode())
        return len(command)

    def read_raw(self, bufsize=None):
        """Read bytes from instrument.

        Args:
            bufsize: Number of bytes to read from instrument. Defaults to resource
            bufsize if None.
        Returns:
            Instrument data. Nothing is stripped from response.
        """
        if bufsize is None:
            bufsize = self.bufsize
        self._addr()
        return self.socket.recv(bufsize)

    def write_raw(self, command):
        """Write a string message to device as raw bytes. No termination
        character is appended.

        Args:
            command: The message to be sent.
        Returns:
            The number of bytes in the message.
        """
        self._addr()
        self.sock.send(command)
        return len(command)

    def write_ascii_values(self, command, values, converter='f', separator=','):
        """Write a string message to device followed by values in ASCII format.

        Args:
            command: Message to be sent to device.
            values: Data to be written to device (as an iterable)
            converter: String format code to be used to convert values.
            separator: Separator between values -- separator.join(data).

        Returns:
            Total number of bytes sent to instrument.
        """
        ascii_vals = to_ascii_block(values, converter, separator)
        return self.write(command + ascii_vals + self.write_termination)

    def query_ascii_values(self, command, converter='f', separator=',',
        container=list, bufsize=None):
        """Write a string message to device and return values as iterable.

        Args:
            command: Message to be sent to device.
            values: Data to be written to device (as an interable)
            converter: String format code to be used to convert values.
            separator: Separator between values -- data.split(separator).
            container: Iterable type to use for output.
            bufsize: Number of bytes to read from instrument. Defaults to resource
            bufsize if None.
        Returns:
            Iterable of values converted from instrument response.
        """
        if bufsize is None:
            bufsize = self.bufsize
        ascii = self.query(command, bufsize=bufsize)
        return from_ascii_block(ascii, convereter, separator, container)

    def write_binary_values(self, command, values, datatype='f',
        is_big_endian=False):
        """Write a string message to device followed by values in binary IEEE
        format using a pyvisa utility function.)

        Args:
            command: String command sent to instrument.
            values: Data to be sent to instrument.
            datatype: Format string for single element.
            is_big_endian: Bool indicating endianness.
        Returns:
            Number of bytes written to instrument.
        """
        data = to_ieee_block(values, datatype=datatype, is_big_endian=is_big_endian)
        return self.write_raw(command.encode()+data)

    def query_binary_values(self, command, datatype='f', container=np.array,
        is_big_endian=False, bufsize=None):
        """Write a string message to device and read binary values, which are
        returned as iterable. Uses a pyvisa utility function.

        Args:
            command: String command sent to instrument.
            values: Data to be sent to instrument.
            datatype: Format string for single element.
            container: Iterable to return number of as.
            is_big_endian: Bool indicating endianness.

        Returns:
            Iterable of data values to be retuned
        """
        if bufsize is None:
            bufsize = self.bufsize
        self.write(command, bufsize=bufsize)
        block = self.read_raw()
        return from_binary_block(block, datatype=datatype,
            is_big_endian=is_big_endian, container=container)
