# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
import socket
import functools
from .instrument import is_valid_ipv4
from .interface import Interface
from auspex.log import logger

_converters = {
    's': str,
    'b': functools.partial(int, base=2),
    'c': chr,
    'd': int,
    'o': functools.partial(int, base=8),
    'x': functools.partial(int, base=16),
    'X': functools.partial(int, base=16),
    'e': float,
    'E': float,
    'f': float,
    'F': float,
    'g': float,
    'G': float,
}

class PrologixError(Exception):
    """Error interacting with the Prologix GPIB-ETHERNET controller."""

class PrologixSocketResource(object):
    """A resource representing a GPIB instrument controlled through a PrologixError
    GPIB-ETHERNET controller. Mimics the functionality of a pyVISA resource object.

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
        self.read_termination = u'\n'
        self.write_termination = u'\n'
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
            logger.error("Cannot open socket to Prologix at {}: {}".format(self.ipaddr,
                err.message))
            raise PrologixError(self.ipaddr) from err
        whoami = self.query('++ver')
        if "Prologix" not in whoami:
            logger.error("The device at {} does not appear to be a Prologix; got {}.".format(whoami))
            raise PrologixError(whoami)
        self.write("++mode 1") #set to controller mode
        self.write("++auto 1") #enable read-after-write

        idn = self.query(self.idn_string)
        if idn is '':
            logger.error("Did not receive response to GPIB command {} " +
                "from GPIB device {} on Prologix at {}.".format(self.idn_string,
                self.gpib, self.ipaddr))
            raise PrologixError(idn)
        else:
            logger.info("Succesfully connected to device {} at GPIB port {} on" +
                " Prologix controller at {}.".format(idn, self.gpib, self.ipaddr))

    def close(self):
        """Close the connection to the Prologix."""
        if self.sock is not None:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()

    def _addr(self):
        """Set PROLOGIX to address of instrument we want to control."""
        self.sock.send(('++addr %d\n' % self.gpib).encode())

    def _strip(self, text):
        """Strip read termination character from a string."""
        if not text.endswith(self.read_termination):
            return text
        return text[:len(text)-len(self.read_termination)]

    def read(self):
        """Read an ASCII value from the instrument.

        Args:
            None.
        Returns:
            The instrument data with termination character stripped.
        """
        self._addr()
        ans = self.sock.recv(self.bufsize)
        return self._strip(ans.decode())

    def query(self):
        """Query instrument with ASCII command then read response.

        Args:
            command: Message to be sent to instrument.
        Returns:
            The instrument data with termination character stripped.
        """
        self._addr()
        self.sock.send((command + self.write_termination).encode())
        ans = self.sock.recv(self.bufsize)
        return self._strip(ans.decode())

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
        converter = '%' + converter
        ascii_vals = separator.join(converter % v for v in values)
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
        try:
            converter = _converters[converter]
        except KeyError as err:
            raise ValueError("Invalid string converter: {0}".format(converter)) from err

        ascii = self.query(command, bufsize=bufsize)
        return container([converter(v) for v in ascii.split(separator)])

    def write_binary_values(self, command, values, datatype='f',
        is_big_endian=False):
        """Write a string message to device followed by values in binary IEEE
        format, same as equivalent pyvisa function (which this code is more or
        less copied from.)

        Args:
            command: String command sent to instrument.
            values: Data to be sent to instrument.
            datatype: Format string for single element.
            is_big_endian: Bool indicating endianness.
        Returns:
            Number of bytes written to instrument.
        """
        array_len = len(values)
        elem_len = struct.calcsize(datatype)
        length = array_len * elem_len
        header = '%d' % length
        header = '#%d%s'%(len(header),header)
        endian = '>' if is_big_endian else '<'
        fullfmt = '%s%d%s'(endian, array_len, datatype)
        data = bytes(header, 'ascii') + struct.pack(fullft, *values)
        return self.write_raw(command.encode()+data)

    def query_binary_values(self, command, datatype=u'h', container=array
        is_big_endian=False, delay=0.1):
        """Write a string message to device and read binary values, which are
        returned as iterable. Again pilfered from pyvisa.

        Args:
            command: String command sent to instrument.
            values: Data to be sent to instrument.
            datatype: Format string for single element.
            container: Iterable to return number of as.
            is_big_endian: Bool indicating endianness.

        Returns:
            Iterable of data values to be retuned
        """
        self.write(command)
        block = self.read_raw()

        #parse IEEE data block
        begin = block.find(b'#')
        if begin < 0:
            raise ValueError("Cound not find (#) in data received from "+
                "instrument indicating start of IEEE block.")
        try:
            header_len = int(block[begin+1:begin+2])
        except ValueError:
            header_len = 0
        offset = begin + 2 + header_len
        if header_len > 0:
            data_len = int(block[begin+2:offset])
        else:
            data_len = len(block) - offset - 1
        expected_len = data+len + offset
        while len(block) < expected_len:
            block += self.read_raw()
        elem_len = struct.calcsize(datatype)
        array_len = int(data_len / elem_len)
        endian = '>' if is_big_endian else '<'
        fullfmt = '%s%d%s'(endian, array_len, datatype)
        try:
            return container(struct.unpack_from(fullfmt, block, offset))
        except struct.error:
            raise ValueError("Could not unpack binary data from instrument.")
