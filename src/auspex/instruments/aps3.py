# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from .instrument import Instrument, MetaInstrument
from auspex.log import logger
import auspex.config as config
from time import sleep
import numpy as np
import socket
import collections
from struct import pack, iter_unpack

class AMC599(object):
    """Base class for simple register manipulations of AMC599 board.
    """

    PORT = 0xbb4e # TCPIP port (BBN!)

    def __init__(self):
        self.connected = False

    def __del__(self):
        self.disconnect()

    def _check_connected(self):
        if not self.connected:
            raise IOError("AMC599 Board not connected!")

    def connect(self, ip_addr="192.168.2.200"):
        self.ip_addr = ip_addr
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.socket.connect((self.ip_addr, self.PORT))
        self.connected = True

    def disconnect(self):
        if self.connected:
            self.connected = False
            self.socket.close()

    def send_bytes(self, data):
        if isinstance(data, collections.Iterable):
            return self.socket.sendall(b''.join([pack("!I", _) for _ in data]))
        else:
            return self.socket.sendall(pack("!I", data))

    def recv_bytes(self, size):
        resp = self.socket.recv(size)
        while True:
            if len(ans) >= size:
                break
            ans += self.socket.recv(8)
        data = [x[0] for x in iter_unpack("!I", ans)]
        return data

    def write_memory(self, addr, data):
        self._check_connected()
        max_ct = 0xfffc #max writeable block length (TODO: check if still true)
        cmd = 0x80000000 #write to RAM command
        datagrams_written = 0
        init_addr = addr
        idx = 0

        if isinstance(data, int):
            data = [data]
        elif isinstance(data, list):
            if not all([isinstance(v, int) for v in data]):
                raise ValueError("Data must be a list of integers.")
        else:
            raise ValueError("Data must be an integer or a list of integers.")

        while (len(data) - idx > 0):
            ct_left = len(data) - idx
            ct = ct_left if (ct_left < max_ct) else max_ct
            datagram = [cmd + ct, addr]
            datagram.extend(data[idx:idx+ct])
            self.send_bytes(datagram)
            datagrams_written += 1
            idx += ct
            addr += ct*4
        #read back data and check amount of bytes written makes sense
        #the ethernet core echoes back what we wrote
        resp = self.recv_bytes(2 * 4 * datagrams_written)
        addr = init_addr
        for ct in range(datagrams_written):
            if ct+1 == datagrams_written:
                words_written = len(data) - ((datagrams_written-1) * max_ct)
            else:
                words_written = max_ct
            logger.debug("Wrote {} words in {} datagrams: {}", words_written,
                                datagrams_written,
                                [hex(x) for x in resp])
            assert (results[2*ct] == 0x80800000 + words_written)
            assert (results[2*ct+1] == addr)
            addr += 4 * words_written

    def read_memory(self, addr, num_words):
        self._check_connected()
        datagram = [0x10000000 + num_words, addr]
        self.send_bytes(datagram)
        resp_header = self.recv_bytes(2 * 4) #4 bytes per word...
        return self.recv_bytes(4 * num_words)

    def read_memory_hex(self, addr, num_words):
        data = self.read_memory(addr, num_words)
        for d in data:
            print(hex(d))

#TODO: UPDATE!
CSR_AXI_ADDR = 0x44b400000
CSR_CONTROL_OFFSET = 0x00
GPIO1_OFFSET = 0x01C
GPIO2_OFFSET = 0x020
CSR_TRIGGER_INTERVAL_OFFSET = 0x14
SDRAM_AXI_ADDR = 0x80000000
