# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['APS3', 'KCU105_Board']

from .instrument import Instrument, SCPIInstrument, VisaInterface, MetaInstrument
#from auspex.log import logger
#import auspex.config as config
from types import MethodType
from unittest.mock import MagicMock
from time import sleep
from visa import VisaIOError
import numpy as np
import socket, collections
from struct import pack, iter_unpack
from copy import deepcopy

class KCU105_Board(object):

    PORT = 0xbb4e #BBN, of course

    def __init__(self):
        pass

    def __del__(self):
        if self.connected:
            self.connected = False
            self.socket.shutdown()
            self.socket.close()

    def connect(self, resource_name="192.168.2.200"):
        self.ip_addr = resource_name
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.socket.connect((self.ip_addr, self.PORT))

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
        data = [x[0] for x in iter_unpack("!I", self.socket.recv(size))]
        if len(data) == 1:
            return data[0]
        else:
            return data

    def write_memory(self, addr, data):
        max_ct = 0xfffc
        cmd = 0x80000000
        datagrams_written = 0
        init_addr = addr
        idx = 0
        while(len(data) - idx > 0):
            ct_left = len(data) - idx
            ct = ct_left if (ct_left < max_ct) else max_ct
            datagram =  [cmd + ct, addr, data[idx:idx+ct]]
            self.send_bytes(datagram)
            datagrams_written += 1
            idx += ct
            addr += ct*4

        #read back and check
        results = self.recv_bytes(2 * 2 * datagrams_written)
        addr = init_addr
        for ct in range(datagrams_written):
            if ct == datagrams_written:
                words_written = len(data)-((datagrams_written-1)*max_ct)
            else:
                words_written = max_ct
            assert (results[2*ct-1] == 0x80800000 + words_written)
            assert (results[2*ct] == addr)
            addr += 4 * words_written

    def read_memory(self, addr, num_words):
        datagram = [0x10000000 + num_words, addr]
        self.send_bytes(datagram)
        resp_header = self.recv_bytes(2 * 2) #2 bytes per word
        print(hex(resp_header))
        return self.recv_bytes(2 * num_words) #2 bytes per word



class APS3(KCU105_Board):

    instrument_type = "AWG"

    CSR_AXI_ADDR = 0x44a00000
    CSR_CONTROL_OFFSET = 0x00
    CSR_TRIGGER_INTERVAL_OFFSET = 0x14
    SDRAM_AXI_ADDR = 0x80000000

    def __init__(self, resource_name=None, name="Unlabled APS"):
        self.name = name
        self.resource_name = resource_name

    def connect(self, resource_name=None):
        raise NotImplementedError

    def _initialize_AD9164_SPI(self):
        raise NotImplementedError

    def _initialize_AD9164_ACE(self):
        raise NotImplementedError
