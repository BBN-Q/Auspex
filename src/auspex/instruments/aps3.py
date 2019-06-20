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

    def write_memory(self, addr, data, offset = 0x0):
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

#####################################################################

#APS3 AMC599 Control and Status Register offsets
#Add to CSR_AXI_ADDR_BASE to get to correct memory location
#Registers are read/write unless otherwise noted
#Current as of 6/20/19

CSR_AXI_ADDR_BASE           = 0x44b4000

CSR_CACHE_CONTROL           = 0x0010 #Cache control register
CSR_SEQ_CONTROL             = 0x0024 #Sequencer control register

CSR_WFA_OFFSET              = 0x0014 #Waveform A Offset
CSR_WFB_OFFSET              = 0x0018 #Waveform B offset
CSR_SEQ_OFFSET              = 0x001C #Sequence data offset

CSR_TRIG_WORD               = 0x002C #Trigger word register, Read Only
CSR_TRIG_INTERVAL           = 0x0030 #trigger interval register

CSR_UPTIME_SEC              = 0x0050 #uptime in seconds, read only
CSR_UPTIME_NS               = 0x0054 #uptime in nanoseconds, read only
CSR_FPGA_REV                = 0x0058 #FPGA revision, read only
CSR_GIT_SHA1                = 0x0060 #git SHA1 hash, read only
CSR_BUILD_TSTAMP            = 0x0064 #build timestamp, read only

CSR_CMAT_R0                 = 0x0068 #correction matrix row 0
CSR_CMAT_R1                 = 0x006C #correction matrix row 1

#### NOT CONNECTED TO ANY LOGIC -- USE FOR VALUE STORAGE ############
CSR_A_AMPLITUDE             = 0x0070 #Channel A amplitude
CSR_B_AMPLITUDE             = 0x0074 #Channel B amplitude
CSR_MIX_AMP                 = 0x0078 #Mixer amplitude correction
CSR_MIX_PHASE               = 0x007C #Mixer phase skew correction
CSR_WFA_LEN                 = 0x0080 #channel A waveform length
CSR_WFB_LEN                 = 0x0084 #channel B waveform length
CSR_WF_MOD_FREQ             = 0x0088 #waveform modulation frequency
######################################################################

CSR_WFA_DELAY               = 0x008C #Channel A delay
CSR_WFB_DELAY               = 0x0080 #channel B delay

CSR_BD_CONTROL              = 0x00A0 #board control register
CSR_FPGA_ID                 = 0x00B4 #FPGA ID (read-only)

CSR_DATA1_IO                = 0x00B8 #Data 1 IO register
CSR_DATA2_IO                = 0x00BC #Data 2 IO register

CSR_MARKER_DELAY            = 0x00C0 #Marker delay

CSR_IPV4                    = 0x00C4 #IPv4 address register

#####################################################################

DRAM_AXI_BASE = 0x80000000
