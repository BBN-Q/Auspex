# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from .instrument import Instrument, MetaInstrument, is_valid_ipv4
from .bbn import MakeSettersGetters
from auspex.log import logger
import auspex.config as config
from time import sleep
import numpy as np
import socket
import collections
from struct import pack, iter_unpack

U32 = 0xFFFFFFFF #mask for 32-bit unsigned int
U16 = 0xFFFF

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
        return data if len(data)>1 else data[0]

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

        data = [x & U32 for x in data] #make sure everything is 32 bits

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

CSR_CORR_OFFSET             = 0x0024
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

#####################################################################

def check_bits(value, shift, mask=0b1):
    return (((value & U32) >> shift) & mask)

def set_bits(value, shift, x, mask=0b1):
    return ((value & U32) & ~(mask << shift)) | ((x & U32) << shift)

class APS3(Instrument, metaclass=MakeSettersGetters):

    def __init__(self, resource_name=None, name="Unlabled APS3"):
        self.name = name
        self.resource_name = resource_name

        self.board = AMC599()

    def connect(self, resource_name=None):
        if resource_name is None and self.resource_name is None:
            raise ValueError("Must supply a resource name!")
        elif resource_name is not None:
            self.resource_name = resource_name
        if is_valid_ipv4(self.resource_name):
            self.board.connect(ip_addr = self.resource_name)

    def disconnect(self):
        if self.board.connected:
            self.board.disconnect()

    def write_csr(self, offset, data):
        self.board.write_memory(CSR_AXI_ADDR_BASE + offset, data)

    def read_csr(self, offset, num_words = 1):
        return self.board.read_memory(CSR_AXI_ADDR_BASE + offset, num_words)

    def write_dram(self, offset, data):
        self.board.write_memory(DRAM_AXI_BASE + offset, data)

    def read_dram(self, offset, num_words = 1):
        return self.board.read_memory(DRAM_AXI_BASE + offset, num_words)

    ####### CACHE CONTROL REGSITER #############################################

    @property
    def cache_controller(self):
        """Get value of cache controller
            False: Cache controller in reset.
            True: Cache controller taken out of reset.
        """
        return bool(self.read_csr(CSR_CACHE_CONTROL) & 0x1)
    @cache_controller.setter
    def cache_controller(self, value):
        self.write_csr(CSR_CACHE_CONTROL, int(value))

    ####### WAVEFORM OFFSET REGISTER ###########################################

    @property
    def waveform_offsets(self):
        """Get waveform A and B offset register values. These are used as
        the DMA source address.
        """
        return [self.read_csr(CSR_WFA_OFFSET),
                self.read_csr(CSR_WFB_OFFSET)]
    @property
    def waveform_offsets(self, offsets):
        """Set waveform A and B offsets, passed as list [A offset, B offset].
        Set one offset to None to not change its value.
        """
        if offsets[0] is not None:
            self.write_csr(CSR_WFA_OFFSET, offsets[0])
        if offsets[1] is not None:
            self.write_csr(CSR_WFB_OFFSET, offsets[1])

    ####### SEQUENCER CONTROL REGISTER #########################################

    @property
    def sequencer_reset(self):
        """Sequencer reset:
            False: Sequencer, trigger input, modulator, SATA, VRAMs, Debug Streams
                disabled.
            True: Sequencer logic taken out of reset.
        """
        reg = self.read_csr(CSR_SEQ_CONTROL)
        return bool(check_bits(csr, 0))
    @sequencer_reset.setter
    def sequencer_reset(self, reset):
        reg = self.read_csr(CSR_SEQ_CONTROL)
        self.write_csr(CSR_SEQ_CONTROL, set_bits(reg, 0, int(reset)))

    @property
    def trigger_source(self):
        trig_val = check_bits(self.read_csr(CSR_SEQ_CONTROL), 1, 0b11)
        trigger_map = {0b00: "EXTERNAL", 0b01: "INTERNAL",
                        0b10: "SOFTWARE", 0b11 "MESSAGE"}
        return trigger_map[trig_val]
    @trigger_source.setter
    def trigger_source(self, value):
        trig_map = {"EXTERNAL": 0b00, "INTERNAL": 0b01,
                    "SOFTWARE": 0b10, "MESSAGE": 0b11}
        if value.upper() not in trig_map.keys():
            raise ValueError(f"Unknown trigger mode. Must be one of {trig_map.keys()}")
        if value.upper == "MESSAGE":
            raise NotImplementedError("APS3 does not yet support message triggers.")
        reg = self.read_csr(CSR_SEQ_CONTROL)
        self.write_csr(CSR_SEQ_CONTROL, set_bits(reg, 1, trig_map[value.upper()], 0b11))

    def soft_trigger(self):
        reg = self.read_csr(CSR_SEQ_CONTROL)
        self.write_csr(CSR_SEQ_CONTROL, set_bits(reg, 3, 0b1))

    @property
    def trigger_enable(self):
        return bool(check_bits(self.read_csr(CSR_SEQ_CONTROL), 4))
    @trigger_enable.setter
    def trigger_enable(self, value):
        reg = self.read_csr(CSR_SEQ_CONTROL)
        self.write_csr(CSR_SEQ_CONTROL, set_bits(reg, 4, 0b1))

    @property
    def bypass_modulator(self):
        return bool(check_bits(self.read_csr(CSR_SEQ_CONTROL), 5))
    @bypass_modulator.setter
    def bypass_modulator(self, value):
        reg = self.read_csr(CSR_SEQ_CONTROL)
        self.write_csr(CSR_SEQ_CONTROL, set_bits(reg, 5, 0b1))

    @property
    def bypass_nco(self):
        return bool(check_bits(self.read_csr(CSR_SEQ_CONTROL), 6))
    @bypass_nco.setter
    def bypass_nco(self, value):
        reg = self.read_csr(CSR_SEQ_CONTROL)
        self.write_csr(CSR_SEQ_CONTROL, set_bits(reg, 6, 0b1))

    ####### CORRECTION CONTROL REGISTER ########################################
    @property
    def correction_control(self):
        reg = self.read_csr(CSR_CORR_OFFSET)
        return ((reg >> 16) & U16, reg & U16) #returns (I, Q)
    @correction_control.setter
    def correction_control(self, value):
        packed_value = ((value[0] & U16) << 16) | (value[1] & U16)
        self.write_csr(CSR_CORR_OFFSET, packed_value)

    ####### TRIGGER INTERVAL ###################################################
    @property
    def trigger_interval(self):
        return self.read_csr(CSR_TRIG_INTERVAL)
    @trigger_interval.setter
    def trigger_interval(self, value)
        return self.write_csr(CSR_TRIG_INTERVAL, value & U32)

    ####### UPTIME REGISTERS ###################################################
    @property
    def uptime_seconds(self):
        return self.read_csr(CSR_UPTIME_SEC)

    @property
    def uptime_nanoseconds(self):
        return self.read_csr(CSR_UPTIME_NS)

    ####### BUILD INFO REGISTERS ###############################################
    def get_firmware_info(self):
        fpga_rev = self.read_csr(CSR_FPGA_REV)
        fpga_id = self.read_csr(CSR_FPGA_ID)
        git_sha1 = self.read_csr(CSR_GIT_SHA1)
        build_tstamp = self.read_csr(CSR_BUILD_TSTAMP)

        fpga_build_minor = check_bits(fpga_rev, 0, 0xFF)
        fpga_build_major = check_bits(fpga_rev, 8, 0xFF)
        commit_history = check_bits(fpga_rev, 16, 0x7FF)
        build_clean = "dirty" if check_bits(fpga_rev, 27, 0xF) else "clean"

        return {"FPGA ID": fpga_id,
                "FPGA REV": f"{fpga_build_major}.{fpga_build_minor} - {commit_history} - {build_clean}",
                "GIT SHA1": hex(git_sha1),
                "DATE": hex(build_tstamp)[2:]}

    ####### CORRECTION MATRIX ##################################################
    def get_correction_matrix(self):
        row0 = self.read_csr(CSR_CMAT_R0)
        row1 = self.read_csr(CSR_CMAT_R1)

        r00 = (row0 >> 16) & U16
        r01 = row0 & U16
        r10 = (row1 >> 16) & U16
        r11 = row0 & U16
        return np.array([[r00, r01], [r10, r11]], dtype=np.uint16)

    def set_correction_matrix(self, matrix):
        row0 = ((matix[0, 0] & U16) << 16) | (matrix[0, 1] & U16)
        row1 = ((matix[1, 0] & U16) << 16) | (matrix[1, 1] & U16)
        self.write_csr(CSR_CMAT_R0, row0)
        self.write_csr(CSR_CMAT_R1, row1)

    ####### BOARD_CONTROL ######################################################
    @property
    def microblaze(self):
        return bool(check_bits(self.read_csr(CSR_BD_CONTROL), 1))
    @microbalze.setter(self, value)
        reg = self.read_csr(CSR_BD_CONTROL)
        self.write_csr(CSR_BD_CONTROL, set_bits(reg, 1, int(value)))
