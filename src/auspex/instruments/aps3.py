# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

#__all__ = ['AMC599', 'APS3']

from .instrument import Instrument, is_valid_ipv4, Command
from .bbn import MakeSettersGetters
from auspex.log import logger
import auspex.config as config
from time import sleep
import numpy as np
import socket
from unittest.mock import Mock
import collections
from struct import pack, iter_unpack

U32 = 0xFFFFFFFF #mask for 32-bit unsigned int
U16 = 0xFFFF

def check_bits(value, shift, mask=0b1):
    """Helper function to get a bit-slice of a 32-bit value.
        Returns (value >> shift) & mask."""
    return (((value & U32) >> shift) & mask)

def set_bits(value, shift, x, mask=0b1):
    """Set bits in a 32-bit value given an offset and a mask. """
    return ((value & U32) & ~(mask << shift)) | ((x & U32) << shift)

class BitFieldCommand(Command):
    """An instrument command that sets/gets a value from a register.
        See also the Command object in .instrument.py

        Args:
            None.
        Kwargs:
            register: Control register address. (required)
            shift: 0-indexed bit position. (required)
            mask: Size of bit field -- i.e. use 0b111 if setting a 3-bit value
                defaults to 0b1 or is inferred from a value map.
    """

    def parse(self):
        super().parse()

        for a in ('register', 'shift', 'mask', 'readonly'):
            if a in self.kwargs:
                setattr(self, a, self.kwargs.pop(a))
            else:
                setattr(self, a, None)

        if self.register is None:
            raise ValueError("Must specify a destination or source register.")
        if self.shift is None:
            raise ValueError("Must specify a bit shift for register command.")

        if self.readonly is None:
            self.readonly = False

        if self.mask is None:
            if self.value_map is not None:
                max_bits = max((v.bit_length() for v in self.value_map.values()))
                self.mask = 2**max_bits - 1
            else:
                self.mask = 0b1

    def convert_set(self, set_value_python):
        if self.python_to_instr is None:
            return int(set_value_python)
        else:
            return self.python_to_instr[set_value_python]

    def convert_get(self, get_value_instrument):
        if self.python_to_instr is None:
            if self.mask == 0b1:
                return bool(get_value_instrument)
            else:
                return get_value_instrument
        else:
            return self.instr_to_python[get_value_instrument]

def add_command_bitfield(instr, name, cmd):
    """Helper function to create a new BitFieldCommand when parsing an instrument."""

    new_cmd = BitFieldCommand(*cmd.args, **cmd.kwargs)
    new_cmd.parse()

    def fget(self, **kwargs):
        val = check_bits(self.read_register(new_cmd.register), new_cmd.shift, new_cmd.mask)
        if new_cmd.get_delay is not None:
            sleep(new_cmd.get_delay)
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

        start_val = self.read_register(new_cmd.register)
        new_val = set_bits(start_val, new_cmd.shift, new_cmd.convert_set(val), new_cmd.mask)
        self.write_register(new_cmd.register, new_val)
        if new_cmd.set_delay is not None:
            sleep(new_cmd.set_delay)

    setattr(instr, name, property(fget, None if new_cmd.readonly else fset, None, new_cmd.doc))
    setattr(instr, "set_"+name, fset)
    setattr(instr, "get_"+name, fget)

    return new_cmd

class MakeBitFieldParams(MakeSettersGetters):
    def __init__(self, name, bases, dct):
        super().__init__(name, bases, dct)

        if 'write_register' not in dct or 'read_register' not in dct:
            raise TypeError("An instrument using BitFieldParams must implement" +
                " `read_register` and `write_register` functions.")

        for k, v in dct.items():
            if isinstance(v, BitFieldCommand):
                logger.debug("Adding %s command", k)
                nv = add_command_bitfield(self, k, v)

class AMC599(object):
    """Base class for simple register manipulations of AMC599 board.
    """

    PORT = 0xbb4e # TCPIP port (BBN!)

    def __init__(self, debug=False):
        self.connected = False
        self.debug = debug
        if self.debug:
            self.debug_memory = {}

    def __del__(self):
        self.disconnect()

    def _check_connected(self):
        if not self.connected:
            raise IOError("AMC599 Board not connected!")

    def connect(self, ip_addr="192.168.2.200"):
        self.ip_addr = ip_addr
        if not self.debug:
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
            if len(resp) >= size:
                break
            resp += self.socket.recv(8)
        data = [x[0] for x in iter_unpack("!I", resp)]
        return data if len(data)>1 else data[0]

    def write_memory(self, addr, data):
        if isinstance(data, int):
            data = [data]
        elif isinstance(data, list):
            if not all([isinstance(v, int) for v in data]):
                raise ValueError("Data must be a list of integers.")
        else:
            raise ValueError("Data must be an integer or a list of integers.")

        if self.debug:
            for off, d in enumerate(data):
                self.debug_memory[addr + off*0x4] = d
            return

        self._check_connected()
        max_ct = 0xfffc #max writeable block length (TODO: check if still true)
        cmd = 0x80000000 #write to RAM command
        datagrams_written = 0
        init_addr = addr
        idx = 0

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
            assert (resp[2*ct] == 0x80800000 + words_written)
            assert (resp[2*ct+1] == addr)
            addr += 4 * words_written

    def read_memory(self, addr, num_words):

        if self.debug:
            response = []
            for x in range(num_words):
                response.append(self.debug_memory.get(addr+0x4*x, 0x0))
            return response[0] if num_words == 1 else response

        self._check_connected()
        datagram = [0x10000000 + num_words, addr]
        self.send_bytes(datagram)
        resp_header = self.recv_bytes(2 * 4) #4 bytes per word...
        return self.recv_bytes(4 * num_words)

#####################################################################

#APS3 AMC599 Control and Status Register offsets
#Add to CSR_AXI_ADDR_BASE to get to correct memory location
#Registers are read/write unless otherwise noted
#Current as of 6/20/19

CSR_AXI_ADDR_BASE           = 0x44b40000

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

class APS3(Instrument, metaclass=MakeBitFieldParams):

    instrument_type = "AWG"

    def __init__(self, resource_name=None, name="Unlabled APS3", debug=False):
        self.name = name
        self.resource_name = resource_name
        self.board = AMC599(debug=debug)
        super().__init__()

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

    def write_register(self, offset, data):
        logger.info(f"Setting CSR: {hex(offset)} to: {hex(data)}")
        self.board.write_memory(CSR_AXI_ADDR_BASE + offset, data)

    def read_register(self, offset, num_words = 1):
        return self.board.read_memory(CSR_AXI_ADDR_BASE + offset, num_words)

    def write_dram(self, offset, data):
        self.board.write_memory(DRAM_AXI_BASE + offset, data)

    def read_dram(self, offset, num_words = 1):
        return self.board.read_memory(DRAM_AXI_BASE + offset, num_words)

    ####### CACHE CONTROL REGSITER #############################################

    cache_controller = BitFieldCommand(register=CSR_CACHE_CONTROL, shift=0,
        doc="""Cache controller enable bit.""")

    ####### WAVEFORM OFFSET REGISTERS ##########################################

    wf_offset_A = BitFieldCommand(register=CSR_WFA_OFFSET, shift=0, mask=U32)
    wf_offset_B = BitFieldCommand(register=CSR_WFB_OFFSET, shift=0, mask=U32)

    ####### SEQUENCER CONTROL REGISTER #########################################

    sequencer_enable = BitFieldCommand(register=CSR_SEQ_CONTROL, shift=0,
        doc="""Sequencer, trigger input, modulator, SATA, VRAM, debug stream enable bit.""")

    trigger_source = BitFieldCommand(register=CSR_SEQ_CONTROL, shift=1,
        value_map={"EXTERNAL": 0b00, "INTERNAL": 0b01, "SOFTWARE": 0b10, "MESSAGE": 0b11})

    soft_trigger = BitFieldCommand(register=CSR_SEQ_CONTROL, shift=3)
    trigger_enable = BitFieldCommand(register=CSR_SEQ_CONTROL, shift=4)
    bypass_modulator = BitFieldCommand(register=CSR_SEQ_CONTROL, shift=5)
    bypass_nco = BitFieldCommand(register=CSR_SEQ_CONTROL, shift=6)

    ####### CORRECTION CONTROL REGISTER ########################################
    @property
    def correction_control(self):
        reg = self.read_register(CSR_CORR_OFFSET)
        return ((reg >> 16) & U16, reg & U16) #returns (I, Q)
    @correction_control.setter
    def correction_control(self, value):
        packed_value = ((value[0] & U16) << 16) | (value[1] & U16)
        self.write_register(CSR_CORR_OFFSET, packed_value)

    ####### TRIGGER INTERVAL ###################################################

    trigger_interval = BitFieldCommand(register=CSR_TRIG_INTERVAL, shift=0, mask=U32)

    ####### UPTIME REGISTERS ###################################################

    uptime_seconds = BitFieldCommand(register=CSR_UPTIME_SEC, shift=0,
                                        mask=U32, readonly=True)
    uptime_nanoseconds = BitFieldCommand(register=CSR_UPTIME_NS, shift=0,
                                        mask=U32, readonly=True)

    ####### BUILD INFO REGISTERS ###############################################
    def get_firmware_info(self):
        fpga_rev = self.read_register(CSR_FPGA_REV)
        fpga_id = self.read_register(CSR_FPGA_ID)
        git_sha1 = self.read_register(CSR_GIT_SHA1)
        build_tstamp = self.read_register(CSR_BUILD_TSTAMP)

        fpga_build_minor = check_bits(fpga_rev, 0, 0xFF)
        fpga_build_major = check_bits(fpga_rev, 8, 0xFF)
        commit_history = check_bits(fpga_rev, 16, 0x7FF)
        build_clean = "dirty" if check_bits(fpga_rev, 27, 0xF) else "clean"

        return {"FPGA ID": hex(fpga_id),
                "FPGA REV": f"{fpga_build_major}.{fpga_build_minor} - {commit_history} - {build_clean}",
                "GIT SHA1": hex(git_sha1),
                "DATE": hex(build_tstamp)[2:]}

    ####### CORRECTION MATRIX ##################################################
    def get_correction_matrix(self):
        row0 = self.read_register(CSR_CMAT_R0)
        row1 = self.read_register(CSR_CMAT_R1)
        r00 = (row0 >> 16) & U16
        r01 = row0 & U16
        r10 = (row1 >> 16) & U16
        r11 = row0 & U16
        return np.array([[r00, r01], [r10, r11]], dtype=np.uint16)

    def set_correction_matrix(self, matrix):
        row0 = ((matix[0, 0] & U16) << 16) | (matrix[0, 1] & U16)
        row1 = ((matix[1, 0] & U16) << 16) | (matrix[1, 1] & U16)
        self.write_register(CSR_CMAT_R0, row0)
        self.write_register(CSR_CMAT_R1, row1)

    def correction_bypass(self):
        row0 = 0x20000000
        row1 = 0x00002000
        self.write_register(CSR_CMAT_R0, row0)
        self.write_register(CSR_CMAT_R0, row1)
        self.write_register(CSR_CORR_OFFSET, 0x0)


    ####### BOARD_CONTROL ######################################################

    microblaze_enable = BitFieldCommand(register=CSR_BD_CONTROL, shift=1)
    dac_output_mux = BitFieldCommand(register=CSR_BD_CONTROL, shift=4,
                                    value_map={"SOF200": 0x0, "APS": 0x1})

    ####### MARKER_DELAY #######################################################

    marker_delay = BitFieldCommand(register=CSR_MARKER_DELAY, shift=0,
                                    mask=U16, allowed_values=[0,U16])

    ###### DRAM OFFSET REGISTERS ###############################################
    def SEQ_OFFSET(self):
        return (self.read_register(CSR_SEQ_OFFSET) - DRAM_AXI_BASE)

    def WFA_OFFSET(self):
        return (self.read_register(CSR_WFA_OFFSET) - DRAM_AXI_BASE)

    def WFB_OFFSET(self):
        return (self.read_register(CSR_WFB_OFFSET) - DRAM_AXI_BASE)

    ###### UTILITIES ###########################################################
    def run(self):
        logger.info("Taking cache controller out of reset...")
        self.cache_controller = True
        sleep(0.01)
        logger.info("Taking sequencer out of reset...")
        self.sequencer_enable = True
        sleep(0.01)
        logger.info("Enabling trigger...")
        self.trigger_enable = True
        sleep(0.01)

    def stop(self):
        logger.info("Resetting cache controller...")
        self.cache_controller = False
        sleep(0.01)
        logger.info("Resetting sequencer...")
        self.sequencer_enable = False

    def write_sequence(self, sequence):
        packed_seq = []
        for instr in sequence:
            packed_seq.append(instr & U32)
            packed_seq.append(instr >> 32)

        self.cache_controller = False
        sleep(0.01)
        self.write_dram(self.SEQ_OFFSET, packed_seq)
        logger.info(f"Wrote {len(packed_seq)} words to sequence memory.")
        sleep(0.01)
        self.cache_controller = True
