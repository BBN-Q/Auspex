# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['APS3']

from .instrument import Instrument, is_valid_ipv4, Command, MetaInstrument
from .bbn import MakeSettersGetters
from auspex.log import logger
import auspex.config as config
from time import sleep
import numpy as np
import socket
from unittest.mock import Mock
import collections
from struct import pack, iter_unpack
import serial

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
    setattr(getattr(instr, "set_"+name), "__doc__", new_cmd.doc)
    setattr(instr, "get_"+name, fget)
    setattr(getattr(instr, "set_"+name), "__doc__", new_cmd.doc)

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
    """Base class for simple register manipulations of AMC599 board and DAC.
    """

    PORT = 0xbb4e # TCPIP port (BBN!)
    ser = None
    ref = ''

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

    def connect(self, resource=("192.168.2.200", "COM1")):
        self.ip_addr = resource[0]
        if not self.debug:
            self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.socket.connect((self.ip_addr, self.PORT))
            self.ser = serial.Serial(resource[1], 115200)
            self.connected = True

    def disconnect(self):
        if self.connected:
            self.connected = False
            self.socket.close()
            self.ser.close()

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
            # logger.debug("Wrote {} words in {} datagrams: {}", words_written,
            #                     datagrams_written,
            #                     [hex(x) for x in resp])
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

    def serial_read_dac_register(self, dac, addr):
        if dac not in [0, 1]:
            raise ValueError('Invalid DAC number ' + str(dac))
        self.ser.reset_output_buffer()
        self.ser.reset_input_buffer()
        self.ser.write(bytearray('rd d{} {:#x}\n'.format(dac, addr), 'ascii'))
        self.ser.readline() # Throw out the echo line from the terminal interface
        resp = self.ser.readline().decode()
        start_index = len('Read value = ')
        end_index = resp.find('@')
        return int(resp[start_index:end_index], 16)

    def serial_write_dac_register(self, dac, addr, val):
        if dac not in [0, 1]:
            raise ValueError('Invalid DAC number ' + str(dac))
        self.ser.reset_output_buffer()
        self.ser.reset_input_buffer()
        self.ser.write(bytearray('wd d{} {:#x} {:#x}\n'.format(dac, addr, val), 'ascii'))
        self.ser.readline() # Throw out the echo line from the terminal interface
        return self.ser.readline() # Echo back the "wrote xx to xx" line

    def serial_configure_JESD(self, dac):
        # Configure the JESD interface properly
        logger.debug(self.serial_write_dac_register(dac, 0x300, 0x00)) # disable all links
        sleep(0.01)
        logger.debug(self.serial_write_dac_register(dac, 0x475, 0x09)) # soft reset DAC0 deframer
        sleep(0.01)
        logger.debug(self.serial_write_dac_register(dac, 0x110, 0x81)) # set interpolation to 2
        sleep(0.01)
        logger.debug(self.serial_write_dac_register(dac, 0x456, 0x01)) # set M=2
        sleep(0.01)
        logger.debug(self.serial_write_dac_register(dac, 0x459, 0x21)) # set S=2
        sleep(0.01)
        logger.debug(self.serial_write_dac_register(dac, 0x477, 0x00)) # disable ILS_MODE for DAC0
        sleep(0.01)
        logger.debug(self.serial_write_dac_register(dac, 0x475, 0x01)) # bring DAC0 deframer out of reset
        sleep(0.01)
        logger.debug(self.serial_write_dac_register(dac, 0x300, 0x01)) # enable all links
        sleep(0.01)

    def serial_set_switch_mode(self, dac, mode):
        '''
        Sets DAC output switch mode to one of NRZ, Mix-Mode, or RZ.
        Parameters:
            mode (string): Switch mode, one of "NRZ", "MIX", or "RZ"
        '''
        if mode == 'NRZ':
            code = 0x00
        elif mode == 'MIX':
            code = 0x01
        elif mode == 'RZ':
            code = 0x02
        else:
            raise Exception('DAC switch mode "' + mode + '" not recognized.')

        if self.ser is None:
            logger.debug('Fake wrote {:#x}'.format(code))
        else:
            logger.debug(self.serial_write_dac_register(dac, 0x152, code))

    def serial_get_switch_mode(self, dac):
        '''
        Reads DAC output switch mode as one of NRZ, Mix-Mode, or RZ.
        Parameters:
            mode (string): Switch mode, one of "NRZ", "MIX", or "RZ"
        '''
        if self.ser is None:
            logger.debug('Fake read mix-mode.')
            return 'MIX'

        code = self.serial_read_dac_register(dac, 0x152) & 0x03
        if code == 0x00:
            return 'NRZ'
        if code == 0x01:
            return 'MIX'
        if code == 0x02:
            return 'RZ'

        raise Exception('Unrecognized DAC switch mode ' + code + '.')

    def serial_set_analog_full_scale_current(self, dac, current):
        '''
        Sets DAC full-scale current, rounding to nearest LSB of current register.
        Parameters:
            current (float): Full-scale current in mA

        Returns:
            (float) actual programmed current in mA
        '''
        if current < 8 or current > 40:
            raise Exception('DAC full-scale current must be between 8 mA and 40 mA.')

        # From AD9164 datasheet:
        # IOUTFS = 32 mA × (ANA_FULL_SCALE_CURRENT[9:0]/1023) + 8 mA
        reg_value = int(1023 * (current - 8) / 32)

        if self.ser is None:
            logger.debug('{:#x}'.format(reg_value & 0x3))
            logger.debug('{:#x}'.format((reg_value >> 2) & 0xFF))
        else:
            logger.debug(self.serial_write_dac_register(dac, 0x041, reg_value & 0x3))
            sleep(0.01)
            logger.debug(self.serial_write_dac_register(dac, 0x042, (reg_value >> 2) & 0xFF))
            sleep(0.01)

        return 32 * (reg_value / 1023) + 8

    def serial_get_analog_full_scale_current(self, dac):
        '''
        Reads programmed full-scale current.
        Returns:
            Full-scale current in mA
        '''
        if self.ser is None:
            return 0

        LSbits = self.serial_read_dac_register(dac, 0x041) & 0x03
        MSbits = self.serial_read_dac_register(dac, 0x042) & 0xFF
        reg_value = (MSbits << 2) & LSbits
        return 32 * (reg_value / 1023) + 8

    def serial_set_nco_enable(self, dac, en):
        '''
        Enables the DAC NCO.
        Parameters:
            en (bool): Enables the NCO if True, disables it if False
            FIR85 (bool): Enables the FIR85 NCO filter if True, disables it if False
        '''
        # Configure NCO_EN (Bit 6) = 0b1
        # Set the reserved bits (Bit 5 and Bit 3) to 0b0
        if self.ser is None:
            logger.debug('Fake read 0x00.')
            code = 0x00
        else:
            code = self.serial_read_dac_register(dac, 0x111)

        if en:
            code |= (1 << 6)
        else:
            code &= ~(1 << 6)

        if self.ser is None:
            logger.debug('Fake wrote {:#x}'.format(code))
        else:
            logger.debug(self.serial_write_dac_register(dac, 0x111, code))

        sleep(0.1)

    def serial_get_nco_enable(self, dac):
        '''
        Checks whether the DAC NCO is enabled.
        Returns:
            True if DAC NCO is enabled, otherwise False
        '''
        if self.ser is None:
            logger.debug('Fake reported DAC NCO enabled.')
            return True

        return (self.serial_read_dac_register(dac, 0x111) & (1 << 6)) != 0

    def serial_set_FIR85_enable(self, dac, FIR85):
        '''
        Enables the DAC NCO FIR85 filter.
        Parameters:
            FIR85 (bool): Enables the FIR85 NCO filter if True, disables it if False
        '''
        if self.ser is None:
            logger.debug('Fake read 0x00.')
            code = 0x00
        else:
            code = self.serial_read_dac_register(dac, 0x111)

        if FIR85:
            code |= (1 << 0)
        else:
            code &= ~(1 << 0)

        if self.ser is None:
            logger.debug('Fake wrote {:#x}'.format(code))
        else:
            logger.debug(self.serial_write_dac_register(dac, 0x111, code))

        sleep(0.1)

    def serial_get_FIR85_enable(self, dac):
        '''
        Checks whether the DAC NCO FIR85 filter is enabled.
        Returns:
            True if DAC NCO FIR85 filter is enabled, otherwise False
        '''
        if self.ser is None:
            logger.debug('Fake reported DAC NCO FIR85 enabled.')
            return True

        return (self.serial_read_dac_register(dac, 0x111) & (1 << 0)) != 0

    def serial_set_nco_frequency(self, dac, f):
        '''
        Writes the given frequency, assuming not in NCO-only mode.
        Follows procedure in Table 44 of AD9164 datasheet.
        '''
        logger.debug('Setting frequency to {}...'.format(f))

        # Configure DC_TEST_EN bit: 0b0 = NCO operation with data interface
        logger.debug(self.serial_write_dac_register(dac, 0x150, 0x00))
        sleep(0.01)

        # Ensure the frequency tuning word write request is low.
        logger.debug(self.serial_write_dac_register(dac, 0x113, 0x00))
        sleep(0.01)

        # Write FTW.
        ftw = [(int((f/5e9)*(1 << 48)) >> x) & 0xFF for x in range(0, 48, 8)]
        for index, b in enumerate(ftw):
            logger.debug(self.serial_write_dac_register(dac, 0x114 + index, b))
            sleep(0.01)

        # Load the FTW to the NCO.
        logger.debug(self.serial_write_dac_register(dac, 0x113, 0x01))
        sleep(0.1)

    def serial_get_nco_frequency(self, dac):
        '''
        Reads the current NCO frequency, assuming not in NCO-only mode.
        '''
        ftw = 0
        for index, shift in enumerate(range(0, 48, 8)):
            ftw |= self.serial_read_dac_register(dac, 0x114 + index) << shift
            sleep(0.01)

        return (ftw / float(1 << 48)) * 5e9

    def serial_set_reference(self, ref):
        '''
        Sets the SOF200 PLL reference to either the front panel or the FPGA.
        Parameters:
            ref (str): Either "REF IN" or "FPGA"
        '''
        if ref == 'REF IN':
            self.ref = ref
            self.ser.reset_output_buffer()
            self.ser.reset_input_buffer()
            self.ser.write(bytearray('rs fp\n', 'ascii'))
            self.ser.readline() # Throw out the echo line from the terminal interface
        elif ref == 'FPGA':
            self.ref = ref
            self.ser.reset_output_buffer()
            self.ser.reset_input_buffer()
            self.ser.write(bytearray('rs fpga\n', 'ascii'))
            self.ser.readline() # Throw out the echo line from the terminal interface
        else:
            logger.debug('Error: unrecognized reference input "' + ref + '".')

    def serial_get_reference(self):
        return self.ref

    def serial_set_shuffle_mode(self, dac, value):
        '''
        Sets DAC shuffle mode.
        Parameters:
            value (int): Sets the shuffle register bits
        '''
        if self.ser is None:
            logger.debug('Fake wrote {:#x}'.format(value & 0x7))
        else:
            logger.debug(self.serial_write_dac_register(dac, 0x151, value & 0x7))

        sleep(0.1)

    def serial_get_shuffle_mode(self, dac):
        '''
        Checks whether DAC shuffle mode is enabled.
        Returns:
            True if DAC shuffle is enabled, otherwise False
        '''
        if self.ser is None:
            logger.debug('Fake reported DAC shuffle mode enabled.')
            return True

        return self.serial_read_dac_register(dac, 0x151) & 0x7

#####################################################################

#APS3 AMC599 Control and Status Register offsets
#Add to CSR_AXI_ADDR_BASE to get to correct memory location
#Registers are read/write unless otherwise noted
#Current as of 6/20/19

CSR_AXI_ADDR_BASE0           = 0x44b40000
CSR_AXI_ADDR_BASE1           = 0x44b10000

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
DRAM_WFA_0_LOC = 0x80000000
DRAM_WFB_0_LOC = 0x90000000
DRAM_SEQ_0_LOC = 0xA0000000
DRAM_WFA_1_LOC = 0xB0000000
DRAM_WFB_1_LOC = 0xC0000000
DRAM_SEQ_1_LOC = 0xD0000000


#####################################################################

class APS3CommunicationManager(object):
    instances = {} # Open instances of AMC599 objects, referenced by (IP, serialport) tuple

    @staticmethod
    def board(resource):
        if resource not in APS3CommunicationManager.instances:
            APS3CommunicationManager.instances[resource] = {'board': AMC599(), 'connected': False, 'running': False}
        return APS3CommunicationManager.instances[resource]['board']

    @staticmethod
    def connect(resource):
        if resource not in APS3CommunicationManager.instances:
            APS3CommunicationManager.instances[resource] = {'board': AMC599(), 'connected': False, 'running': False}
        if not APS3CommunicationManager.connected(resource):
            APS3CommunicationManager.instances[resource]['board'].connect(resource)
            APS3CommunicationManager.instances[resource]['connected'] = True

    @staticmethod
    def connected(resource):
        if resource not in APS3CommunicationManager.instances:
            return False
        return APS3CommunicationManager.instances[resource]['connected']

    @staticmethod
    def disconnect(resource):
        if APS3CommunicationManager.connected(resource):
            APS3CommunicationManager.instances[resource]['board'].disconnect()
            APS3CommunicationManager.instances[resource]['connected'] = False

    @staticmethod
    def set_run(resource):
        APS3CommunicationManager.instances[resource]['running'] = True

    @staticmethod
    def set_stop(resource):
        APS3CommunicationManager.instances[resource]['running'] = False

class APS3(Instrument, metaclass=MakeBitFieldParams):

    instrument_type = "AWG"
    dac = -1
    address = None

    def __init__(self, resource_name=None, name="Unlabeled APS3", debug=False):
        self.name = name
        self.resource_name = resource_name
        super().__init__()

    def connect(self, resource_name=None):
        if resource_name is None and self.resource_name is None:
            raise ValueError("Must supply a resource name!")
        elif resource_name is not None:
            self.resource_name = resource_name

        if len(self.resource_name) != 3:
            raise ValueError("Resource name must have 3 elements!")
        if self.resource_name[0] == None:
            raise ValueError("Resource name must contain IP address!")
        if self.resource_name[1] == None:
            raise ValueError("Resource name must contain serial port!")
        if self.resource_name[2] == None:
            raise ValueError("Resource name must contain channel!")
        if not isinstance(self.resource_name[2], int) or not self.resource_name[2] in [0, 1]:
            raise ValueError("Channel name must be 0 or 1!")

        if not is_valid_ipv4(self.resource_name[0]):
            raise ValueError("IP address must be valid!")

        self.address = (self.resource_name[0], self.resource_name[1])
        self.dac = self.resource_name[2]

        APS3CommunicationManager.connect(self.address)

        # Write the memory locations immediately
        self.write_register(CSR_WFA_OFFSET, (DRAM_WFA_0_LOC if self.dac == 0 else DRAM_WFA_1_LOC))
        self.write_register(CSR_WFB_OFFSET, (DRAM_WFB_0_LOC if self.dac == 0 else DRAM_WFB_1_LOC))
        self.write_register(CSR_SEQ_OFFSET, (DRAM_SEQ_0_LOC if self.dac == 0 else DRAM_SEQ_1_LOC))

    def disconnect(self):
        if APS3CommunicationManager.connected(self.address):
            APS3CommunicationManager.disconnect(self.address)

    def write_register(self, offset, data):
        logger.debug(f"Setting CSR: {hex(offset)} to: {hex(data)}")
        APS3CommunicationManager.board(self.address).write_memory((CSR_AXI_ADDR_BASE0 if self.dac == 0 else CSR_AXI_ADDR_BASE1) + offset, data)

    def read_register(self, offset, num_words = 1):
        return APS3CommunicationManager.board(self.address).read_memory((CSR_AXI_ADDR_BASE0 if self.dac == 0 else CSR_AXI_ADDR_BASE1) + offset, num_words)

    def write_dram(self, offset, data):
        APS3CommunicationManager.board(self.address).write_memory(DRAM_AXI_BASE + offset, data)

    def read_dram(self, offset, num_words = 1):
        return APS3CommunicationManager.board(self.address).read_memory(DRAM_AXI_BASE + offset, num_words)

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
        value_map={"external": 0b00, "internal": 0b01, "software": 0b10, "message": 0b11})

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

    @property
    def trigger_interval(self):
        """Gets/sets the trigger interval in seconds, based on a 312.5 MHz clock."""
        return int(self.read_register(CSR_TRIG_INTERVAL)) / 312.5e6
    @trigger_interval.setter
    def trigger_interval(self, value):
        trig_bits = int(value * 312.5e6)
        assert (trig_bits >= 0 and trig_bits < U32), "Trigger interval out of range!"
        self.write_register(CSR_TRIG_INTERVAL, trig_bits)

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

    csr0_master = BitFieldCommand(register=CSR_BD_CONTROL, shift=0,
        doc="""True: CSR0 is the Master CSR; when this bit is set the Cache Reset and Sequencer Resets
            are controlled by CSR0 for both DAC0 and DAC1; so in effect both DACs circuits are reset at the same time.
            False: CSR0 only controls DAC0 resets; CSR1 controls the DAC1 resets.""")

    microblaze_reset = BitFieldCommand(register=CSR_BD_CONTROL, shift=1,
        doc="True resets Microblaze softcore. False takes Microblaze out of reset.")

    dac_output_mux = BitFieldCommand(register=CSR_BD_CONTROL, shift=4,
                                    value_map={"SOF200": 0x0, "APS": 0x1},
                                    doc="Select SOF200 test output or APS sequencer output from DAC.")

    @property
    def trigger_output_select(self):
        """Gets/sets the marker delay in seconds, based on a 312.5 MHz clock."""
        return (self.read_register(CSR_BD_CONTROL) >> 5) & 0x3
    @trigger_output_select.setter
    def trigger_output_select(self, value):
        reg = self.read_register(CSR_BD_CONTROL)
        reg &= ~(0x3 << 5)
        reg |= (value & 0x3) << 5
        self.write_register(CSR_BD_CONTROL, reg)

    trigger_input_select = BitFieldCommand(register=CSR_BD_CONTROL, shift=7,
        doc="""True: Use the trigger form the other DAC as the trigger source.  When this bit is set then
                for CSR0: DAC0 use DAC1 trigger; For CSR1 for DAC1 use DAC0 trigger.
               False: Use the trigger from the same DAC as the source.
                for CSR0: DAC0 use DAC0 trigger; For CSR1 for DAC1 use DAC1 trigger.""")

    ####### MARKER_DELAY #######################################################

    @property
    def marker_delay(self):
        """Gets/sets the marker delay in seconds, based on a 312.5 MHz clock."""
        return (1 + int(self.read_register(CSR_MARKER_DELAY))) / 312.5e6
    @marker_delay.setter
    def marker_delay(self, value):
        trig_bits = int(value * 312.5e6) + 1
        assert (trig_bits >= 0 and trig_bits < U16), "Marker delay out of range!"
        self.write_register(CSR_MARKER_DELAY, trig_bits)

    ###### DRAM OFFSET REGISTERS ###############################################
    def SEQ_OFFSET(self):
        return (self.read_register(CSR_SEQ_OFFSET) - DRAM_AXI_BASE)

    def WFA_OFFSET(self):
        return (self.read_register(CSR_WFA_OFFSET) - DRAM_AXI_BASE)

    def WFB_OFFSET(self):
        return (self.read_register(CSR_WFB_OFFSET) - DRAM_AXI_BASE)

    ####### GPIO REGISTERS #####################################################

    GPIO0_high = BitFieldCommand(register=CSR_BD_CONTROL, shift=16, mask=U16)
    GPIO0_low  = BitFieldCommand(register=CSR_BD_CONTROL, shift=8, mask=0xFF)
    GPIO1 = BitFieldCommand(register=CSR_DATA1_IO, shift=0, mask=U32)
    GPIO2 = BitFieldCommand(register=CSR_DATA2_IO, shift=0, mask=U32)

    ###### UTILITIES ###########################################################
    def run(self):
        logger.debug("Configuring JESD...")
        #APS3CommunicationManager.board(self.address).serial_configure_JESD()
        sleep(0.01)
        logger.debug("Taking cache controller out of reset...")
        self.cache_controller = True
        sleep(0.01)
        logger.debug("Taking sequencer out of reset...")
        self.sequencer_enable = True
        sleep(0.01)
        logger.debug("Enabling trigger...")
        self.trigger_enable = True
        sleep(0.01)

    def stop(self):
        logger.debug("Resetting cache controller...")
        self.cache_controller = False
        sleep(0.01)
        logger.debug("Resetting sequencer...")
        self.sequencer_enable = False

    ###### SEQUENCE LOADING ####################################################

    sequence_filename = ''

    def load_sequence(self, sequence):
        packed_seq = []
        for instr in sequence:
            packed_seq.append(instr & U32)
            packed_seq.append(instr >> 32)

        sleep(0.01)
        self.write_dram(self.SEQ_OFFSET(), packed_seq)
        logger.debug(f"Wrote {len(packed_seq)} words to sequence memory.")
        sleep(0.01)

    def load_waveforms(self, wfA, wfB):
        wfA_32 = [((wfA[2*i+1] << 16) | wfA[2*i]) for i in range(len(wfA) // 2)]
        wfB_32 = [((wfB[2*i+1] << 16) | wfB[2*i]) for i in range(len(wfB) // 2)]

        if len(wfA_32) > 0 and len(wfB_32) > 0:
            self.write_dram(self.WFA_OFFSET(), wfA_32) # I
            self.write_dram(self.WFB_OFFSET(), wfB_32) # Q
        else:
            logger.warning('Discarding zero-length waveform.')

    def read_waveforms(self, wf_len):
        wfA_32 = self.read_dram(self.WFA_OFFSET(), wf_len // 2)
        wfB_32 = self.read_dram(self.WFB_OFFSET(), wf_len // 2)

        wfA = []
        wfB = []

        for i in range(wf_len // 2):
            wfA.append(wfA_32[i] & 0xFFFF)
            wfA.append((wfA_32[i] >> 16) & 0xFFFF)
            wfB.append(wfB_32[i] & 0xFFFF)
            wfB.append((wfB_32[i] >> 16) & 0xFFFF)

        return wfA, wfB

    def load_sequence_file(self, seq_file):
        self.sequence_filename = seq_file
        with open(seq_file, 'rb') as file:
            instrument_type = file.read(4)
            if instrument_type != b'APS3':
                raise ValueError('Sequence file not designated for APS3; header for ' + str(instrument_type) + ' found.')

            version = np.frombuffer(file.read(4), dtype=np.float32)[0]
            min_firmware_version = np.frombuffer(file.read(4), dtype=np.float32)[0]
            num_channels = int(np.frombuffer(file.read(2), dtype=np.uint16)[0])

            if num_channels != 2:
                raise ValueError('Unexpected number of channels, sequence file reports ' + str(num_channels) + ' channels.')

            instructions_size = int(np.frombuffer(file.read(8), dtype=np.uint64)[0])
            instructions = [int(x) for x in np.frombuffer(file.read(instructions_size*8), dtype=np.uint64)]

            data = []
            for chan in range(num_channels):
                data_size = int(np.frombuffer(file.read(8), dtype=np.uint64)[0])
                data.append([int(x) for x in np.frombuffer(file.read(data_size*2), dtype=np.uint16)])

            self.load_waveforms(data[0], data[1])
            self.load_sequence(instructions)

    def serial_check_alive(self):
        return (APS3CommunicationManager.board(self.address).serial_read_dac_register(self.dac, 0x005) == 0x91 and
        APS3CommunicationManager.board(self.address).serial_read_dac_register(self.dac,0x004) == 0x64)

    def configure_with_proxy(self, proxy_obj):
        super(APS3, self).configure_with_proxy(proxy_obj)

    @property
    def sequence_file(self):
        return self.sequence_filename

    @sequence_file.setter
    def sequence_file(self, value):
        self.load_sequence_file(value)

    @property
    def dac_switch_mode(self):
        return APS3CommunicationManager.board(self.address).serial_get_switch_mode(self.dac)

    @dac_switch_mode.setter
    def dac_switch_mode(self, value):
        APS3CommunicationManager.board(self.address).serial_set_switch_mode(self.dac, value)

    @property
    def dac_full_scale_current(self):
        return APS3CommunicationManager.board(self.address).serial_get_analog_full_scale_current(self.dac)

    @dac_full_scale_current.setter
    def dac_full_scale_current(self, value):
        APS3CommunicationManager.board(self.address).serial_set_analog_full_scale_current(self.dac, value)

    @property
    def dac_nco_enable(self):
        return APS3CommunicationManager.board(self.address).serial_get_nco_enable(self.dac)

    @dac_nco_enable.setter
    def dac_nco_enable(self, value):
        APS3CommunicationManager.board(self.address).serial_set_nco_enable(self.dac, value)

    @property
    def dac_FIR85_enable(self):
        return APS3CommunicationManager.board(self.address).serial_get_FIR85_enable(self.dac)

    @dac_FIR85_enable.setter
    def dac_FIR85_enable(self, value):
        APS3CommunicationManager.board(self.address).serial_set_FIR85_enable(self.dac, value)

    @property
    def dac_nco_frequency(self):
        return APS3CommunicationManager.board(self.address).serial_get_nco_frequency(self.dac)

    @dac_nco_frequency.setter
    def dac_nco_frequency(self, value):
        APS3CommunicationManager.board(self.address).serial_set_nco_frequency(self.dac, value)

    @property
    def dac_pll_reference(self):
        return APS3CommunicationManager.board(self.address).serial_get_reference()

    @dac_pll_reference.setter
    def dac_pll_reference(self, value):
        APS3CommunicationManager.board(self.address).serial_set_reference(value)

    @property
    def dac_shuffle_mode(self):
        return APS3CommunicationManager.board(self.address).serial_get_shuffle_mode(self.dac)

    @dac_shuffle_mode.setter
    def dac_shuffle_mode(self, value):
        APS3CommunicationManager.board(self.address).serial_set_shuffle_mode(self.dac, value)
