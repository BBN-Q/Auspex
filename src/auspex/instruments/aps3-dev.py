# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['APS3', 'KCU105_Board']

from .instrument import Instrument, MetaInstrument, is_valid_ipv4
from .bbn import MakeSettersGetters
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

import h5py
import fractions

class KCU105_Board(object):

    PORT = 0xbb4e #BBN, of course

    def __init__(self):
        pass

    def __del__(self):
        if self.connected:
            self.connected = False
            self.socket.shutdown()
            self.socket.close()

    def _check_connected(self):
        if not self.connected:
            raise IOError("KCU105 Board is not connected!")

    def connect(self, ip_addr="192.168.2.200"):
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
        self._check_connected()
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
        self._check_connected()
        datagram = [0x10000000 + num_words, addr]
        self.send_bytes(datagram)
        resp_header = self.recv_bytes(2 * 2) #2 bytes per word
        print(hex(resp_header))
        return self.recv_bytes(2 * num_words) #2 bytes per word

CSR_AXI_ADDR = 0x44a00000
CSR_CONTROL_OFFSET = 0x00
GPIO1_OFFSET = 0x01C
GPIO2_OFFSET = 0x020
CSR_TRIGGER_INTERVAL_OFFSET = 0x14
SDRAM_AXI_ADDR = 0x80000000

class APS3(Instrument, metaclass=MakeSettersGetters):

    yaml_template = """
        APS3-Name:
            type: APS3
            enabled: true
            master: true
            address:
            trigger_interval: 0.0
            seq_file: test.h5
            dac_clock: 5e9
            dac_mode: mix #mix, nz or rz
            nco_frequency: 1e9
            tx_channels:
              '1':
                enabled: true
    """

    instrument_type = "AWG"

    def __init__(self, resource_name=None, name="Unlabled APS3"):
        self.name = name
        self.resource_name = resource_name

        self.connected = False
        self.board = KCU105_Board()

        self.dac_clock = 5e9
        self.nco_frequency = 0
        self.dac_mode = "MIX"

    def connect(self, resource_name=None):
        if resource_name is None and self.resource_name is None:
            raise Exception("Must supply a resource name to `connect` if the instrument was initialized without one.")
        elif resource_name is not None:
            self.resource_name = resource_name
        if is_valid_ipv4(self.resource_name):
            self.board.connect(ip_addr = resource_name)
            self.connected = True

    def set_all(self, settings_dict, prefix=""):
        settings = deepcopy(settings_dict)
        channels = settings.pop('tx_channels')
        super(APS3, self).set_all(settings)

        for key in ['address', 'seq_file', 'master', 'trigger_interval']:
            if key not in settings.keys():
                raise ValueError("Instrument {} configuration lacks mandatory key {}".format(self, key))

        for chan_group in ('1'):
            chan = channels.pop(chan_group, None)
            if not chan:
                raise ValueError("APS3 {} Expected to receive channel '{}'".format(self, chan_group))
            pass #for now do nothing as channels do not yet have individual properties....

    def _load_waves_from_file(self, filename):

        with h5py.File(filename, "w") as f:

            target = FID['/'].attrs['target hardware']
            if not (isinstance(target, str) and (target == "APS3")):
                raise IOError("Invalid sequence file!")

            self.num_seq = FID['/'].attrs['num sequences']
            self.marker_delay = FID['/'].attrs['marker delay']

            waves = FID['waveforms']

        if self.num_seq != len(waves):
            raise ValueError("Sequence file attributes and waveform data out of sync!")

        N = len(waves)
        wf_lengths = np.array([len(wf) for wf in waves], dtype=np.uint32)
        N_pad = ((2*N-1)|15) + 1
        header = np.zeros(N_pad, dtype=np.uint32)
        addr = 4 * N_pad
        wf_addrs = np.array([], dtype=np.uint32)
        for wf in waves:
            wf_addrs.append(addr, np.uint32(addr))
            addr += 4*len(wf)
        header[0:N-1] = wf_lengths
        header[N:2*N-1] = wf_addrs
        self.board.write_memory(SDRAM_AXI_ADDR, header)
        addr = 4 * N_pad
        for wf in waves:
            self.board.write_memory(SDRAM_AXI_ADDR, wf)
            addr += 4*len(wf)

        self.setup_waveform()


    @property
    def seq_file(self):
        return self._sequence_filename
    @seq_file.setter
    def seq_file(self, filename):
        self._sequence_filename = sequence_filename
        self._load_waves_from_file(filename)

    @property
    def num_seq(self):
        return self._num_seq
    @num_seq.setter
    def num_seq(self, num_seq):
        self._num_seq = num_seq

    @property
    def num_rr(self):
        return self._num_rr
    @num_rr.setter
    def num_rr(self, num_rr):
        self._num_rr = num_rr

    @property
    def marker_delay(self):
        return self._marker_delay
    @marker_delay.setter
    def marker_delay(self, marker_delay):
        self._marker_delay = marker_delay

    @property
    def dac_clock(self):
        return self._dac_clock
    @dac_clock.setter
    def dac_clock(self, clock):
        self._dac_clock = dac_clock

    @property
    def dac_mode(self):
        return self._dac_mode
    @dac_mode.setter
    def dac_mode(self, mode):
        if mode.upper() not in ("MIX", "NRZ", "RZ"):
            raise ValueError("Unrecognized AD9164 DAC mode.")
        self._dac_mode = mode

    @property
    def nco_frequency(self):
        return self._nco
    @nco_frequency.setter
    def nco_frequency(self, freq):
        self._nco = freq
        mod_frac = fractions.Fraction(int(freq), int(self.dac_clock))
        M = mod_frac.numerator
        N = mod_frac.denominator
        X = int(2**48 / N)
        Y = M*2**48 - X*N
        A = fractions.Fraction(Y, N).numerator
        B = fractions.Fraction(Y, N).denominator

        assert B > 0
        assert A < B

        print("NCO switching not yet implemented in Auspex... ")
        print("NCO FTW: {}".format(hex(X)))
        print("NCO Modulus A: {}".format(A))
        print("NCO Modulus B: {}".format(B))

    def set_trigger_interval(self, interval):
        num_clcks = np.uint32(interval * CLOCK_FREQ)
        self.board.write_memory(CSR_AXI_ADDR + CSR_TRIGGER_INTERVAL_OFFSET, [num_clcks - 0x2])

    def get_csr_value(self):
        return self.board.read_memory(CSR_AXI_ADDR + CSR_CONTROL_OFFSET, 1)

    def run(self):
        csr_val = self.get_csr_value()
        self.board.write_memory(CSR_AXI_ADDR + CSR_CONTROL_OFFSET, [csr_val | (1 << 16)])

    def stop(self):
        csr_val = self.get_csr_value()
        self.board.write_memory(CSR_AXI_ADDR + CSR_CONTROL_OFFSET, [csr_val | (0xF << 16)])

    def setup_waveform(self):
        self.board.write_memory(CSR_AXI_ADDR + GPIO1_OFFSET, [self.num_seq])
        self.board.write_memory(CSR_AXI_ADDR + GPIO2_OFFSET, [self.num_rr | (self.marker_delay << 16)])
        csr_val = self.get_csr_value()
        self.board.write_memory(CSR_AXI_ADDR + CSR_CONTROL_OFFSET, [csr_val | (2 << 16)])

    def memory_debug(self, addr_start, addr_end):
        self.board.write_memory(CSR_AXI_ADDR + GPIO1_OFFSET, [addr_start])
        self.board.write_memory(CSR_AXI_ADDR + GPIO2_OFFSET, [addr_end])
        csr_val = self.get_csr_value()
        self.board.write_memory(CSR_AXI_ADDR + CSR_CONTROL_OFFSET, [csr_val | (5 << 16)])
