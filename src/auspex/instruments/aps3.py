# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['APS3', 'KCU105']

from auspex.log import logger
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

pack_cython = False
try:
    from auspex.cython_modules.aps3_wavepack import pack_aps3_waveform_c
    pack_cython = True
except ImportError:
    logger.warning('Could not import Cython APS3 waveform packer, falling back to pure Python version.')
    pack_cython = False

def read_aps3_file(filename):

    with h5py.File(filename, "r") as FID:

        target = FID['/'].attrs['target hardware']
        if not (isinstance(target, str) and (target == "APS3")):
            raise IOError("Invalid sequence file!")

        num_seq = FID['/'].attrs['num sequences']
        marker_delay = FID['/'].attrs['marker delay']
        wf_lengths = FID['seq_lens'][:]

        if num_seq != len(wf_lengths):
            raise ValueError("Sequence file attributes and waveform data out of sync!")

        waves = []
        for ct in range(num_seq):
            waves.append(FID['seq_data_{:d}'.format(ct)][:])

    out = {'num_seq': num_seq,
                'marker_delay': marker_delay,
                'wf_lengths': wf_lengths,
                'waves': waves}
    return out



def pack_aps3_waveform(wave):

    N = len(wave)
    assert N % 8 == 0
    quad_ct = int(N/8)


    wf_scaled = wave * ((1 << 15) - 1) #scale to size
    wf_re = np.int16(np.around(np.real(wf_scaled)))
    wf_im = np.int16(np.around(np.imag(wf_scaled)))

    packed_wf = np.empty(N, np.uint32)
    packed_wf.fill(0xBAAA_AAAD)

    def pack_byte(b, pos):
        return ((np.int32(b).view(np.uint32) & 0x0000_00FF) << 8 * pos)

    # for eight lanes:
    # * lane 0 = I0 real 15 downto 8
    # * lane 1 = I0 real  7 downto 0
    # * lane 2 = I1 real 15 downto 8
    # * lane 3 = I1 real  7 downto 0
    # * lane 4 = Q0 imag 15 downto 8
    # * lane 5 = Q0 imag  7 downto 0
    # * lane 6 = Q1 imag 15 downto 8
    # * lane 7 = Q1  imag  7 downto 0
    for ct in range(0, N-8, 8):
        #Lane 0 I0 real MSB 15 downto 8
        packed_wf[ct] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_re[ct + 2*x] >> 8, x) for x in range(4)), dtype=np.uint32))
        #Lane 1 I0 real MSB 7 downto 0
        packed_wf[ct+1] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_re[ct + 2*x], x) for x in range(4)), dtype=np.uint32))
        #Lane 2 I1 real MSB 15 downto 8
        packed_wf[ct+2] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_re[ct+1 + 2*x] >> 8, x) for x in range(4)), dtype=np.uint32))
        #Lane 3 I1 real MSB 7 downto 0
        packed_wf[ct+3] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_re[ct+1 + 2*x], x) for x in range(4)), dtype=np.uint32))
        #Lane 4 I0 imag MSB 15 downto 8
        packed_wf[ct+4] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_im[ct + 2*x] >> 8, x) for x in range(4)), dtype=np.uint32))
        #Lane 5 I0 imag MSB 7 downto 0
        packed_wf[ct+5] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_im[ct + 2*x], x) for x in range(4)), dtype=np.uint32))
        #Lane 6 I1 imag MSB 15 downto 8
        packed_wf[ct+6] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_im[ct+1 + 2*x] >> 8, x) for x in range(4)), dtype=np.uint32))
        #Lane 7 I1 imag MSB 7 downto 0
        packed_wf[ct+7] = np.bitwise_or.reduce(np.fromiter((pack_byte(wf_im[ct+1 + 2*x], x) for x in range(4)), dtype=np.uint32))

    return packed_wf


class KCU105(object):

    PORT = 0xbb4e #BBN, of course

    def __init__(self):
        pass

    def __del__(self):
        if self.connected:
            self.connected = False
            self.socket.close()

    def _check_connected(self):
        if not self.connected:
            raise IOError("KCU105 Board is not connected!")

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
        ans = self.socket.recv(size)
        while True:
            if len(ans) == size:
                break
            ans += self.socket.recv(8)
        data = [x[0] for x in iter_unpack("!I", ans)]
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
            datagram =  [cmd + ct, addr]
            datagram.extend(data[idx:idx+ct])
            self.send_bytes(datagram)
            datagrams_written += 1
            idx += ct
            addr += ct*4
        #read back and check
        results = self.recv_bytes(2 * 4 * datagrams_written)
        addr = init_addr
        for ct in range(datagrams_written):
            if ct+1 == datagrams_written:
                words_written = len(data)-((datagrams_written-1) * max_ct)
            else:
                words_written = max_ct
            #print("Wrote {} words in {} datagrams: {}".format(words_written, datagrams_written, [hex(_) for _ in results]))
            assert (results[2*ct] == 0x80800000 + words_written)
            assert (results[2*ct+1] == addr)
            addr += 4 * words_written

    def read_memory(self, addr, num_words):
        self._check_connected()
        datagram = [0x10000000 + num_words, addr]
        self.send_bytes(datagram)
        resp_header = self.recv_bytes(2 * 4) #4 bytes per word
        #print([hex(x) for x in resp_header])
        return self.recv_bytes(4 * num_words) #4 bytes per word

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
            markers:
              1m1:
                delay: 0.0

    """

    instrument_type = "AWG"

    def __init__(self, resource_name=None, name="Unlabled APS3"):
        self.name = name
        self.resource_name = resource_name

        self.connected = False
        self.board = KCU105()

        self.dac_clock = 5e9
        self.nco_frequency = 0
        self.dac_mode = "MIX"

        self.fake_seq_file = False

        self.num_rr = 0xFFFF #this sets to board to play in "infinite-loop" mode

    def connect(self, resource_name=None):
        if resource_name is None and self.resource_name is None:
            raise Exception("Must supply a resource name to `connect` if the instrument was initialized without one.")
        elif resource_name is not None:
            self.resource_name = resource_name
        if is_valid_ipv4(self.resource_name):
            self.board.connect(ip_addr = self.resource_name)
            self.connected = True

    def disconnect(self):
        if self.connected:
            self.board.disconnect()

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
        print(filename)
        with h5py.File(filename, "r") as FID:

            target = FID['/'].attrs['target hardware']
            if not (isinstance(target, str) and (target == "APS3")):
                raise IOError("Invalid sequence file!")

            self.num_seq = FID['/'].attrs['num sequences']
            self.marker_delay = 2512 #FID['/'].attrs['marker delay']

            #self.marker_delay = 1600

            wf_lengths = FID['seq_lens'][:]

            if self.num_seq != len(wf_lengths):
                raise ValueError("Sequence file attributes and waveform data out of sync!")

            waves = []
            for ct in range(self.num_seq):
                waves.append(FID['seq_data_{:d}'.format(ct)][:])

        if pack_cython:
            waves = [pack_aps3_waveform_c(wf) for wf in waves]
        else:
            waves = [pack_aps3_waveform(wf) for wf in waves]

        N = self.num_seq
        wf_lengths = np.array([len(wf) for wf in waves], dtype=np.uint32)
        N_pad = ((2*N-1)|15) + 1
        header = np.zeros(N_pad, dtype=np.uint32)
        addr = 4 * N_pad
        wf_addrs = np.array([], dtype=np.uint32)
        for wf in waves:
            wf_addrs = np.append(wf_addrs, np.uint32(addr))
            addr += 4*len(wf)
        header[0:N] = wf_lengths
        header[N:2*N] = wf_addrs
        self.board.write_memory(SDRAM_AXI_ADDR, header)
        addr = 4 * N_pad
        for wf in waves:
            self.board.write_memory(SDRAM_AXI_ADDR + addr, wf)
            addr += 4*len(wf)

        self.setup_waveform()

        logger.info("Waveforms loaded.")

        sleep(0.1)

    @property
    def fake_seq_file(self):
        return self._fake_seq_file
    @fake_seq_file.setter
    def fake_seq_file(self, load):
        self._fake_seq_file = bool(load)


    @property
    def seq_file(self):
        return self._sequence_filename
    @seq_file.setter
    def seq_file(self, filename):
        self._sequence_filename = filename
        if not self._fake_seq_file:
            self._load_waves_from_file(filename)
        else:
            print("Didn't actually load a sequence file...")

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
        self._dac_clock = clock

    @property
    def dac_mode(self):
        return self._dac_mode
    @dac_mode.setter
    def dac_mode(self, mode):
        if mode.upper() not in ("MIX", "NRZ", "RZ"):
            raise ValueError("Unrecognized AD9164 DAC mode.")
        self._dac_mode = mode

    def _get_nco_params(self, freq, print=False):
        if freq < self.dac_clock / 2**48:
            X = 1
            A = 0
            B = 1
        else:
            mod_frac = fractions.Fraction(int(freq), int(self.dac_clock))
            M = mod_frac.numerator
            N = mod_frac.denominator
            X = int(2**48 / N)
            Y = M*2**48 - X*N
            A = fractions.Fraction(Y, N).numerator
            B = fractions.Fraction(Y, N).denominator

        assert B > 0
        assert A < B

        if print:
            print("NCO switching not yet implemented in Auspex... ")
            print("NCO FTW: {}".format(hex(X)))
            print("NCO Modulus A: {}".format(A))
            print("NCO Modulus B: {}".format(B))

    @property
    def nco_frequency(self):
        return self._nco
    @nco_frequency.setter
    def nco_frequency(self, freq):
        self._nco = freq

    @property
    def trigger_interval(self):
        num_clcks = self.board.read_memory(CSR_AXI_ADDR + CSR_TRIGGER_INTERVAL_OFFSET, 1)[0]
        return int(num_clcks) / self.dac_clock
    @trigger_interval.setter
    def trigger_interval(self, interval):
        num_clcks = np.uint32(interval * self.dac_clock)
        self.board.write_memory(CSR_AXI_ADDR + CSR_TRIGGER_INTERVAL_OFFSET, [num_clcks - 0x2])

    def get_csr_value(self):
        return self.board.read_memory(CSR_AXI_ADDR + CSR_CONTROL_OFFSET, 1)[0]

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
