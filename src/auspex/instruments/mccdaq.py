# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['USB3105']

import socket
import time
import copy
import re
import numpy as np
import ctypes
from numbers import Number
from .instrument import Command, StringCommand, FloatCommand, IntCommand, is_valid_ipv4, Instrument
from auspex.log import logger
from mcculw import ul
from mcculw.enums import BoardInfo, InfoType, ULRange, ErrorCode, ScanOptions, FunctionType, DigitalPortType, DigitalIODirection

# Values 0-31 work for digital triggers. AO channel values are assumed to match
channel_map =   {'C1B':0,
                 'C2B':1,
                 'C3B':2,
                 'C4B':3,
                 'R1B':4,
                 'R2B':5,
                 'R3B':6,
                 'R4B':7,
                 'C1T':8,
                 'C2T':9,
                 'C3T':10,
                 'C4T':11,
                 'R1T':12,
                 'R2T':13,
                 'R3T':14,
                 'R4T':15,
                 'E1':16,
                 'E2':17,
                 'E3':18,
                 'E4':19,
                 'E5':20,
                 'E6':21,
                 'E7':22,
                 'E8':23,
                 'C1R':24,
                 'C2R':25,
                 'C3R':26,
                 'C4R':27,
                 'EXT':28,
                 'AI_CLK':29,
                 'R_USER':30,
                 'R_STD':31,
                 }
trigger_map = {}
for label,chan in channel_map.items():
    trigger_map[label] = 2**chan

class USB3105(Instrument):
    """USB-3105 AO"""
    def __init__(self, resource_name=None, *args, **kwargs):
        super(USB3105, self).__init__(*args, **kwargs)
        self.name = "MCC USB-3105 AO"
        self.resource_name = resource_name

    def set_all(self,arr):
        for idx in range(16):
            v = arr[idx]
            v_bin = int(((v+10)/20)*(2**16-1))
            ul.a_out(self.resource_name, idx, ULRange.BIP10VOLTS, v_bin)

    def set_selected(self,chs, vs):
        if isinstance(vs, Number):
            v = vs
            for ch in chs:
                if isinstance(ch, str):
                    ch = channel_map[ch]
                v_bin = int(((v+10)/20)*(2**16-1))
                ul.a_out(self.resource_name, ch, ULRange.BIP10VOLTS, v_bin)
        else:
            for ch, v in zip(chs, vs):
                if isinstance(ch, str):
                    ch = channel_map[ch]
                v_bin = int(((v+10)/20)*(2**16-1))
                ul.a_out(self.resource_name, ch, ULRange.BIP10VOLTS, v_bin)

    def set_one(self,ch,val):
        if isinstance(ch, str):
            ch = channel_map[ch]
        v_bin = int(((val+10)/20)*(2**16-1))
        ul.a_out(self.resource_name, ch, ULRange.BIP10VOLTS, v_bin)

    def zero_all(self):
        v_bin = int(0.5*(2**16-1))
        for ch in range(16):
            ul.a_out(self.resource_name, ch, ULRange.BIP10VOLTS, v_bin)

class USB1608GX(Instrument):
    def __init__(self, resource_name=None, *args, **kwargs):
        super(USB1608GX, self).__init__(*args, **kwargs)
        self.name = "MCC USB-1608GX AI"
        self.resource_name = resource_name

    def read_all_clocked(self, samples, background = False):
        chans = 16
        total_count = samples*chans
        rate = 100e6
        scan_options = ScanOptions.SCALEDATA
        scan_options = scan_options | ScanOptions.EXTCLOCK
        if background:
            scan_options = scan_options | ScanOptions.BACKGROUND
        #scan_options |= ScanOptions.BACKGROUND
        #memhandle = ul.scaled_win_buf_alloc(total_count)
        # = ctypes.cast(memhandle, ctypes.POINTER(ctypes.c_double))
        data_py = np.empty(total_count, np.double)
        ul.a_in_scan(self.resource_name, 0, 15, int(total_count), int(rate), ULRange.BIP10VOLTS, data_py.ctypes, scan_options)

        return(data_py.reshape(samples,chans))

    def read_selected_clocked(self, chs, samples, background=False):
        chans = len(chs)
        total_count = samples*chans
        rate = 100e6
        scan_options = ScanOptions.SCALEDATA
        scan_options = scan_options | ScanOptions.EXTCLOCK
        if background:
            scan_options = scan_options | ScanOptions.BACKGROUND
        #scan_options = scan_options | ScanOptions.EXTTRIGGER
        #scan_options |= ScanOptions.BACKGROUND
        #memhandle = ul.scaled_win_buf_alloc(total_count)
        # = ctypes.cast(memhandle, ctypes.POINTER(ctypes.c_double))
        chs_num = list(chs)
        for idx, ch in enumerate(chs):
            if isinstance(ch, str):
                chs_num[idx] = channel_map[ch]
        data_py = np.zeros(total_count, np.double)
        if chs_num != list(range(min(chs_num), max(chs_num)+1)):
            raise InstrumentError('Only continuous channels are allowed')
        ul.a_in_scan(self.resource_name, min(chs_num), max(chs_num), int(total_count), int(rate), ULRange.BIP10VOLTS, data_py.ctypes, scan_options)

        return(data_py.reshape(samples,chans))

    def read_one_clocked(self, ch, samples):
        if isinstance(ch, str):
            ch = channel_map[ch]
        rate = 100e6
        scan_options = ScanOptions.SCALEDATA
        scan_options = scan_options | ScanOptions.EXTCLOCK
        #scan_options = scan_options | ScanOptions.EXTTRIGGER
        #scan_options |= ScanOptions.BACKGROUND
        #memhandle = ul.scaled_win_buf_alloc(total_count)
        # = ctypes.cast(memhandle, ctypes.POINTER(ctypes.c_double))
        data_py = np.empty(samples, np.double)
        memh = data_py.ctypes
        ul.a_in_scan(self.resource_name, int(ch), int(ch), int(samples), int(rate), ULRange.BIP10VOLTS, memh, scan_options)

        return(data_py)

    def stop(self):
        option = FunctionType.AIFUNCTION
        ul.stop_background(self.resource_name, option)

    def sample_all(self, samples):
        chans = 16
        total_count = chans
        rate = 10e3
        scan_options = ScanOptions.SCALEDATA
        #scan_options = scan_options | ScanOptions.EXTCLOCK
        #scan_options = scan_options | ScanOptions.EXTTRIGGER
        #scan_options |= ScanOptions.BACKGROUND
        #memhandle = ul.scaled_win_buf_alloc(total_count)
        #data_py = ctypes.cast(memhandle, ctypes.POINTER(ctypes.c_double))
        data_py = np.empty(total_count, np.double)
        memh = data_py.ctypes
        ul.a_in_scan(self.resource_name, 0, 15, total_count, int(rate), ULRange.BIP10VOLTS, memh, scan_options)
        return(data_py)

class USBDIO32HS(Instrument):
    def __init__(self, resource_name=None, *args, **kwargs):
        super(USBDIO32HS, self).__init__(*args, **kwargs)
        self.name = "MCC USB-DIO32HS DO"
        self.resource_name = resource_name
        ul.d_config_port(self.resource_name, DigitalPortType.AUXPORT0, DigitalIODirection.OUT)
        ul.d_config_port(self.resource_name, DigitalPortType.AUXPORT1, DigitalIODirection.OUT)

    def stream_32bit(self, arr, rate, continuous = False, background = True):
        #memh = ul.win_buf_alloc_32(len(arr))
        #mem_py = ctypes.cast(memh, ctypes.POINTER(ctypes.c_ulong))
        mem_py = np.ascontiguousarray(arr, np.uint32)
        #mem_py[:] = arr
        options = int(0)
        options = options | ScanOptions.DWORDXFER
        if background:
            options = options | ScanOptions.BACKGROUND
        if continuous:
            options = options | ScanOptions.CONTINUOUS

        ul.d_out_scan(self.resource_name, DigitalPortType.AUXPORT0, len(arr), int(rate), mem_py.ctypes, options)

    def output_32(self, val):
        memh = ul.win_buf_alloc_32(1)
        mem_py = ctypes.cast(memh, ctypes.POINTER(ctypes.c_ulong))
        mem_py[0] = val
        options = int(0)
        options = options | ScanOptions.DWORDXFER
        ul.d_out_scan(self.resource_name, DigitalPortType.AUXPORT0, 1, 1000000, memh, options)

    def dmm_test(self, rate=1):
        memh = ul.win_buf_alloc_32(4)
        mem_py = ctypes.cast(memh, ctypes.POINTER(ctypes.c_ulong))
        mem_py[0] = 0
        mem_py[1] = 0
        mem_py[2] = 2**32-1
        mem_py[3] = 2**32-1

        options = int(0)
        options = options | ScanOptions.DWORDXFER
        options = options | ScanOptions.CONTINUOUS
        options = options | ScanOptions.BACKGROUND
        ul.d_out_scan(self.resource_name, DigitalPortType.AUXPORT0, 4, int(rate), memh, options)

    def bitwise_test(self, rate=1):
        seq_length = 32;
        memh = ul.win_buf_alloc_32(seq_length)
        mem_py = ctypes.cast(memh, ctypes.POINTER(ctypes.c_ulong))
        for idx in np.arange(seq_length):
            mem_py[idx] = 2**idx

        options = int(0)
        options = options | ScanOptions.DWORDXFER
        options = options | ScanOptions.CONTINUOUS
        options = options | ScanOptions.BACKGROUND
        ul.d_out_scan(self.resource_name, DigitalPortType.AUXPORT0, seq_length, int(rate), memh, options)

    def stop(self):
        option = FunctionType.DOFUNCTION
        ul.stop_background(self.resource_name, option)

class driver_test(object):
    def __init__(self):
        self.ao = USB3105(resource_name=-1)
        self.ai = USB1608GX(resource_name=-1)
        self.do = USBDIO32HS(resource_name=-1)

        self.do.zero_all()
        self.ao.zero_all()

        self.rate = 10000

    def test_driver(self, bchan, tchan):

        sequence = [('ALLZERO', 0, 3)
                    (bchan, 1, 1),
                    ('READ', 1, 1),
                    ('READ', 0, 1),
                    (tchan, 1, 1),
                    ('READ', 1, 1)
                    ('READ', 0, 1)]

def parse_triggers(sequence, repeats = 1):
    total_cycles = 1
    for command in sequence:
        total_cycles += command[2]
    seq_int = np.zeros(total_cycles, np.uint32)
    idx = 1
    for command in sequence:
        action = command[0]
        value = command[1]
        cycles = command[2]
        if action == 'ALLZERO':
            seq_int[idx:idx+cycles] = 0
        elif action == 'HOLD':
            seq_int[idx:idx+cycles] = seq_int[idx-1]
        elif isinstance(action, (tuple, list)):
            for item in action:
                if isinstance(item, str):
                    bin = np.uint32(trigger_map[item])
                else:
                    bin = np.uint32(2**item)
                if value:
                    seq_int[idx:] = seq_int[idx] | bin
                else:
                    seq_int[idx:] = seq_int[idx] & ~bin
        elif value:
            if isinstance(action, str):
                bin = np.uint32(trigger_map[action])
            else:
                bin = np.uint32(2**action)
            seq_int[idx:] = seq_int[idx] | bin
        else:
            if isinstance(action, str):
                bin = np.uint32(trigger_map[action])
            else:
                bin = np.uint32(2**action)
            seq_int[idx:] = seq_int[idx] & ~bin
        idx = idx + cycles
    seq_int = np.tile(seq_int, repeats)
    return seq_int
