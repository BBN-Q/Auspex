# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['KeysightM8190A', 'Sequence', 'Scenario']

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, Command
from .binutils import BitField, BitFieldUnion

import logging
import warnings
import numpy as np

class SequenceControlWord(BitFieldUnion):
    reserved0             = BitField(12)
    freq_table_increment  = BitField(1)
    freq_table_init       = BitField(1)
    amp_table_increment   = BitField(1)
    amp_table_init        = BitField(1)
    advance_mode_segment  = BitField(4)
    advance_mode_sequence = BitField(4)
    marker_enable         = BitField(1)
    reserved1             = BitField(3)
    init_marker_sequence  = BitField(1)
    end_marker_scenario   = BitField(1)
    end_marker_sequence   = BitField(1)
    data_cmd_sel          = BitField(1)

class WaveformEntry(object):
    """Waveform entry for sequence table"""

    fmt_str = "STAB{{{{:d}}}}:DATA {{:d}}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}"

    def __init__(self, segment_id, loop_ct=1, marker_enable=True):
        super(WaveformEntry, self).__init__()
        self.segment_id = segment_id
        self.loop_ct    = loop_ct

class IdleEntry(object):
    """Idle entry for sequence table"""

    fmt_str = "STAB{{{{:d}}}}:DATA {{:d}}, {:d}, {:d}, 0, {:d}, {:d}, 0"

    def __init__(self, length, amp):
        super(IdleEntry, self).__init__()
        #in WSPEED mode min is 640, in WRPRECISION min is 480
        if length < 640:
            raise ValueError("Minimum idle entry is 640")
        self.length = length
        self.amp    = amp

    def dac_level(self, vertical_resolution=12):
        """Calculate DAC level for amplitude"""
        scale = 2**(vertical_resolution-1)
        dac_level = int(scale * self.amp)
        return np.clip(dac_level, -scale, scale-1)

# class ConfigEntry(object):
#     """Idle object"""
#     def __init__(self, start=False, stop=False):
#         super(ConfigEntry, self).__init__()
#         self.start = start
#         self.stop = stop
#         self.word = SequenceControlWord()
#         if self.start:
#             self.word.init_marker


class Sequence(object):
    """Bundle of sequence table entries as a "sequence" """
    def __init__(self, sequence_loop_ct=1):
        super(Sequence, self).__init__()
        self.sequence_loop_ct = sequence_loop_ct
        self.sequence_items = []

    def add_waveform(self, segment_id, **kwargs):
        self.sequence_items.append(WaveformEntry(segment_id, **kwargs))

    def add_idle(self, length, amp=0):
        self.sequence_items.append(IdleEntry(length, amp))

    def add_command(self):
        pass

    def scpi_strings(self):
        """Returns a list of SCPI strings that can be pushed to instrument after formatting with index"""
        scpi_strs = []
        for ct, entry in enumerate(self.sequence_items):
            control_word = SequenceControlWord(\
                init_marker_sequence = 1 if ct==0 else 0, \
                end_marker_sequence = 1 if ct==(len(self.sequence_items) - 1) else 0, \
                marker_enable = 1 if isinstance(entry, WaveformEntry) else 0, \
                data_cmd_sel = 1 if isinstance(entry, IdleEntry) else 0 \
            )
            if isinstance(entry, WaveformEntry):
                scpi_str = entry.fmt_str.format(control_word.packed, self.sequence_loop_ct if ct==0 else 0, entry.loop_ct, entry.segment_id, 0, 0xffffffff)
            elif isinstance(entry, IdleEntry):
                scpi_str = entry.fmt_str.format(control_word.packed, self.sequence_loop_ct if ct==0 else 0, entry.dac_level(), entry.length)
            else:
                raise TypeError("Unhandled sequence entry type")

            scpi_strs.append(scpi_str)
        return scpi_strs

    def table(self):
        """Return a 2D array of UInt32 representing the sequence"""
        table = np.zeros((len(self.sequence_items), 6), dtype=np.uint32)
        for ct, entry in enumerate(self.sequence_items):
            control_word = SequenceControlWord(\
                init_marker_sequence = 1 if ct==0 else 0, \
                end_marker_sequence = 1 if ct==(len(self.sequence_items) - 1) else 0, \
                marker_enable = 1 if isinstance(entry, WaveformEntry) else 0, \
                data_cmd_sel = 1 if isinstance(entry, IdleEntry) else 0 \
            )
            table[ct,0] = control_word.packed
            table[ct,1] = self.sequence_loop_ct if ct==0 else 0
            if isinstance(entry, WaveformEntry):
                table[ct, 2] = entry.loop_ct
                table[ct, 3] = entry.segment_id
                table[ct, 5] = 0xffffffff
            elif isinstance(entry, IdleEntry):
                table[ct, 3] = entry.dac_level()
                table[ct, 4] = entry.length
            else:
                raise TypeError("Unhandled sequence entry type")

        return table

class Scenario(object):
    """Bundle of sequences as a "scenario" """
    def __init__(self):
        super(Scenario, self).__init__()
        self.sequences = []

    def scpi_strings(self, start_idx=0):
        scpi_strs = []
        #Extract SCPI strings from sequences
        for seq in self.sequences:
            scpi_strs.extend(seq.scpi_strings())
        #interpolate in table indices
        scpi_strs = [s.format(ct + start_idx) for ct,s in enumerate(scpi_strs)]
        #add end scenario flag
        last_control_word_str = scpi_strs[-1].split(',')[1]
        last_control_word = SequenceControlWord(packed=int(last_control_word_str))
        last_control_word.end_marker_scenario = 1
        scpi_strs[-1] = scpi_strs[-1].replace(last_control_word_str, str(last_control_word.packed))

        return scpi_strs

    def table(self):
        table = np.empty((0,6), dtype=np.uint32)
        for seq in self.sequences:
            table = np.concatenate((table, seq.table()), axis=0)
        #add end scenario flag
        last_control_word = SequenceControlWord(packed=table[-1, 0])
        last_control_word.end_marker_scenario = 1
        table[-1,0] = last_control_word.packed

        return table

class KeysightM8190A(SCPIInstrument):
    """Keysight M8190A arbitrary waveform generator"""

    ref_source         = StringCommand(scpi_string=":ROSC:SOUR",
                          allowed_values=("EXTERNAL", "AXI", "INTERNAL"))
    ref_source_freq    = FloatCommand(scpi_string=":ROSC:FREQ", value_range=(1e6, 200e6))
    sample_freq        = FloatCommand(scpi_string=":FREQ:RAST")       # internal sample frequency
    sample_freq_ext    = FloatCommand(scpi_string=":FREQ:RAST:EXT")   # external sample frequency
    sample_freq_source = StringCommand(scpi_string=":FREQ:RAST:SOUR", # sample frequency source
                          allowed_values=("INTERNAL", "EXTERNAL"))
    coupled            = Command(scpi_string=":INST:COUP:STAT", value_map={True:"1", False:"0"})

    waveform_output_mode  = StringCommand(scpi_string=":TRAC{channel:d}:DWID", additional_args=['channel'],
                             allowed_values=("WSPEED", "WPRECISION", "INTX3", "INTX12", "INTX24", "INT48"))
    sequence_mode         = StringCommand(scpi_string=":FUNC{channel:d}:MODE", additional_args=['channel'],
                                          value_map={"ARBITRARY":"ARB", "SEQUENCE":"STS", "SCENARIO":"STSC"})
    scenario_loop_ct      = IntCommand(scpi_string=":STAB{channel:d}:SCEN:COUN", additional_args=['channel'])
    scenario_advance_mode = StringCommand(scpi_string=":STAB{channel:d}:SCEN:ADV", additional_args=['channel'],
                             value_map={"AUTOMATIC":"AUTO", "CONDITIONAL":"COND", "REPEAT":"REP", "SINGLE":"SING"})
    scenario_start_index  = IntCommand(scpi_string=":STAB{channel:d}:SCEN:SEL", additional_args=['channel'])

    output            = StringCommand(scpi_string=":OUTP{channel:d}:NORM",
                         value_map={False:"0", True:"1"}, additional_args=['channel'])
    output_complement = StringCommand(scpi_string=":OUTP{channel:d}:COMP",
                         value_map={False:"0", True:"1"}, additional_args=['channel'])
    output_route      = StringCommand(scpi_string=":OUTP{channel:d}:ROUT", allowed_values=["DAC","AC","DC"],
                         additional_args=["channel"])

    voltage_amplitude = FloatCommand(scpi_string=":VOLT{channel:d}:AMPL", additional_args=['channel'])
    #TODO: voltage_amplitude for the different output routes
    voltage_offset    = FloatCommand(scpi_string=":VOLT{channel:d}:OFFS", additional_args=['channel'])

    marker_level_low  = FloatCommand(scpi_string=":MARK{channel:d}:{marker_type:s}:VOLT:LOW",
                         additional_args=["channel", "marker_type"]) # marker (sync/samp) low
    marker_level_high = FloatCommand(scpi_string=":MARK{channel:d}:{marker_type:s}:VOLT:HIGH",
                         additional_args=["channel", "marker_type"]) # marker (sync/samp) high

    continuous_mode = StringCommand(scpi_string=":INIT:CONT:STAT", value_map={True:"1", False:"0"})
    gate_mode       = StringCommand(scpi_string=":INIT:GATE:STAT", value_map={True:"1", False:"0"})

    def __init__(self, resource_name, *args, **kwargs):
        super(KeysightM8190A, self).__init__(resource_name, *args, **kwargs)
        self.name = "KeysightM8190A AWG"

        #Aliases for run/stop
        self._unfreeze()
        self.run = self.initiate
        self.stop = self.abort
        self._freeze()

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        if "::inst0::INSTR" not in self.resource_name: #user guide recommends HiSLIP protocol
            self.resource_name += "::inst0::INSTR"
        super(KeysightM8190A, self).connect(resource_name=resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"

    def abort(self):
        """Abort/stop signal generation on a channel"""
        # Never add channel number, or 't' !!!!!
        self.interface.write(":ABOR")

    def initiate(self, channel=1):
        self.interface.write(":INIT:IMM{:d}".format(channel))

    def get_ref_source_available(self, source):
        """Checks whether reference source is available"""
        allowed_values = ("EXTERNAL", "AXI", "INTERNAL")
        if source not in allowed_values:
            raise ValueError("reference source must be one of {:s}".format(str(allowed_values)))
        return self.interface.query(":ROSC:SOUR:CHEC? {:s}".format(source)) == '1'

    def define_waveform(self, length, segment_id=None, channel=1):
        if segment_id:
            self.interface.write(":TRAC{:d}:DEF {:d},{:d}".format(channel, segment_id, length))
        else:
            r = self.interface.query(":TRAC{:d}:DEF:NEW? {:d}".format(channel, length))
            try:
                segment_id = int(r)
            except:
                raise ValueError("KeysightM8190A did not return a reasonable segment ID, but rather {}".format(r))
        return segment_id

    def upload_waveform(self, wf_data, segment_id, channel=1, binary=True):
        """Uploads the waveform to the device. Technically we should split the data into multiple chunks
        if we exceed the 999999999 Bytes, i.e. 1GB SCPI transfer limit.
        """
        if np.dtype(np.float16).itemsize*len(wf_data) > 999999999:
            raise ValueError("Waveform is too large for single transfer, go improve the upload_waveform() method.")
        offset = 0
        command_string = ":TRAC{:d}:DATA {:d},{:d},".format(channel, segment_id, offset)

        if binary:
            # Explicity set the endianess of the transfer
            # self.interface.write(":FORMat:BORD NORM")
            self.interface.write_binary_values(command_string, wf_data, datatype='h')
        else:
            ascii_string = ",".join(["{:d}".format(val) for val in wf_data])
            self.interface.write(command_string + ascii_string)

    def delete_waveform(self, segment_id, channel=1):
        self.interface.write(":TRAC{:d}:DEL {:d}".format(channel, segment_id))

    def delete_all_waveforms(self, channel=1):
        self.interface.write(":TRAC{:d}:DEL:ALL".format(channel) )

    def select_waveform(self, segment_id, channel=1):
        self.interface.write(":TRAC{:d}:SEL {:d}".format(channel, segment_id))

    def use_waveform(self, wf_data, segment_id=None, channel=1):
        self.abort()
        if segment_id:
            self.delete_waveform(segment_id, channel=channel)
        segment_id = self.define_waveform(len(wf_data), segment_id=segment_id, channel=channel)
        self.upload_waveform(wf_data, segment_id, channel=channel)
        self.select_waveform(segment_id, channel=channel)
        self.initiate(channel=channel)

    @staticmethod
    def create_binary_wf_data(wf, sync_mkr=0, samp_mkr=0, vertical_resolution=12):
        """Given numpy arrays of waveform and marker data convert to binary format.
        Assumes waveform data is np.float in range -1 to 1 and marker data can be cast to bool
        Binary format is waveform in MSB and and markers in LSB
        waveform       sync_mkr samp_mkr
        15 downto 4/2     1      0
        """
        #cast the waveform to integers
        if not((vertical_resolution == 12) or (vertical_resolution == 14)):
            raise ValueError("vertical resolution must be 12 or 14 bits")

        #convert waveform to integers
        scale_factor = 2**(vertical_resolution-1)
        bin_data = np.int16((scale_factor-1)*np.array(wf))

        #clip if necessary
        if np.max(bin_data) > scale_factor-1 or np.min(bin_data) < -scale_factor:
            warnings.warn("Clipping waveform. Max value: {:d} Min value: {:d}. Scale factor: {:d}.".format(np.max(bin_data), np.min(bin_data),scale_factor))
            bin_data = np.clip(bin_data, -scale_factor, scale_factor-1)

        # bin_data = bin_data.byteswap()
        #shift up to the MSB
        bin_data = np.left_shift(bin_data, 4 if vertical_resolution == 12 else 2)

        #add in the marker bits
        bin_data = np.bitwise_or(bin_data, np.bitwise_or(np.left_shift(np.bitwise_and(sync_mkr, 0x1), 1), np.bitwise_and(samp_mkr, 0x1)))

        return bin_data

    def reset_sequence_table(self, channel=1):
        self.interface.write(":STAB{:d}:RESet".format(channel))

    def upload_scenario(self, scenario, channel=1, start_idx=0, binary=True):
        if binary:
            cmd_str = ":STAB{:d}:DATA {:d},".format(channel, start_idx)
            self.interface.write_binary_values(cmd_str, scenario.table().flatten(), datatype="I")
        else:
            strs = scenario.scpi_strings(start_idx=start_idx)
            for s in strs:
                self.interface.write(s.format(channel))

    def advance(self, channel=1):
        self.interface.write(":TRIG:ADV{:d}".format(channel))

    def trigger(self, channel=1):
        self.interface.write(":TRIG:BEG{:d}".format(channel))
