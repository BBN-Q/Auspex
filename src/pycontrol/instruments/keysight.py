from .instrument import Instrument, Command, FloatCommand, IntCommand
from .binutils import BitField, BitFieldUnion

import logging
import warnings
import numpy as np
import h5py

class SequenceControlWord(metaclass=BitFieldUnion):
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

    fmt_str = "STAB{:d}:DATA {{:d}}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}"

    def __init__(self, segment_id, loop_ct=1, marker_enable=True):
        super(WaveformEntry, self).__init__()
        self.segment_id = segment_id
        self.loop_ct    = loop_ct

class IdleEntry(object):
    """Idle entry for sequence table"""

    fmt_str = "STAB{:d}:DATA {{:d}}, {:d}, {:d}, 0, {:d}, {:d}, 0"

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
#

class SequenceControlWord(metaclass=BitFieldUnion):
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

class Sequence(object):
    """Bundle of sequence table entries as a "sequence" """
    def __init__(self, channel=1, sequence_loop_ct=1):
        super(Sequence, self).__init__()
        self.channel = channel
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
            control_word = SequenceControlWord()
            if ct == 0:
                control_word.init_marker_sequence = 1
            elif ct == len(self.sequence_items)-1:
                control_word.end_marker_sequence = 1
            if isinstance(entry, WaveformEntry):
                control_word.marker_enable = 1
                scpi_str = entry.fmt_str.format(self.channel, control_word.packed, self.sequence_loop_ct if ct==0 else 0, entry.loop_ct, entry.segment_id, 0, 0xffffffff)
            elif isinstance(entry, IdleEntry):
                control_word.data_cmd_sel = 1
                scpi_str = entry.fmt_str.format(self.channel, control_word.packed, self.sequence_loop_ct if ct==0 else 0, entry.dac_level(), entry.length)
            else:
                raise TypeError("Unhandled sequence entry type")

            scpi_strs.append(scpi_str)
        return scpi_strs

class Scenario(object):
    """Bundle of sequences as a "scenario" """
    def __init__(self):
        super(Scenario, self).__init__()
        self.sequences = []

    def scpi_strings(self):
        scpi_strings = []
        #Extract SCPI strings from sequences
        for seq in self.sequences:
            scpi_strings.extend(seq.scpi_strings())
        #interpolate in table indcies
        scpi_strings = [s.format(ct) for ct,s in enumerate(scpi_strings)]
        #add end scenario flag
        last_control_word_str = scpi_strings[-1].split(',')[1]
        last_control_word = SequenceControlWord()
        last_control_word.packed = int(last_control_word_str)
        last_control_word.end_marker_scenario = 1
        scpi_strings[-1] = scpi_strings[-1].replace(last_control_word_str, str(last_control_word.packed))

        return scpi_strings


class M8190A(Instrument):
    """M8190A arbitrary waveform generator"""

    ref_source         = Command("reference source", scpi_string=":ROSC:SOUR",
        allowed_values=("EXTERNAL", "AXI", "INTERNAL"))
    ref_source_freq    = FloatCommand("reference source frequency", scpi_string=":ROSC:FREQ", value_range=(1e6, 200e6))
    sample_freq        = FloatCommand("internal sample frequency", scpi_string=":FREQ:RAST")
    sample_freq_ext    = FloatCommand("external sample frequency", scpi_string=":FREQ:RAST:EXT")
    sample_freq_source = Command("sample frequency source", scpi_string=":FREQ:RAST:SOUR",
        allowed_values=("INTERNAL", "EXTERNAL"))
    waveform_output_mode = Command("waveform output mode", scpi_string=":TRAC:DWID",
        allowed_values=("WSPEED", "WPRECISION", "INTX3", "INTX12", "INTX24", "INT48"))
    output = Command("Channel output", scpi_string=":OUTP{channel:s}:NORM",
        value_map={False:"0", True:"1"}, additional_args=['channel'])

    def __init__(self, name, resource_name, *args, **kwargs):
        resource_name += "::inst0::INSTR" #user guide recommends HiSLIP protocol
        super(M8190A, self).__init__(name, resource_name, *args, **kwargs)
        self.interface._resource.read_termination = u"\n"

    def abort(self, channel=None):
        """Abort/stop signal generation on a channel"""
        if channel is None:
            self.interface.write(":ABORT")
        else:
            self.interface.write(":ABORT{:d}")

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
                raise ValueError("M8190A did not return a reasonable segment ID, but rather {}".format(r))
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
        print("Returned segment id {}".format(segment_id))
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

    def reset_sequence_table(self):
        self.interface.write(":STAB:RES")

    def upload_scenario(self, scenario):
        strs = scenario.scpi_strings()
        for s in strs:
            self.interface.write(s)
