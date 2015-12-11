from .instrument import Instrument, Command, FloatCommand, IntCommand

import logging
import warnings
import numpy as np

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
