from .instrument import Instrument, Command, FloatCommand, IntCommand

import warnings
import numpy as np

class M8190A(Instrument):
    """M8190A arbitrary waveform generator"""

    ref_source = Command("reference source", get_string=":ROSC:SOUR?", set_string="ROSC:SOUR {:s}", allowed_values=("EXTERNAL", "AXI", "INTERNAL"))
    ref_source_freq = FloatCommand("reference source frequency", get_string=":ROSC:FREQ?", set_string=":ROSC:FREQ {:E}",value_range=(1e6, 200e6))
    sample_freq = FloatCommand("internal sample frequency", get_string=":FREQ:RAST?", set_string=":FREQ:RAST {:E}")
    sample_freq_ext = FloatCommand("external sample frequency", get_string=":FREQ:RAST:EXT?", set_string=":FREQ:RAST:EXT {:E}")
    sample_freq_source = Command("sample frequency source", get_string=":FREQ:RAST:SOUR?", set_string=":FREQ:RAST:SOUR", allowed_values=("INTERNAL", "EXTERNAL"))

    def __init__(self, name, resource_name, *args, **kwargs):
        resource_name += "::hislip0::INSTR" #user guide recommends HiSLIP protocol
        super(M8190A, self).__init__(name, resource_name, *args, **kwargs)
        self.interface._resource.read_termination = u"\n"

    def abort(self, channel):
        """Abort/stop signal generation on a channel"""
        self.interface.write("ABORT{:d}")

    def get_ref_source_available(self, source):
        """Checks whether reference source is available"""
        allowed_values = ("EXTERNAL", "AXI", "INTERNAL")
        if source not in allowed_values:
            raise ValueError("reference source must be one of {:s}".format(str(allowed_values)))
        return self.interface.query(":ROSC:SOUR:CHEC? {:s}".format(source)) == '1'

    @staticmethod
    def create_binary_wf_data(wf, sync_mkr, samp_mkr, vertical_resolution=12):
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
        bin_data = np.int16(scale_factor*wf)

        #clip if necessary
        if np.max(bin_data) > scale_factor-1 or np.min(bin_data) < -scale_factor:
            warnings.warn("Clipping waveform")
            bin_data = np.clip(bin_data, -scale_factor, scale_factor-1)

        #shift up to the MSB
        bin_data = np.left_shift(bin_data, 4 if vertical_resolution == 12 else 2)

        #add in the marker bits
        bin_data = np.bitwise_or(bin_data, np.bitwise_or(np.left_shift(np.bitwise_and(sync_mkr, 0x1), 1), np.bitwise_and(samp_mkr, 0x1)))

        return bin_data
