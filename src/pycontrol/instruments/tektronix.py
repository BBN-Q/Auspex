from .instrument import Instrument, Command, FloatCommand, IntCommand
import numpy as np

class DPO72004C(Instrument):
    """Tektronix DPO72004C Oscilloscope"""
    encoding = Command("Data Encoding", get_string="DAT:ENC;", set_string="DAT:ENC {:s};",
                        allowed_values=["ASCI","RIB","RPB","FPB","SRI","SRP","SFP"])
    bit_depth = IntCommand("Bit Depth", get_string="WFMOutpre:BYT_Nr?;",
                            set_string="WFMOutpre:BYT_Nr {:d};", allowed_values=[1,2,4,8])
    data_start = IntCommand("Data Start", get_string="DAT:STAR?;", set_string="DAT:STAR {:d};")
    data_stop  = IntCommand("Data Stop", get_string="DAT:STOP?;", set_string="DAT:STOP {:d};")

    # Fast Frames
    fast_frame = Command("Fast Frame State", get_string="HORizontal:FASTframe:STATE?;", set_string="HORizontal:FASTframe:STATE {:s};",
                         value_map={True: '1', False: '0'})
    num_fast_frames = IntCommand("Number of fast frames", get_string="HOR:FAST:COUN?;", set_string="HOR:FAST:COUN {:d};")

    preamble   = Command("Curve preamble ", get_string="WFMOutpre?;")
    record_length = IntCommand("Record Length", get_string="HOR:ACQLENGTH?;")

    def __init__(self, name, resource_name, *args, **kwargs):
        resource_name += "::4000::SOCKET" #user guide recommends HiSLIP protocol
        super(DPO72004C, self).__init__(name, resource_name, *args, **kwargs)
        self.interface._resource.read_termination = u"\n"

    def snap(self):
        """Sets the start and stop points to the the current front panel display.
        This doesn't actually seem to work, strangely."""
        self.interface.write("DAT SNAp;")

    def get_curve(self, channel=1):
        channel_string = "CH{:d}".format(channel)
        self.interface.write("DAT:SOU {:s};".format(channel_string))
        self.source_channel = 1
        self.encoding = "SRI" # Signed ints
        self.bit_depth  = 1
        record_length = self.record_length
        self.data_start = 1
        self.data_stop  = record_length

        curve = self.interface.query_binary_values("CURVe?;", datatype='b')
        scale = self.interface.value('WFMO:YMU?;')
        offset = self.interface.value('WFMO:YOF?;')
        curve = (curve - offset)*scale
        if self.fast_frame:
            curve.resize((self.num_fast_frames, record_length))
        return curve

    def get_fastaq_curve(self, channel=1):
        channel_string = "CH{:d}".format(channel)
        self.interface.write("DAT:SOU {:s};".format(channel_string))
        self.source_channel = 1
        self.encoding = "SRP" # Unsigned ints
        self.bit_depth  = 8
        self.data_start = 1
        self.data_stop  = self.record_length
        curve = self.interface.query_binary_values("CURVe?;", datatype='Q').reshape((1000,252))
        return curve

    def get_math_curve(self, channel=1):
        pass
