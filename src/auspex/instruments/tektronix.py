# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['TekDPO72004C','TekAWG5014']

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, is_valid_ipv4
import numpy as np

class TekDPO72004C(SCPIInstrument):
    """Tektronix DPO72004C Oscilloscope"""
    encoding   = StringCommand(get_string="DAT:ENC?;", set_string="DAT:ENC {:s};",
                        allowed_values=["ASCI","RIB","RPB","FPB","SRI","SRP","SFP"])
    source_channel = IntCommand(get_string="DAT:SOU?;", set_string="DAT:SOU {:s};",value_map={1:"CH1",2:"CH2",3:"CH3",4:"CH4"})
    byte_depth = IntCommand(get_string="WFMOutpre:BYT_Nr?;",
                            set_string="WFMOutpre:BYT_Nr {:d};", allowed_values=[1,2,4,8])
    data_start = IntCommand(get_string="DAT:STAR?;", set_string="DAT:STAR {:d};")
    data_stop  = IntCommand(get_string="DAT:STOP?;", set_string="DAT:STOP {:d};")

    fast_frame      = StringCommand(get_string="HORizontal:FASTframe:STATE?;", set_string="HORizontal:FASTframe:STATE {:s};",
                       value_map       = {True: '1', False: '0'})
    num_fast_frames = IntCommand(get_string="HOR:FAST:COUN?;", set_string="HOR:FAST:COUN {:d};")

    preamble = StringCommand(get_string="WFMOutpre?;") # Curve preamble

    record_length   = IntCommand(get_string="HOR:ACQLENGTH?;")
    record_duration = FloatCommand(get_string="HOR:ACQDURATION?;")

    def __init__(self, resource_name=None, *args, **kwargs):
        # resource_name += "::4000::SOCKET" #user guide recommends HiSLIP protocol
        # super(DPO72004C, self).__init__(resource_name, *args, **kwargs)
        # self.name = "Tektronix DPO72004C Oscilloscope"
        # self.interface._resource.read_termination = u"\n"
        super(TekDPO72004C, self).__init__(resource_name, *args, **kwargs)
        self.name = "Tektronix DPO72004C Oscilloscope"

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(self.resource_name):
            self.resource_name += "::4000::SOCKET" #user guide recommends HiSLIP protocol
        super(TekDPO72004C, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n" 

    def clear(self):
        self.interface.write("CLEAR ALL;")

    def snap(self):
        """Sets the start and stop points to the the current front panel display.
        This doesn't actually seem to work, strangely."""
        self.interface.write("DAT SNAp;")

    def get_trace(self, channel=1, byte_depth=2):
        self.source_channel = channel
        self.encoding = "SRI" # Signed ints

        record_length = self.record_length
        self.data_start = 1
        self.data_stop  = record_length

        self.byte_depth = byte_depth
        strf_from_depth = {1: 'b', 2: 'h', 4: 'l', 8: 'q'}

        vals = self.interface.query_binary_values("CURVe?;", datatype=strf_from_depth[byte_depth])
        scale = self.interface.value('WFMO:YMU?;')
        offset = self.interface.value('WFMO:YOF?;')
        vals = (vals - offset)*scale
        vals = vals.reshape((vals.size,))
        time = np.linspace(0, self.record_duration, self.record_length)
        # if self.fast_frame:
        #     vals.resize((self.num_fast_frames, record_length))
        return (time,vals)

    def get_fastaq_curve(self, channel=1):
        self.source_channel = channel
        self.encoding = "SRP" # Unsigned ints
        self.byte_depth  = 8
        self.data_start = 1
        self.data_stop  = self.record_length
        curve = self.interface.query_binary_values("CURVe?;", datatype='Q').reshape((1000,252))
        return curve

    def get_math_curve(self, channel=1):
        pass



class TekAWG5014(SCPIInstrument):
    """Tektronix AWG 5014"""

    CHANNEL = 1 # Default Channel 

    def __init__(self, resource_name=None, *args, **kwargs):
        super(TekAWG5014, self).__init__(resource_name, *args, **kwargs)
        self.name = "Tektronix AWG 5014"

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(self.resource_name):
            self.resource_name += "::4000::SOCKET" #LAN must be enabled and Port must be defined in GPIB/LAN Configuration on instrument
        super(TekAWG5014, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\r" 
        self.interface._resource.write_termination = u"\n"

    # Select Active Channel

    @property
    def channel(self):
        return self.CHANNEL

    @channel.setter
    def channel(self, channel=1):
        if channel not in range(1,self.interface.query_ascii_values("AWGCONTROL:CONFIGURE:CNUMBER?",converter=u'd')[0]+1):
            raise ValueError("Channel must be integer between 1 and {}".format(self.interface.query_ascii_values("AWGCONTROL:CONFIGURE:CNUMBER?",converter=u'd')[0]))
        else:
             self.CHANNEL = channel

    # Channel Amplitude

    @property
    def amplitude(self):

        query_str = "SOURCE{:d}:VOLTAGE:AMPLITUDE?".format(self.CHANNEL)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @amplitude.setter
    def amplitude(self, val=2e-2):
 
        if (val>2) or (val<2e-2): 
            raise ValueError("Amplitude must be between 0.02 and 2 Volts pk-pk.")
        else:
            self.interface.write(("SOURCE{:d}:VOLTAGE:AMPLITUDE {:E}".format(self.CHANNEL,val)))

    # Channel Offset
    @property
    def offset(self):

        query_str = "SOURCE{:d}:VOLTAGE:OFFSET?".format(self.CHANNEL)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @offset.setter
    def offset(self, val):

        self.interface.write(("SOURCE{:d}:VOLTAGE:OFFSET {:E}".format(self.CHANNEL,val)))

    # Channel High Voltage

    @property
    def high(self):

        query_str = "SOURCE{:d}:VOLTAGE:HIGH?".format(self.CHANNEL)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @high.setter
    def high(self, val):
 
        self.interface.write(("SOURCE{:d}:VOLTAGE:HIGH {:E}".format(self.CHANNEL,val)))

    # Channel Low Voltage

    @property
    def low(self):

        query_str = "SOURCE{:d}:VOLTAGE:LOW?".format(self.CHANNEL)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @low.setter
    def low(self, val):
 
        self.interface.write(("SOURCE{:d}:VOLTAGE:LOW {:E}".format(self.CHANNEL,val)))