# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['DPO72004C']

from auspex.log import logger
from .instrument import SCPIInstrument, Command, StringCommand, BoolCommand, FloatCommand, IntCommand, is_valid_ipv4
import numpy as np

class DPO72004C(SCPIInstrument):
    """Tektronix DPO72004C Oscilloscope"""
    encoding   = StringCommand(get_string="DAT:ENC;", set_string="DAT:ENC {:s};",
                        allowed_values=["ASCI","RIB","RPB","FPB","SRI","SRP","SFP"])
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

    button_press = StringCommand(set_string="FPAnel:PRESS {:s};",
        allowed_values=["RUnstop", "SINGleseq"])

    def __init__(self, resource_name, *args, **kwargs):
        resource_name += "::4000::SOCKET" #user guide recommends HiSLIP protocol
        super(DPO72004C, self).__init__(resource_name, *args, **kwargs)
        self.name = "Tektronix DPO72004C Oscilloscope"

    def clear(self):
        self.interface.write("CLEAR ALL;")

    def snap(self):
        """Sets the start and stop points to the the current front panel display.
        This doesn't actually seem to work, strangely."""
        self.interface.write("DAT SNAp;")

    def get_curve(self, channel=1, byte_depth=2):
        channel_string = "CH{:d}".format(channel)
        self.interface.write("DAT:SOU {:s};".format(channel_string))
        #self.source_channel = 1
        self.encoding = "SRI" # Signed ints

        record_length = self.record_length
        self.data_start = 1
        self.data_stop  = record_length

        self.byte_depth = byte_depth
        strf_from_depth = {1: 'b', 2: 'h', 4: 'l', 8: 'q'}

        curve = self.interface.query_binary_values("CURVe?;", datatype=strf_from_depth[byte_depth])
        scale = self.interface.value('WFMO:YMU?;')
        offset = self.interface.value('WFMO:YOF?;')
        curve = (curve - offset)*scale
        if self.fast_frame:
            curve.resize((self.num_fast_frames, record_length))
        return curve

    def get_timebase(self):
        return np.linspace(0, self.record_duration, self.record_length)

    def get_fastaq_curve(self, channel=1):
        channel_string = "CH{:d}".format(channel)
        self.interface.write("DAT:SOU {:s};".format(channel_string))
        self.source_channel = 1
        self.encoding = "SRP" # Unsigned ints
        self.byte_depth  = 8
        self.data_start = 1
        self.data_stop  = self.record_length
        curve = self.interface.query_binary_values("CURVe?;", datatype='Q').reshape((1000,252))
        return curve

    def get_math_curve(self, channel=1):
        pass

class RSA3308A(SCPIInstrument):
    """Tektronix RSA3308A SA"""
    instrument_type = "Spectrum Analyzer"

    frequency_center = FloatCommand(scpi_string=":FREQuency:CENTer")
    frequency_span   = FloatCommand(scpi_string=":FREQuency:SPAN")
    frequency_start  = FloatCommand(scpi_string=":FREQuency:STARt")
    frequency_stop   = FloatCommand(scpi_string=":FREQuency:STOP")

    num_sweep_points = FloatCommand(scpi_string=":SWEep:POINTs")
    resolution_bandwidth = FloatCommand(scpi_string=":BANDwidth")
    sweep_time = FloatCommand(scpi_string=":SWEep:TIME")
    averaging_count = IntCommand(scpi_string=':AVER:COUN')

    marker1_amplitude = FloatCommand(scpi_string=':CALC:MARK1:Y')
    marker1_position = FloatCommand(scpi_string=':CALC:MARK1:X')

    mode = StringCommand(scpi_string=":INSTrument", allowed_values=["SA", "BASIC", "PULSE", "PNOISE"])

    # phase noise application commands
    pn_offset_start = FloatCommand(scpi_string=":LPLot:FREQuency:OFFSet:STARt")
    pn_offset_stop  = FloatCommand(scpi_string=":LPLot:FREQuency:OFFSet:STOP")
    pn_carrier_freq = FloatCommand(scpi_string=":FREQuency:CARRier")

    def __init__(self, resource_name=None, *args, **kwargs):
        super(RSA3308A, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(self.resource_name):
            self.resource_name += "::5025::SOCKET"
        super(RSA3308A, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.timeout = 3000 #seem to have trouble timing out on first query sometimes

    def get_axis(self):
        return np.linspace(self.frequency_start, self.frequency_stop, self.num_sweep_points)

    def get_trace(self, num=1):
        self.interface.write(':FORM:DATA REAL,32')
        return self.interface.query_binary_values(":TRACE:DATA? TRACE{:d}".format(num),
            datatype="f", is_big_endian=True)

    def get_pn_trace(self, num=3):
        # num = 3 is raw data
        # num = 4 is smoothed data
        # returns a tuple of (freqs, dBc/Hz)
        self.interface.write(":FORM:DATA ASCII")
        response = self.interface.query(":FETCH:LPLot{:d}?".format(num))
        xypts = np.array([float(x) for x in response.split(',')])
        return xypts[::2], xypts[1::2]

    def restart_sweep(self):
        """ Aborts current sweep and restarts. """
        self.interface.write(":INITiate:RESTart")

    def peak_search(self, marker=1):
        self.interface.write(':CALC:MARK{:d}:MAX'.format(marker))

    def marker_to_center(self, marker=1):
        self.interface.write(':CALC:MARK{:d}:CENT'.format(marker))

    def clear_averaging(self):
        self.interface.write(':AVER:CLE')
