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

    MEAS = 1 # Default Measurement 
    MEAS_TYPES = ['AMP','ARE','BUR','CAR','CME','CRM','DEL','DISTDU','EXTINCTDB','EXTINCTPCT','EXTINCTRATIO','EYEH','EYEWI','FALL','FREQ','HIGH','HIT','LOW','MAX','MEAN','MED','MINI','NCRO','NDU','NOV','NWI','PBAS','PCRO','PCTCRO','PDU','PEAKH','PERI','PHA','PK2P','PKPKJ','PKPN','POV','PTOP','PWI','QFAC','RIS','RMS','RMSJ','RMSN','SIGMA1','SIGMA2','SIGMA3','SIXS','SNR','STD','WAVEFORMS']
    SOURCES = ['CH1','CH2','CH3','CH4','MATH1','MATH2','MATH3','MATH4','REF1','REF2','REF3','REF4','HIS']

    encoding   = StringCommand(get_string="DAT:ENC?;", set_string="DAT:ENC {:s};",
                        allowed_values=["ASCI","RIB","RPB","FPB","SRI","SRP","SFP"])
    channel = IntCommand(get_string="DAT:SOU?;", set_string="DAT:SOU {:s};",value_map={1:"CH1",2:"CH2",3:"CH3",4:"CH4"})
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

    # Acquistion options and control
    num_averages = IntCommand(get_string="ACQUIRE:NUMAVG?;",set_string="ACQUIRE:NUMAVE {:d};")

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

    @property
    def clear(self):
        self.interface.write("CLEAR ALL;")

    @property
    def snap(self):
        """Sets the start and stop points to the the current front panel display.
        This doesn't actually seem to work, strangely."""
        self.interface.write("DAT SNAp;")

    # Acquistion options and control
    num_averages = IntCommand(get_string="ACQUIRE:NUMAVG?;",set_string="ACQUIRE:NUMAVE {:d};")

    # Turn acquisition ON/OFF
    @property 
    def run(self):
        if self.interface.query_ascii_values("ACQUIRE:STATE?",converter=u'd')[0] == 0:
            return 'OFF'
        else: 
            return 'ON'

    @run.setter
    def run(self,val='OFF'):
        if val not in ['ON','OFF']:
            raise ValueError("Run must be ON or OFF.")
        else: 
            self.interface.write("ACQUIRE:STATE {:s}".format(val))

    # Set Acquisition to single or continuous
    @property
    def acquire(self):
        return self.interface.query_ascii_values("ACQUIRE:STOPAFTER?",converter=u's')[0]

    @acquire.setter
    def acquire(self,val='SEQ'):
        if val not in ['RUNST','SEQ']:
            raise ValueError("Acquisition must be RUNST (continuous) or SEQ (single).")
        else: 
            self.interface.write("ACQUIRE:STOPAFTER {:s}".format(val))

    # Get number of aquisitions since last reset
    @property
    def acquire_num(self):
        return self.interface.query_ascii_values("ACQUIRE:NUMAC?",converter=u'd')[0]

    # Get Waveform trace
    @property
    def get_trace(self):
        self.encoding = "SRI" # Signed ints

        record_length = self.record_length
        self.data_start = 1
        self.data_stop  = record_length

        strf_from_depth = {1: 'b', 2: 'h', 4: 'l', 8: 'q'}

        vals = self.interface.query_binary_values("CURVe?;", datatype=strf_from_depth[self.byte_depth])
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

    # Select Measurement
    @property
    def measurement(self): 
        return self.MEAS

    @measurement.setter
    def measurement(self, measurement=1):
        if measurement not in range(1,9):
            raise ValueError("Measurement assignment must be 1 through 8.")
        else:
            self.MEAS = measurement

    # Measurement Type
    @property
    def measurement_type(self): 
        return self.interface.query_ascii_values("MEASUREMENT:MEAS{:d}:TYP?".format(self.MEAS),converter=u's')[0]

    @measurement_type.setter
    def measurement_type(self, mtype='AMP'):
        if mtype not in self.MEAS_TYPES:
            raise ValueError(("Measurement Type must be "+'|'.join(['{}']*len(self.MEAS_TYPES))).format(*self.MEAS_TYPES))
        else:
            self.interface.write("MEASUREMENT:MEAS{:d}:TYP {:s}".format(self.MEAS,mtype))

    # Measurement Source1
    @property
    def measurement_source1(self): 
        return self.interface.query_ascii_values("MEASUREMENT:MEAS{:d}:SOU1?".format(self.MEAS),converter=u's')[0]

    @measurement_source1.setter
    def measurement_source1(self, source):
        if source is None: 
            source='CH{:d}'.format(self.channel)
        if source not in self.SOURCES:
            raise ValueError(("Measurement Type must be "+'|'.join(['{}']*len(self.SOURCES))).format(*self.SOURCES))
        else:
            self.interface.write("MEASUREMENT:MEAS{:d}:SOU1 {:s}".format(self.MEAS,source))

    # Measurement Source2
    @property
    def measurement_source2(self): 
        return self.interface.query_ascii_values("MEASUREMENT:MEAS{:d}:SOU2?".format(self.MEAS),converter=u's')[0]

    @measurement_source2.setter
    def measurement_source2(self, source):
        if source is None: 
            source='CH{:d}'.format(self.channel)
        if source not in self.SOURCES:
            raise ValueError(("Measurement Type must be "+'|'.join(['{}']*len(self.SOURCES))).format(*self.SOURCES))
        else:
            self.interface.write("MEASUREMENT:MEAS{:d}:SOU2 {:s}".format(self.MEAS,source))

    # Measurement Mean
    @property
    def measurement_mean(self): 
        return self.interface.query_ascii_values("MEASUREMENT:MEAS{:d}:MEAN?".format(self.MEAS),converter=u'e')[0]

    # Measurement Standard Deviation
    @property
    def measurement_std(self): 
        return self.interface.query_ascii_values("MEASUREMENT:MEAS{:d}:STD?".format(self.MEAS),converter=u'e')[0]

    # Measurement Max
    @property
    def measurement_max(self): 
        return self.interface.query_ascii_values("MEASUREMENT:MEAS{:d}:MAX?".format(self.MEAS),converter=u'e')[0]

    # Measurement Min
    @property
    def measurement_min(self): 
        return self.interface.query_ascii_values("MEASUREMENT:MEAS{:d}:MIN?".format(self.MEAS),converter=u'e')[0]




class TekAWG5014(SCPIInstrument):
    """Tektronix AWG 5014"""

    CHANNEL = 1 # Default Channel 
    MARKER = 1 # Default Marker
    ONOFF_VALUES    = ['ON', 'OFF']

    runmode = StringCommand(scpi_string="AWGCONTROL:RMODE",allowed_values=['CONT','TRIG','GAT','SEQ','ENH'])

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

    # Run Selected Waveforms
    @property
    def run(self):
        self.interface.write("AWGCONTROL:RUN")

    # Stop Waveforms
    @property
    def stop(self):
        self.interface.write("AWGCONTROL:STOP")

    # Load Waveform
    def loadwaveform(self,name,points):

        if name is not None: 
            self.interface.write("WLIST:WAVEFORM:DELETE {:s}".format(name))
            self.interface.write("WLIST:WAVEFORM:NEW {:s}, {:d}, INT".format(name,len(points)))
            self.interface.write_binary_values("WLIST:WAVEFORM:DATA {:s},".format(name),points,dataype='d',is_big_endian=False)

        else: 
            raise ValueError("No Name given for Waveform.")


    # Select Channel
    @property
    def channel(self):
        return self.CHANNEL

    @channel.setter
    def channel(self, channel=1):
        if channel not in range(1,self.interface.query_ascii_values("AWGCONTROL:CONFIGURE:CNUMBER?",converter=u'd')[0]+1):
            raise ValueError("Channel must be integer between 1 and {}".format(self.interface.query_ascii_values("AWGCONTROL:CONFIGURE:CNUMBER?",converter=u'd')[0]))
        else:
            self.CHANNEL = channel

    # Select Marker
    @property
    def marker(self):
        return self.MARKER

    @marker.setter
    def marker(self, marker=1):
        if marker not in range(1,3):
            raise ValueError("Marker must be 1 or 2")
        else:
            self.MARKER = marker

    # Channel Output 
    @property
    def output(self):

        query_str = "OUTPUT{:d}:STATE?".format(self.CHANNEL)
        if self.interface.query_ascii_values(query_str, converter=u'd')[0] == 0: 
            return 'OFF'
        else: 
            return 'ON'

    @output.setter
    def output(self, val='OFF'):

        if val not in self.ONOFF_VALUES: 
            raise ValueError("Channel Output must be ON or OFF.")
        self.interface.write("OUTPUT{:d}:STATE {:s}".format(self.CHANNEL,val))


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

    # Channel Skew
    @property
    def skew(self):

        query_str = "SOURCE{:d}:SKEW?".format(self.CHANNEL)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @skew.setter
    def skew(self, val=0):
 
        if abs(val)>100e-12: 
            raise ValueError("Skew must be <= 100 ps")
        else:
            self.interface.write(("SOURCE{:d}:SKEW {:E}".format(self.CHANNEL,val)))

    # Channel Sampling Frequency
    @property
    def frequency(self):

        query_str = "SOURCE{:d}:FREQUENCY?".format(self.CHANNEL)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @frequency.setter
    def frequency(self, val=1.2e9):
 
        if (val<10e6) or (val>1.2e9): 
            raise ValueError("Sampling Frequency must be between 10 MHz and 1.2 GHz.")
        else: 
            self.interface.write(("SOURCE{:d}:FREQUENCY {:E}".format(self.CHANNEL,val)))

    # Channel Waveform
    @property
    def waveform(self):

        query_str = "SOURCE{:d}:WAVEFORM?".format(self.CHANNEL)
        return self.interface.query_ascii_values(query_str, converter=u's')[0].strip()

    @waveform.setter
    def waveform(self, val):
 
        self.interface.write(("SOURCE{:d}:WAVEFORM {:s}".format(self.CHANNEL,val)))

    # Marker Amplitude
    @property
    def marker_amplitude(self):

        query_str = "SOURCE{:d}:MARKER{:d}:VOLT:AMPLITUDE?".format(self.CHANNEL,self.MARKER)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @marker_amplitude.setter
    def marker_amplitude(self, val=2e-2):
 
        if (val>2) or (val<2e-2): 
            raise ValueError("Amplitude must be between 0.02 and 2 Volts pk-pk.")
        else:
            self.interface.write(("SOURCE{:d}:MARKER{:d}:VOLT:AMPLITUDE {:E}".format(self.CHANNEL,self.MARKER,val)))

    # Marker Offset
    @property
    def marker_offset(self):

        query_str = "SOURCE{:d}:MARKER{:d}:VOLT:OFFSET?".format(self.CHANNEL,self.MARKER)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @marker_offset.setter
    def marker_offset(self, val):

        self.interface.write(("SOURCE{:d}:MARKER{:d}:VOLT:OFFSET {:E}".format(self.CHANNEL,self.MARKER,val)))

    # Marker High Voltage
    @property
    def marker_high(self):

        query_str = "SOURCE{:d}:MARKER{:d}:VOLT:HIGH?".format(self.CHANNEL,self.MARKER)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @marker_high.setter
    def marker_high(self, val):
 
        self.interface.write(("SOURCE{:d}:MARKER{:d}:VOLT:HIGH {:E}".format(self.CHANNEL,self.MARKER,val)))

    # Marker Low Voltage
    @property
    def marker_low(self):

        query_str = "SOURCE{:d}:MARKER{:d}:VOLT:LOW?".format(self.CHANNEL,self.MARKER)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @marker_low.setter
    def marker_low(self, val):
 
        self.interface.write(("SOURCE{:d}:MARKER{:d}:VOLT:LOW {:E}".format(self.CHANNEL,self.MARKER,val)))

    # Marker Delay
    @property
    def marker_delay(self):

        query_str = "SOURCE{:d}:MARKER{:d}:DELAY?".format(self.CHANNEL,self.MARKER)
        return self.interface.query_ascii_values(query_str, converter=u'e')[0]

    @marker_delay.setter
    def marker_delay(self, val):
 
        self.interface.write(("SOURCE{:d}:MARKER{:d}:DELAY {:E}".format(self.CHANNEL,self.MARKER,val)))