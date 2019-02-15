# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Agilent33220A', 'Agilent33500B', 'Agilent34970A', 'AgilentE8363C', 'AgilentN5183A', 'AgilentE9010A']

import socket
import time
import copy
import re
import numpy as np
from .instrument import SCPIInstrument, Command, StringCommand, FloatCommand, IntCommand, is_valid_ipv4
from auspex.log import logger

class Agilent33220A(SCPIInstrument):
    """Agilent 33220A Function Generator"""

    def __init__(self, resource_name=None, *args, **kwargs):
        super(Agilent33220A, self).__init__(resource_name, *args, **kwargs)
        self.name = "Agilent 33220A AWG"

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(self.resource_name):
            self.resource_name += "::5025::SOCKET"
        super(Agilent33220A, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.timeout = 3000 #seem to have trouble timing out on first query sometimes

    FUNCTION_MAP = {"Sine": "SIN",
                    "Square": "SQU",
                    "Ramp": "RAMP",
                    "Pulse": "PULS",
                    "Noise": "NOIS",
                    "DC": "DC",
                    "User": "USER"}

    frequency = FloatCommand(scpi_string="FREQ")
    function = StringCommand(get_string="FUNCtion?", set_string="FUNCTION {:s}",
                                value_map = FUNCTION_MAP)

    #Voltage
    dc_offset = FloatCommand(scpi_string="VOLT:OFFSET")
    output = Command(get_string="OUTP?", set_string="OUTP {:s}", value_map = {True: "1", False: "0"})
    polarity = Command(get_string="OUTP:POL?", set_string="OUTP:POL {:s}", value_map = {1 : "NORM", -1 : "INV"})
    auto_range = Command(scpi_string="VOLTage:RANGe:AUTO", value_map={True: "1", False: "0"})
    load_resistance = FloatCommand(scpi_string="OUTPut:LOAD")
    amplitude = FloatCommand(scpi_string="VOLT")
    low_voltage = FloatCommand(scpi_string="VOLTage:LOW")
    high_voltage = FloatCommand(scpi_string="VOLTage:HIGH")
    output_units = Command(get_string="VOLTage:UNIT?", set_string="VOLTage:UNIT {:s}",
                            value_map={"Vpp" : "VPP", "Vrms" : "VRMS", "dBm" : "DBM"})

    #Trigger, Burst, etc...
    output_sync = Command(get_string="OUTPut:SYNC?", set_string="OUTPut:SYNC {:s}",
                                value_map = {True: "OFF", False: "ON"})
    burst_state = Command(get_string="BURSt:STATE?", set_string="BURSt:STATE {:s}",
                                value_map = {True: "1", False: "0"})
    burst_cycles = FloatCommand(scpi_string="BURSt:NCYCles")
    burst_mode = Command(get_string="BURSt:MODE?", set_string="BURSt:MODE {:s}",
                                value_map = {"Triggered": "TRIG", "Gated": "GAT"})
    trigger_source = Command(get_string="TRIGger:SOURce?", set_string="TRIGger:SOURce {:s}",
                        value_map = {"Internal": "IMM", "External": "EXT", "Bus": "BUS"})
    trigger_slope = Command(get_string="TRIGger:SLOPe?", set_string="TRIGger:SLOPe {:s}",
                                value_map = {"Positive": "POS", "Negative": "NEG"})

    # Pulse characteristics
    pulse_width = FloatCommand(scpi_string="FUNCtion:PULSe:WIDTh")
    pulse_period = FloatCommand(scpi_string="PULSe:PERiod")
    pulse_edge = FloatCommand(scpi_string="FUNCtion:PULSe:TRANsition")
    pulse_dcyc = IntCommand(scpi_string="FUNCtion:PULSe:DCYCle")

    ramp_symmetry = FloatCommand(scpi_string="FUNCtion:RAMP:SYMMetry")

    def trigger(self):
        self.interface.write("*TRG")


class Agilent33500B(SCPIInstrument):
    """ Agilent/Keysight 33500 series 2-channel Arbitrary Waveform Generator

    Replacement model for 33220 series with some changes and additional sequencing functionality
    """
    def __init__(self, resource_name=None, *args, **kwargs):
        super(Agilent33500B, self).__init__(resource_name, *args, **kwargs)
        self.name = "Agilent 33500B AWG"

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(self.resource_name):
            self.resource_name += "::5025::SOCKET"
        super(Agilent33500B, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.timeout = 5000 #seem to have trouble timing out on first query sometimes

    FUNCTION_MAP = {"Sine": "SIN",
                    "Square": "SQU",
                    "Triangle": "TRI",
                    "Ramp": "RAMP",
                    "Pulse": "PULS",
                    "PRBS": "PRBS",
                    "Noise": "NOIS",
                    "DC": "DC",
                    "User": "ARB"}
    frequency = FloatCommand(scpi_string="SOURce{channel:d}:FREQuency",additional_args=['channel'])
    # FUNCtion subsystem
    function = StringCommand(scpi_string="SOURce{channel:d}:FUNCtion",
                                value_map = FUNCTION_MAP,additional_args=['channel'])
    ramp_symmetry = FloatCommand(scpi_string="FUNCtion:RAMP:SYMMetry")
    # When function is Pulse:
    pulse_width = FloatCommand(scpi_string="SOURce{channel:d}:FUNCtion:PULSe:WIDTh",
                                additional_args=['channel'])
    pulse_period = FloatCommand(scpi_string="SOURce{channel:d}:FUNCtion:PULSe:PERiod",
                                additional_args=['channel'])
    pulse_edge = FloatCommand(scpi_string="SOURce{channel:d}:FUNCtion:PULSe:TRANsition",
                                additional_args=['channel'])
    pulse_dcyc = IntCommand(scpi_string="SOURce{channel:d}:FUNCtion:PULSe:DCYCle",
                                additional_args=['channel'])
    # When function is ARBitrary:
    arb_waveform = StringCommand(scpi_string="SOURce{channel:d}:FUNCtion:ARBitrary",
                                additional_args=['channel'])
    arb_advance = StringCommand(scpi_string="SOURce{channel:d}:FUNCtion:ARBitrary:ADVance",
                                allowed_values=["Trigger","Srate"],
                                additional_args=['channel'],
                                doc="Advance mode to the next point: 'Trigger' or 'Srate' (Sample Rate)")
    arb_frequency = FloatCommand(scpi_string="SOURce{channel:d}:FUNCtion:ARBitrary:FREQuency",
                                additional_args=['channel'])
    arb_amplitude = FloatCommand(scpi_string="SOURce{channel:d}:FUNCtion:ARBitrary:PTPeak",
                                additional_args=['channel'])
    arb_sample = FloatCommand(scpi_string="SOURce{channel:d}:FUNCtion:ARBitrary:SRATe",
                                additional_args=['channel'],doc="Sample Rate")
    # VOLTage subsystem
    amplitude = FloatCommand(scpi_string="SOURce{channel:d}:VOLT",
                            additional_args=['channel'])
    dc_offset = FloatCommand(scpi_string="SOURce{channel:d}:VOLT:OFFSET",
                            additional_args=['channel'])
    auto_range = Command(scpi_string="SOURce{channel:d}:VOLTage:RANGe:AUTO",
                            value_map={True: "1", False: "0"},
                            additional_args=['channel'])
    low_voltage = FloatCommand(scpi_string="SOURce{channel:d}:VOLTage:LOW",
                            additional_args=['channel'])
    high_voltage = FloatCommand(scpi_string="SOURce{channel:d}:VOLTage:HIGH",
                            additional_args=['channel'])
    output_units = Command(scpi_string="SOURce{channel:d}:VOLTage:UNIT",
                            value_map={"Vpp" : "VPP", "Vrms" : "VRMS", "dBm" : "DBM"},
                            additional_args=['channel'])
    # Output subsystem
    output = Command(scpi_string="OUTP{channel:d}", value_map={True: "1", False: "0"},
                            additional_args=['channel'])
    load = FloatCommand(scpi_string="OUTP{channel:d}:LOAD", value_range=[1,1e4],
                            additional_args=['channel'],
                            doc="Expected load resistance, 1-10k")
    output_gated = Command(scpi_string="OUTP{channel:d}:MODE", value_map={True:"GATed", False:"NORMal"},
                            additional_args=['channel'])
    polarity = Command(scpi_string="OUTP{channel:d}:POL",
                            value_map = {1 : "NORM", -1 : "INV"},
                            additional_args=['channel'])
    output_sync = Command(scpi_string="OUTP{channel:d}:SYNC",
                            value_map = {True: "1", False: "0"},
                            additional_args=['channel'])
    sync_mode = StringCommand(scpi_string="OUTP{channel:d}:SYNC:MODE",
                            allowed_values = ["Normal","Carrier","Marker"],
                            additional_args=['channel'])
    sync_polarity = Command(scpi_string="OUTP{channel:d}:SYNC:POL",
                            value_map = {1 : "NORM", -1 : "INV"},
                            additional_args=['channel'])
    sync_source = Command(scpi_string="OUTP:SYNC:SOURce",
                            value_map = {1 : "CH1", 2 : "CH2"})
    output_trigger = Command(scpi_string="OUTP:TRIGger", value_map={True:'1', False:'0'})
    output_trigger_source = Command(scpi_string="OUTP:TRIGger:SOURce",
                        value_map = {1: "CH1", 2: "CH2"})
    output_trigger_slope = StringCommand(scpi_string="OUTP:TRIGger:SLOPe",
                                value_map = {"Positive": "POS", "Negative": "NEG"})
    #Trigger, Burst, etc...
    burst_state = Command(scpi_string="SOURce{channel:d}:BURSt:STATE",
                                value_map = {True: '1', False: '0'},
                                additional_args=['channel'])
    burst_cycles = FloatCommand(scpi_string="SOURce{channel:d}:BURSt:NCYCles",
                            additional_args=['channel'])
    burst_mode = StringCommand(scpi_string="SOURce{channel:d}:BURSt:MODE",
                                value_map = {"Triggered": "TRIG", "Gated": "GAT"},
                            additional_args=['channel'])
    trigger_source = StringCommand(scpi_string="TRIGger:SOURce",
                        value_map = {"Internal": "IMM", "External": "EXT", "Bus": "BUS"})
    trigger_slope = StringCommand(scpi_string="TRIGger:SLOPe",
                                value_map = {"Positive": "POS", "Negative": "NEG"})
    # Data subsystem
    sequence = StringCommand(scpi_string="SOURce{channel:d}:DATA:SEQuence",
                            additional_args=['channel'])

    def set_infinite_load(self, channel=1):
        self.interface.write("OUTP{channel:d}:LOAD INF".format(channel=channel))

    def clear_waveform(self,channel=1):
        """ Clear all waveforms loaded in the memory """
        logger.debug("Clear all waveforms loaded in the memory of %s" %self.name)
        self.interface.write("SOURce%d:DATA:VOLatile:CLEar" %channel)

    def upload_waveform(self,data,channel=1,name="mywaveform",dac=True):
        """ Load string-converted data into a waveform memory

        dac: True if values are converted to integer already
        """
        # Check number of data points
        N = len(data)
        if N<8 or N>65536:
            log.error("Data has invalid length = %d. Must be between 8 and 65536. Cannot upload waveform." %N)
            return False
        # Check length of waveform length, must be <=12
        if len(name)>12:
            logger.warning("Waveform length is larger than the limit of 12. Will be clipped off: %s --> %s" \
                            %(name,name[:12]))
        name = name[:12] # Arb waveform name at most 12 characters
        if dac: # Data points are integer
            dac_str = ":DAC"
            # Values must be within -32767 and 32767
            if abs(np.max(data))>32767:
                logger.warning("Some points out of range [-32767,32767] will be clipped off.")
            data = [int(max(min(datum,32767),-32767)) for datum in data]
        else: # Data points are float
            dac_str = ""
            # Values must be within -1 and 1
            if abs(np.max(data))>1:
                logger.warning("Some points out of range [-1,1] will be clipped off.")
            data = [max(min(datum,1),-1) for datum in data]
        # convert array into string
        data_str = ','.join([str(item) for item in data])
        logger.debug("Upload waveform %s to instrument %s, channel %d: %s" %(name,self.name,channel,data))
        self.interface.write("SOURce%s:DATA:ARBitrary1%s %s,%s" %(channel,dac_str,name,data_str))
        # Check if successfully uploaded or not
        data_pts = int(self.interface.query("SOURce%s:DATA:ATTR:POIN? %s" %(channel,name)))
        if data_pts==N:
            logger.debug("Successfully uploaded waveform %s to instrument %s, channel %d" %(name,self.name,channel))
            return True
        else:
            logger.error("Failed uploading waveform %s to instrument %s, channel %d" %(name,self.name,channel))
            return False

    def upload_waveform_binary(self,data,channel=1,name="mywaveform",dac=True):
        """ NOT YET WORKING - DO NOT USE
        Load binary data into a waveform memory

        dac: True if values are converted to integer already
        """
        logger.warning("Binary upload is under development. May not work as intended. Please consider using ASCII upload: upload_waveform()")
        N = len(data)
        if N<8 or N>16e6:
            log.error("Data has invalid length = %d. Must be between 8 and 16M. Cannot upload waveform." %N)
            return False
        # Check length of waveform length, must be <=12
        if len(name)>12:
            logger.warning("Waveform length is larger than the limit of 12. Will be clipped off: %s --> %s" \
                            %(name,name[:12]))
        name = name[:12] # Arb waveform name at most 12 characters
        # We don't support float values, so must convert to integer
        if not dac:
            logger.warning("We current do not support uploading float values. Waveform values will be converted to integer.")
            # Convert to integer option (dac=True)
            data = [datum*32767 for datum in data]
        # Values must be within -32767 and 32767
        if abs(np.max(data))>32767:
            logger.warning("Some points out of range [-32767,32767] will be clipped off.")
        data = [int(max(min(datum,32767),-32767)) for datum in data]
        wf = []
        N = N*2 # 2 bytes for each point
        n = int(np.log10(N))+1 # Number of digits of N
        # Split 2-byte integer into 2 separate bytes
        for datum in data:
            if datum>0:
                wf.append(datum >> 8)
                wf.append(datum % 1<<8)
            else:
                datum = -datum
                wf.append(-(datum >> 8))
                wf.append(-(datum % 1<<8))
        wf = np.array(wf,dtype='int8') # Force datatype to 8-bit integer
        logger.debug("Upload waveform %s to instrument %s, channel %d: %s" %(name,self.name,channel,wf))
        self.interface.write_binary_values("SOURce%s:DATA:ARBitrary1:DAC %s,#%d%d" %(channel,name,n,N),
                                            wf, datatype='b', is_big_endian=False)
        # Check if successfully uploaded or not
        data_pts = int(self.interface.query("SOURce%s:DATA:ATTR:POIN? %s" %(channel,name)))
        if data_pts==len(data):
            logger.debug("Successfully uploaded waveform %s to instrument %s, channel %d" %(name,self.name,channel))
            return True
        else:
            logger.error("Failed uploading waveform %s to instrument %s, channel %d" %(name,self.name,channel))
            return False

    def upload_sequence(self,sequence,channel=1,binary=False):
        """ Upload a sequence to the instrument """
        # Upload each segment
        if binary:
            for seg in sequence.segments:
                self.upload_waveform_binary(seg.data,channel=channel,name=seg.name,dac=seg.dac)
        else:
            for seg in sequence.segments:
                self.upload_waveform(seg.data,channel=channel,name=seg.name,dac=seg.dac)
        # Now upload the sequence
        descriptor = sequence.get_descriptor()
        logger.debug("Upload sequence %s to %s: %s" %(sequence.name,self.name,descriptor))
        self.set_sequence(descriptor,channel=channel)

    def arb_sync(self):
        """ Restart the sequences and synchronize them """
        self.interface.write("FUNCtion:ARBitrary:SYNChronize")

    def trigger(self,channel=1):
        self.interface.write("TRIGger%d" %channel)

    def abort(self):
        self.interface.write("ABORt")

    # Subclasses of Agilent33500B
    class Segment(object):
        def __init__(self,name,data=[],dac=True,control="once",repeat=0,mkr_mode="maintain",mkr_pts=4):
            """ Information of a segment/waveform

            dac: True if data is converted to integer already
            control: once - play once
                    onceWaitTrig - play once then wait for trigger
                    repeat - repeat a number of times (repeat count)
                    repeatInf - repeat forever until stopped
                    repeatTilTrig - repeat until triggered then advance
            repeat: number of repeats if control == repeat
            mkr_mode: marker mode: maintain-maintaincurrentmarkerstateatstartofsegment
                                lowAtStart-forcemarkerlowatstartofsegment
                                highAtStart-forcemarkerhighatstartofsegment
                                highAtStartGoLow-forcemarkerhighatstartofsegmentandthenlowatmarkerposition
            mkr_pts: marker points
            """
            self.name = name
            self.data = data
            self.dac = dac
            self.control = control
            self.repeat = repeat
            self.mkr_mode = mkr_mode
            self.mkr_pts = mkr_pts

        def self_check(self):
            N = len(self.data)
            if N<8:
                logger.error("Waveform %s must have at least 8 points" %self.name)
                return False
            else:
                if self.mkr_pts < 4:
                    self.mkr_pts = 4
                    logger.warning("Marker points of waveform %s is less than 4. Set to 4." %self.name)
                if self.mkr_pts > N-3:
                    self.mkr_pts = N-3
                    logger.warning("Marker points of waveform %s is longer than lenght-3 = %d. Set to lenght-3." %(self.name,N-3))
                return True

        def update(self,**kwargs):
            for k,v in kwargs.items():
                if k in ["name","data","control","repeat","mkr_mode","mkr_pts"]:
                    logger.debug("Set %s.%s = %s" %(self.name,k,v))
                    setattr(self,k,v)
                else:
                    logger.warning("Key %s is not valid for Segment %s. Ignore." %(k,self.name))
            return self.self_check()

    class Sequence(object):
        def __init__(self,name):
            self.name = name
            self.segments = []

        def add_segment(self,segment,**kwargs):
            """ Create a copy of the segment, update its values, then add to the sequence.
            The copy and update are to allow reuse of the same segment with different configurations.
            For safety, avoid reuse, but add different segment objects to the sequence.
            """
            seg = copy.copy(segment)
            if seg.update(**kwargs):
                logger.debug("Add segment %s into sequence %s" %(seg.name,self.name))
                self.segments.append(seg)

        def get_descriptor(self):
            """ Return block descriptor to upload to the instrument """
            descriptor = '"%s"' %self.name
            for seg in self.segments:
                descriptor += ',"%s",%d,%s,%s,%d' %(seg.name,seg.repeat,seg.control,seg.mkr_mode,seg.mkr_pts)
            N = len(descriptor)
            n = int(np.log10(N))+1
            descriptor = "#%d%d%s" %(n,N,descriptor)
            logger.debug("Block descriptor for sequence %s: %s" %(self.name,descriptor))
            return descriptor


class Agilent34970A(SCPIInstrument):
    """Agilent 34970A MUX"""

# Array of Channels to configure
    CONFIG_LIST     = []

# Allowed value arrays

    RES_VALUES      = ['AUTO',1E2, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8]
    PLC_VALUES      = [0.02, 0.2, 1, 10, 20, 100, 200]
    ONOFF_VALUES    = ['ON', 'OFF']
    TRIGSOUR_VALUES = ['BUS','IMM','EXT','TIM']
    ADVSOUR_VALUES  = ['EXT','BUS','IMM']

# Commands needed to configure MUX for measurement with an external instrument

    dmm            = StringCommand(scpi_string="INST:DMM",value_map={'ON': '1', 'OFF': '0'})
    trigger_source = StringCommand(scpi_string="TRIG:SOUR",allowed_values=TRIGSOUR_VALUES)
    advance_source = StringCommand(scpi_string="ROUT:CHAN:ADV:SOUR",allowed_values=ADVSOUR_VALUES)

# Generic init and connect methods

    def __init__(self, resource_name=None, *args, **kwargs):
        super(Agilent34970A, self).__init__(resource_name, *args, **kwargs)
        self.name = "Agilent 34970A MUX"

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        super(Agilent34970A, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"

#Channel to String helper function converts int array to channel list string

    def ch_to_str(self, ch_list):
        return ("(@"+','.join(['{:d}']*len(ch_list))+")").format(*ch_list)

#Helper function to sort channels by resistance measurement type

    def r_lists(self):
        fres_list, res_list = [], []
        res_map = self.resistance_wire

        for ch in self.CONFIG_LIST:
            if res_map[ch] =='"FRES"':
                fres_list.append(ch)
            if res_map[ch] =='"RES"':
                res_list.append(ch)

        return fres_list, res_list

#Setup Scan List

    @property
    def scanlist(self):
        slist = re.findall('\d{3}(?=[,)])',self.interface.query("ROUT:SCAN?"))
        return [int(i) for i in slist]

    @scanlist.setter
    def scanlist(self, ch_list):
        self.interface.write("ROUT:SCAN "+self.ch_to_str(ch_list))

#Setup Config List

    @property
    def configlist(self):
        return self.CONFIG_LIST

    @configlist.setter
    def configlist(self, ch_list):
        self.CONFIG_LIST = ch_list

#Start Scan
    def scan(self):
        self.interface.write("INIT")

#Read Values
    def read(self):
        if self.dmm=="ON":
            return self.interface.query_ascii_values("FETC?", converter=u'e')
        else:
            raise Exception("Cannot issue command when DMM is disabled. Enable DMM")

# Commands that configure resistance measurement type, 2 or 4 wire

    @property
    def resistance_wire(self):
        if self.dmm=="ON":
            query_str = "SENS:FUNC? "+self.ch_to_str(self.CONFIG_LIST)
            output = self.interface.query_ascii_values(query_str, converter=u's')
        else:
            query_str = "ROUT:CHAN:FWIR? "+self.ch_to_str(self.CONFIG_LIST)
            output = self.interface.query_ascii_values(query_str, converter=u'd')
        return {ch: val for ch, val in zip(self.CONFIG_LIST, output)}

    @resistance_wire.setter
    def resistance_wire(self, fw = 2):
        if self.dmm=="ON":
            fw_char = "F" if fw == 4 else ""
            self.interface.write(("CONF:{}RES "+self.ch_to_str(self.CONFIG_LIST)).format(fw_char))
        else:
            fw_char = "ON," if fw == 4 else "OFF,"
            self.interface.write(("ROUT:CHAN:FWIR {}"+self.ch_to_str(self.CONFIG_LIST)).format(fw_char))

# Commands that configure resistance measurements with internal DMM

    @property
    def resistance_range(self):
        if self.dmm=="OFF":
            raise Exception("Cannot issue command when DMM is disabled. Enable DMM")
        else:
            fres_list, res_list = self.r_lists()
            output = {}
            if len(fres_list)>0:
                query_str = ("SENS:FRES:RANG? "+self.ch_to_str(fres_list))
                fres_rng  = self.interface.query_ascii_values(query_str, converter=u'e')
                output.update({ch: val for ch, val in zip(fres_list, fres_rng)})
            if len(res_list)>0:
                query_str = ("SENS:RES:RANG? "+self.ch_to_str(res_list))
                res_rng   = self.interface.query_ascii_values(query_str, converter=u'e')
                output.update({ch: val for ch, val in zip(res_list, res_rng)})
            return output

    @resistance_range.setter
    def resistance_range(self, val="AUTO"):
        if val not in self.RES_VALUES:
            raise ValueError(("Resistance range must be "+'|'.join(['{}']*len(self.RES_VALUES))+" Ohms").format(*self.RES_VALUES))
        if self.dmm=="OFF":
            raise Exception("Cannot issue command when DMM is disabled. Enable DMM")
        else:
            fres_list, res_list = self.r_lists()
            if len(fres_list)>0:
                if val=="AUTO":
                    self.interface.write("SENS:FRES:RANG:AUTO ON,"+self.ch_to_str(fres_list))
                else:
                    self.interface.write("SENS:FRES:RANG:AUTO OFF,"+self.ch_to_str(fres_list))
                    self.interface.write(("SENS:FRES:RANG {:E},"+self.ch_to_str(fres_list)).format(val))
            if len(res_list)>0:
                if val=="AUTO":
                    self.interface.write("SENS:RES:RANG:AUTO ON,"+self.ch_to_str(res_list))
                else:
                    self.interface.write("SENS:RES:RANG:AUTO OFF,"+self.ch_to_str(res_list))
                    self.interface.write(("SENS:RES:RANG {:E},"+self.ch_to_str(res_list)).format(val))

    @property
    def resistance_resolution(self):
        if self.dmm=="OFF":
            raise Exception("Cannot issue command when DMM is disabled. Enable DMM")
        else:
            fres_list, res_list = self.r_lists()
            output = {}
            if len(fres_list)>0:
                query_str = ("SENS:FRES:NPLC? "+self.ch_to_str(fres_list))
                fres_resl = self.interface.query_ascii_values(query_str, converter=u'e')
                output.update({ch: val for ch, val in zip(fres_list, fres_resl)})
            if len(res_list)>0:
                query_str = ("SENS:RES:NPLC? "+self.ch_to_str(res_list))
                res_resl  = self.interface.query_ascii_values(query_str, converter=u'e')
                output.update({ch: val for ch, val in zip(res_list, res_resl)})
            return output

    @resistance_resolution.setter
    def resistance_resolution(self, val=1):
        if val not in self.PLC_VALUES:
            raise ValueError(("PLC integration times must be "+'|'.join(['{:E}']*len(self.PLC_VALUES))+" cycles").format(*self.PLC_VALUES))
        if self.dmm=="OFF":
            raise Exception("Cannot issue command when DMM is disabled. Enable DMM")
        else:
            fres_list, res_list = self.r_lists()
            if len(fres_list)>0:
                self.interface.write(("SENS:FRES:NPLC {:E},"+self.ch_to_str(fres_list)).format(val))
            if len(res_list)>0:
                self.interface.write(("SENS:RES:NPLC {:E},"+self.ch_to_str(res_list)).format(val))

    @property
    def resistance_zcomp(self):
        if self.dmm=="OFF":
            raise Exception("Cannot issue command when DMM is disabled. Enable DMM")
        else:
            fres_list, res_list = self.r_lists()
            output = {}
            if len(fres_list)>0:
                query_str = ("SENS:FRES:OCOM? "+self.ch_to_str(fres_list))
                fres_zcom = self.interface.query_ascii_values(query_str, converter=u'd')
                output.update({ch: val for ch, val in zip(fres_list, fres_zcom)})
            if len(res_list)>0:
                query_str = ("SENS:RES:OCOM? "+self.ch_to_str(res_list))
                res_zcom  = self.interface.query_ascii_values(query_str, converter=u'd')
                output.update({ch: val for ch, val in zip(res_list, res_zcom)})
            return output

    @resistance_zcomp.setter
    def resistance_zcomp(self, val="OFF"):
        if val not in self.ONOFF_VALUES:
            raise ValueError("Zero compensation must be ON or OFF. Only valid for resistance range less than 100 kOhm")
        if self.dmm=="OFF":
            raise Exception("Cannot issue command when DMM is disabled. Enable DMM")
        else:
            fres_list, res_list = self.r_lists()
            if len(fres_list)>0:
                self.interface.write(("SENS:FRES:OCOM {:s},"+self.ch_to_str(fres_list)).format(val))
            if len(res_list)>0:
                self.interface.write(("SENS:RES:OCOM {:s},"+self.ch_to_str(res_list)).format(val))

class AgilentN5183A(SCPIInstrument):
    """AgilentN5183A microwave source"""
    instrument_type = "Microwave Source"

    frequency = FloatCommand(scpi_string=":freq")
    power     = FloatCommand(scpi_string=":power")
    phase     = FloatCommand(scpi_string=":phase")

    alc       = StringCommand(scpi_string=":power:alc", value_map={True: '1', False: '0'})
    mod       = StringCommand(scpi_string=":output:mod", value_map={True: '1', False: '0'})

    output    = StringCommand(scpi_string=":output", value_map={True: '1', False: '0'})

    def __init__(self, resource_name=None, *args, **kwargs):
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        super(AgilentN5183A, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type="VISA"):
        if resource_name is not None:
            self.resource_name = resource_name
        if is_valid_ipv4(self.resource_name):
            if "::5025::SOCKET" not in self.resource_name:
                self.resource_name += "::5025::SOCKET"
        print(self.resource_name)
        super(AgilentN5183A, self).connect(resource_name=resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.timeout = 3000 #seem to have trouble timing out on first query sometimes

    def set_all(self, settings):
        super(AgilentN5183A, self).set_all(settings)

class AgilentE8363C(SCPIInstrument):
    """Agilent E8363C VNA"""
    instrument_type = "Vector Network Analyzer"

    AVERAGE_TIMEOUT = 12. * 60. * 60. * 1000. #milliseconds

    power              = FloatCommand(scpi_string=":SOURce:POWer:LEVel:IMMediate:AMPLitude", value_range=(-27, 20))
    frequency_center   = FloatCommand(scpi_string=":SENSe:FREQuency:CENTer")
    frequency_span     = FloatCommand(scpi_string=":SENSe:FREQuency:SPAN")
    frequency_start    = FloatCommand(scpi_string=":SENSe:FREQuency:STARt")
    frequency_stop     = FloatCommand(scpi_string=":SENSe:FREQuency:STOP")
    sweep_num_points   = IntCommand(scpi_string=":SENSe:SWEep:POINts")
    averaging_factor   = IntCommand(scpi_string=":SENSe1:AVERage:COUNt")
    averaging_enable   = StringCommand(get_string=":SENSe1:AVERage:STATe?", set_string=":SENSe1:AVERage:STATe {:s}", value_map={False:"0", True:"1"})
    averaging_complete = StringCommand(get_string=":STATus:OPERation:AVERaging1:CONDition?", value_map={False:"+0", True:"+2"})
    if_bandwidth       = FloatCommand(scpi_string=":SENSe1:BANDwidth")
    sweep_time         = FloatCommand(get_string=":SENSe:SWEep:TIME?")

    def __init__(self, resource_name=None, *args, **kwargs):
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        super(AgilentE8363C, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type="VISA"):
        if resource_name is not None:
            self.resource_name = resource_name
        if is_valid_ipv4(self.resource_name):
            self.resource_name += "::hpib7,16::INSTR"
        else:
            logger.error("The resource name for the Agilent E8363C: {} is " +
                "not a valid IPv4 address.".format(self.resource_name))
        super(AgilentE8363C, self).connect(resource_name=None,
            interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"

    def averaging_restart(self):
        """ Restart trace averaging """
        self.interface.write(":SENSe1:AVERage:CLEar")

    def reaverage(self):
        """ Restart averaging and block until complete """
        if self.averaging_enable:
            self.averaging_restart()
            #trigger after the requested number of points has been averaged
            self.interface.write("SENSe1:SWEep:GROups:COUNt %d"%self.averaging_factor)
            self.interface.write("ABORT;SENSe1:SWEep:MODE GRO")
        else:
            #restart current sweep and send a trigger
            self.interface.write("ABORT;SENS:SWE:MODE SING")
        #wait for the measurement to finish, with a temporary long timeout
        tmout = self.interface._resource.timeout
        self.interface._resource.timeout = self.AVERAGE_TIMEOUT
        self.interface.WAI()
        while not self.averaging_complete:
            time.sleep(0.1) #TODO: Asynchronous check of SRQ
        self.interface._resource.timeout = tmout

    def get_trace(self, measurement=None):
        """ Return a tupple of the trace frequencies and corrected complex points """
        #If the measurement is not passed in just take the first one
        if measurement is None:
            traces = self.interface.query(":CALCulate:PARameter:CATalog?")
            #traces come e.g. as  u'"CH1_S11_1,S11,CH1_S21_2,S21"'
            #so split on comma and avoid first quote
            measurement = traces.split(",")[0][1:]
        #Select the measurment
        self.interface.write(":CALCulate:PARameter:SELect '{}'".format(measurement))
        #Take the data as interleaved complex values
        interleaved_vals = self.interface.values(":CALCulate:DATA? SDATA")
        vals = interleaved_vals[::2] + 1j*interleaved_vals[1::2]
        #Get the associated frequencies
        freqs = np.linspace(self.frequency_start, self.frequency_stop, self.sweep_num_points)
        return (freqs, vals)

class AgilentE9010A(SCPIInstrument):
    """Agilent E9010A SA"""
    instrument_type = "Spectrum Analyzer"

    frequency_center = FloatCommand(scpi_string=":FREQuency:CENTer")
    frequency_span   = FloatCommand(scpi_string=":FREQuency:SPAN")
    frequency_start  = FloatCommand(scpi_string=":FREQuency:STARt")
    frequency_stop   = FloatCommand(scpi_string=":FREQuency:STOP")

    num_sweep_points = FloatCommand(scpi_string=":SWEep:POINTs")
    resolution_bandwidth = FloatCommand(scpi_string=":BANDwidth")
    sweep_time = FloatCommand(scpi_string=":SWEep:TIME")

    mode = StringCommand(scpi_string=":INSTrument", allowed_values=["SA", "BASIC", "PULSE", "PNOISE"])

    # phase noise application commands
    pn_offset_start = FloatCommand(scpi_string=":LPLot:FREQuency:OFFSet:STARt")
    pn_offset_stop  = FloatCommand(scpi_string=":LPLot:FREQuency:OFFSet:STOP")
    pn_carrier_freq = FloatCommand(scpi_string=":FREQuency:CARRier")

    def __init__(self, resource_name=None, *args, **kwargs):
        super(AgilentE9010A, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(resource_name):
            resource_name += "::5025::SOCKET"
        super(AgilentE9010A, self).connect(resource_name=resource_name, interface_type=interface_type)
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

    @property
    def power(self):
        return self.interface.query_ascii_values("MEAS:CHP:CHP?",converter='e')[0]
