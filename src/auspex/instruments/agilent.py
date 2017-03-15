# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import socket
import time
import re
import numpy as np
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, is_valid_ipv4
from auspex.log import logger

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

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        if is_valid_ipv4(self.resource_name):
            self.resource_name += "::5025::SOCKET"
        print(self.resource_name)
        super(AgilentN5183A, self).connect(resource_name=resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.timeout = 3000 #seem to have trouble timing out on first query sometimes

    def set_all(self, settings):
        settings['frequency'] = settings['frequency']*1e9
        super(AgilentN5183A, self).set_all(settings)

class AgilentE8363C(SCPIInstrument):
    """Agilent E8363C VNA"""
    instrument_type = "Vector Network Analyzer"

    power              = FloatCommand(scpi_string=":SOURce:POWer:LEVel:IMMediate:AMPLitude", value_range=(-27, 20))
    frequency_center   = FloatCommand(scpi_string=":SENSe:FREQuency:CENTer")
    frequency_span     = FloatCommand(scpi_string=":SENSe:FREQuency:SPAN")
    frequency_start    = FloatCommand(scpi_string=":SENSe:FREQuency:STARt")
    frequency_stop     = FloatCommand(scpi_string=":SENSe:FREQuency:STOP")
    sweep_num_points   = IntCommand(scpi_string=":SENSe:SWEep:POINts")
    averaging_factor   = IntCommand(scpi_string=":SENSe1:AVERage:COUNt")
    averaging_enable   = StringCommand(get_string=":SENSe1:AVERage:STATe?", set_string=":SENSe1:AVERage:STATe {:c}", value_map={False:"0", True:"1"})
    averaging_complete = StringCommand(get_string=":STATus:OPERation:AVERaging1:CONDition?", value_map={False:"+0", True:"+2"})

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
        self.averaging_restart()
        while not self.averaging_complete:
            #TODO with Python 3.5 turn into coroutine and use await asyncio.sleep()
            time.sleep(0.1)

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
