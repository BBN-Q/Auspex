# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, is_valid_ipv4
from auspex.log import logger
import socket
import time
import numpy as np

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

    # This seems to return incorrect numbers for large sweeps?
    num_sweep_points = FloatCommand(scpi_string="OBW:SWE:POIN")

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
