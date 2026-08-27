
__all__ = ['RSA3045N']

import socket
import time
import copy
import re
import numpy as np
from itertools import product
from .instrument import SCPIInstrument, Command, StringCommand, BoolCommand, FloatCommand, IntCommand, is_valid_ipv4
from auspex import log
from auspex.log import logger
import pyvisa.util as util


class RSA3045N(SCPIInstrument):
    """Rigol RSA3045N Spectrum Analyzer"""
    instrument_type = "Spectrum Analyzer"

    frequency_center = FloatCommand(scpi_string=":FREQuency:CENTer")
    frequency_span   = FloatCommand(scpi_string=":FREQuency:SPAN")
    frequency_start  = FloatCommand(scpi_string=":FREQuency:STARt")
    frequency_stop   = FloatCommand(scpi_string=":FREQuency:STOP")

    num_sweep_points        = FloatCommand(scpi_string=":SWEep:POINTs")
    resolution_bandwidth    = FloatCommand(scpi_string=":BANDwidth")
    video_bandwidth         = FloatCommand(scpi_string=":BANDwidth:VIDeo")
    video_auto              = BoolCommand(get_string=":BANDwidth:VIDEO:AUTO?",
                                set_string=":BANDwidth:VIDEO:AUTO {:s}",
                                value_map={False: "0", True: "1"})
    sweep_time              = FloatCommand(scpi_string=":SWEep:TIME")
    averaging_count         = IntCommand(scpi_string=':AVER:COUN')

    marker1_amplitude = FloatCommand(scpi_string=':CALC:MARK1:Y')
    marker1_position = FloatCommand(scpi_string=':CALC:MARK1:X')

    mode = StringCommand(scpi_string=":INSTrument", allowed_values=["SA", "BASIC", "PULSE", "PNOISE"])

    # phase noise application commands
    pn_offset_start = FloatCommand(scpi_string=":LPLot:FREQuency:OFFSet:STARt")
    pn_offset_stop  = FloatCommand(scpi_string=":LPLot:FREQuency:OFFSet:STOP")
    pn_carrier_freq = FloatCommand(scpi_string=":FREQuency:CARRier")

    def __init__(self, resource_name=None, *args, **kwargs):
        super(RSA3045N, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        #if is_valid_ipv4(self.resource_name):
        #    self.resource_name += "::5025::SOCKET"
        super(RSA3045N, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.timeout = 3000 #seem to have trouble timing out on first query sometimes

    def get_axis(self):
        return np.linspace(self.frequency_start, self.frequency_stop, int(self.num_sweep_points))

    def get_trace(self, num=1):
        self.interface.write(':FORM:DATA REAL,32')
        return self.interface.query_binary_values(":TRACE:DATA? TRACE{:d}".format(num),
            datatype="f", is_big_endian=False)

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

    @property
    def marker_Y(self, marker=1):
        """ Queries marker Y-value.

        Args:
            marker (int): Marker index (1-12).
        Returns:
            Trace value at selected marker.
        """
        return self.interface.query(":CALC:MARK{:d}:Y?".format(marker))

    @property
    def marker_X(self, marker=1):
        """ Queries marker X-value.

        Args:
            marker (int): Marker index (1-12).
        Returns:
            X axis value of selected marker.
        """
        return self.interface.query(":CALC:MARK{:d}:X?".format(marker))

    @marker_X.setter
    def marker_X(self, value, marker=1):
        """Sets marker X-value.

        Args:
            value (float): Marker x-axis value to set.
            marker (int):  Marker index (1-2).
        Returns:
            None.
        """
        self.interface.write(":CALC:MARK{:d}:X {:f}".format(marker, value))

    def noise_marker(self, marker=1, enable=True):
        """Set/unset marker as a noise marker for noise figure measurements.

        Args:
            marker (int): Index of marker, [1,12].
            enable (bool): Toggles between noise marker (True) and regular marker (False).
        Returns:
            None.
        """
        if enable:
            self.interface.write(":CALC:MARK{:d}:FUNC NOISe".format(marker))
        else:
            self.interface.write(":CALC:MARK{:d}:FUNC OFF".format(marker))

    def clear_averaging(self):
        self.interface.write(":SENSe:AVERage:CLEar")

    def wait_for_average(self, tt: float = None, timeout: float = 30):
        """
        Wait until the current averaging count reaches the configured count.
        tt is kept only for compatibility with older code.
        """
        aves = float(self.interface.query(':SENSe:AVERage:COUNt?'))
        t0 = time.time()

        while float(self.interface.query(':SENSe:AVERage:COUNt:CURRent?')) < aves:
            if time.time() - t0 > timeout:
                raise TimeoutError("Rigol averaging timed out.")
            time.sleep(0.1)

    def setup_averaging(self, trace=1):
        aves = self.averaging_count
        self.interface.write(f":TRACe{trace}:MODE AVERage")
        self.interface.write(":SENSe:AVERage:MODE REPeat")
        self.interface.write(f":SENSe:AVERage:COUNt {int(aves)}")       

