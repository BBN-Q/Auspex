from .instrument import Instrument, Command, FloatCommand, IntCommand
import socket
import time
import numpy as np

#Helper function to check for IPv4 address
#See http://stackoverflow.com/a/11264379
def is_valid_ipv4(ipv4_address):
    try:
        socket.inet_aton(ipv4_address)
        return True
    except:
        return False

class E8363C(Instrument):
    """Agilent E8363C VNA"""

    power              = FloatCommand("Output Power", get_string=":SOURce:POWer:LEVel:IMMediate:AMPLitude?", set_string=":SOURce:POWer:LEVel:IMMediate:AMPLitude {:g}", value_range=(-27, 20))
    frequency_center   = FloatCommand("Sweep Center Frequency", get_string=":SENSe:FREQuency:CENTer?", set_string=":SENSe:FREQuency:CENTer {:g}")
    frequency_span     = FloatCommand("Sweep Frequency Span", get_string=":SENSe:FREQuency:SPAN?", set_string=":SENSe:FREQuency:SPAN {:g}")
    frequency_start    = FloatCommand("Sweep Frequency Start", get_string=":SENSe:FREQuency:STARt?", set_string=":SENSe:FREQuency:STARt {:g}")
    frequency_stop     = FloatCommand("Sweep Frequency Start", get_string=":SENSe:FREQuency:STOP?", set_string=":SENSe:FREQuency:STOP {:g}")
    sweep_num_points   = IntCommand("Sweep Number of Points", get_string=":SENSe:SWEep:POINts?", set_string=":SENSe:SWEep:POINts {:d}")
    averaging_factor   = IntCommand("Number of Averages", get_string=":SENSe1:AVERage:COUNt?", set_string=":SENSe1:AVERage:COUNt {:d}")
    averaging_enable   = Command("Averaging On/Off", get_string=":SENSe1:AVERage:STATe?", set_string=":SENSe1:AVERage:STATe {:c}", value_map={False:"0", True:"1"})
    averaging_complete = Command("Averaging finished", get_string=":STATus:OPERation:AVERaging1:CONDition?", value_map={False:"+0", True:"+2"})

    def __init__(self, name, resource_name, *args, **kwargs):
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(resource_name):
            resource_name += "::5025::SOCKET"
        super(E8363C, self).__init__(name, resource_name, *args, **kwargs)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
        self.interface._resource.timeout = 3000 #seem to have trouble timing out on first query sometimes

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
