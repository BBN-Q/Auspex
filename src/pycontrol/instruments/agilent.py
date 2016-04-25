from .instrument import Instrument, StringCommand, FloatCommand, IntCommand
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

    power              = FloatCommand(scpi_string=":SOURce:POWer:LEVel:IMMediate:AMPLitude", value_range=(-27, 20))
    frequency_center   = FloatCommand(scpi_string=":SENSe:FREQuency:CENTer")
    frequency_span     = FloatCommand(scpi_string=":SENSe:FREQuency:SPAN")
    frequency_start    = FloatCommand(scpi_string=":SENSe:FREQuency:STARt")
    frequency_stop     = FloatCommand(scpi_string=":SENSe:FREQuency:STOP")
    sweep_num_points   = IntCommand(scpi_string=":SENSe:SWEep:POINts")
    averaging_factor   = IntCommand(scpi_string=":SENSe1:AVERage:COUNt")
    averaging_enable   = StringCommand(get_string=":SENSe1:AVERage:STATe?", set_string=":SENSe1:AVERage:STATe {:c}", value_map={False:"0", True:"1"})
    averaging_complete = StringCommand(get_string=":STATus:OPERation:AVERaging1:CONDition?", value_map={False:"+0", True:"+2"})

    def __init__(self, name, resource_name, *args, **kwargs):
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        if is_valid_ipv4(resource_name):
            resource_name += "::5025::SOCKET"
        super(E8363C, self).__init__(name, resource_name, *args, **kwargs)
        self.name = "Agilent E8363C VNA"
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
