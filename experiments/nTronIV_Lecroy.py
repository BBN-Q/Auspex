from PyDAQmx import *

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Print, WriteToHDF5, Plotter, XYPlotter, Averager
from auspex.instruments import Agilent33500B, HDO6104
from auspex.log import logger


import numpy as np
import time

# Experiment setup
# AWG -> 1 kOhm -> Sample
#     |         |
#    CH1       CH2

class IVExperiment(Experiment):

    awg = Agilent33500B("192.168.5.117")
    lecroy = HDO6104("TCPIP0::192.168.5.118::INSTR")

    amplitude = FloatParameter(default=0.1, unit="V")
    frequency  = 167.0 # FloatParameter(default=167.0, unit="Hz")

    # Parameters for the Lecroy
    sample_rate = 1e9
    num_points = 2.5e6 # Number of points per repeat
    num_bursts  = 10
    repeat      = 1
    delay       = 1 # Delay between repeats

    awg_amplification = 5
    preamp_gain       = 1
    r_ref             = 10.0e3

    voltage_input  = OutputConnector(unit="V")
    voltage_sample = OutputConnector(unit="V")

    def init_streams(self):
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage_input'
        descrip.add_axis(DataAxis("index", np.arange(self.num_points+2)))
        descrip.add_axis(DataAxis("repeat", np.arange(self.repeat)))
        self.voltage_input.set_descriptor(descrip)

        descrip = DataStreamDescriptor()
        descrip.data_name='voltage_sample'
        descrip.add_axis(DataAxis("index", np.arange(self.num_points+2)))
        descrip.add_axis(DataAxis("repeat", np.arange(self.repeat)))
        self.voltage_sample.set_descriptor(descrip)

    def init_instruments(self):
        # Configure the AWG
        self.awg.set_output(False, channel=1)
        self.awg.set_function('Triangle', channel=1)
        self.awg.set_load(50.0, channel=1)
        self.awg.set_auto_range(True, channel=1)
        self.awg.set_amplitude(self.amplitude.value/self.awg_amplification, channel=1) # Preset to avoid danger
        self.awg.set_dc_offset(0.0, channel=1)
        self.awg.set_frequency(self.frequency, channel=1)
        self.awg.set_burst_state(True, channel=1)
        self.awg.set_burst_cycles(self.num_bursts, channel=1)
        self.awg.set_trigger_source("Bus")
        self.awg.set_output_trigger_source(1)
        self.awg.set_output(True, channel=1)
        self.lecroy.set_channel_enabled(True,channel=1)
        self.lecroy.set_channel_enabled(True,channel=2)
        self.lecroy.sample_points = self.num_points

        self.amplitude.assign_method(lambda x: self.awg.set_amplitude(x/self.awg_amplification, channel=1))

    def shutdown_instruments(self):
        self.awg.set_output(False, channel=1)
        self.lecroy.set_channel_enabled(False,channel=1)
        self.lecroy.set_channel_enabled(False,channel=2)

    def run(self):
        """This is run for each step in a sweep."""
        for rep in range(self.repeat):
            self.awg.trigger()
            while not self.lecroy.interface.query("*OPC?") == "1":
                time.sleep(1)
                print("waiting")
            self.voltage_input.push(self.lecroy.fetch_waveform(1)[1])
            self.voltage_sample.push(self.lecroy.fetch_waveform(2)[1])
            time.sleep(self.delay)
