from PyDAQmx import *

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Print, WriteToHDF5, Plotter, XYPlotter, Averager
from auspex.instruments import Agilent33220A
from auspex.log import logger

import asyncio
import numpy as np
import time

# Experiment setup
# AWG -> 1 kOhm -> Sample
#     |         |
#    AI1       AI0
# AWG Sync -> PFI0

class IVExperiment(Experiment):

    awg = Agilent33220A("192.168.5.199")

    amplitude = FloatParameter(default=0.1, unit="V")
    frequency  = 167.0 # FloatParameter(default=167.0, unit="Hz")

    sample_rate = 5e5
    num_bursts  = 10

    preamp_gain = 1
    r_ref       = 1e3

    current_input  = OutputConnector(unit="V")
    voltage_sample = OutputConnector(unit="V")

    def init_streams(self):
        descrip = DataStreamDescriptor()
        descrip.data_name='current_input'
        descrip.add_axis(DataAxis("time", np.arange(int(self.sample_rate*self.num_bursts/self.frequency))/self.sample_rate))
        self.current_input.set_descriptor(descrip)

        descrip = DataStreamDescriptor()
        descrip.data_name='voltage_sample'
        descrip.add_axis(DataAxis("time", np.arange(int(self.sample_rate*self.num_bursts/self.frequency))/self.sample_rate))
        self.voltage_sample.set_descriptor(descrip)

    def init_instruments(self):
        # Configure the AWG
        self.awg.output         = False
        self.awg.function       = 'Sine'
        self.awg.load_resistance = self.r_ref
        self.awg.auto_range     = True
        self.awg.amplitude      = self.amplitude.value # Preset to avoid danger
        self.awg.dc_offset      = 0.0
        self.awg.frequency      = self.frequency
        self.awg.burst_state    = True
        self.awg.burst_cycles   = self.num_bursts + 2
        self.awg.trigger_source = "Bus"
        self.awg.output         = True

        self.amplitude.assign_method(self.awg.set_amplitude)
        # self.frequency.assign_method(self.awg.set_frequency)

        # Setup the NIDAQ
        max_voltage = 2.0 #self.amplitude.value*2.0
        self.num_samples_total = int(self.sample_rate*(self.num_bursts+2)/self.frequency)
        self.num_samples_trimmed = int(self.sample_rate*(self.num_bursts)/self.frequency)
        self.trim_len = int(self.sample_rate/self.frequency)
        self.analog_input = Task()
        self.read = int32()
        self.analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_Diff,
            -max_voltage, max_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff,
            -max_voltage, max_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", self.sample_rate, DAQmx_Val_Rising,
            DAQmx_Val_FiniteSamps , self.num_samples_total)
        self.analog_input.CfgInputBuffer(2*self.num_samples_total)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        self.analog_input.StartTask()
        # self.analog_input.SetStartTrigRetriggerable(1)

    def shutdown_instruments(self):
        self.awg.output     = False
        # self.awg.auto_range = True
        try:
            self.analog_input.StopTask()
            self.analog_input.ClearTask()
        except Exception as e:
            logger.warning("Failed to clear DAQ task!")

    async def run(self):
        """This is run for each step in a sweep."""

        self.awg.trigger()

        buf = np.empty(2*self.num_samples_total)
        self.analog_input.ReadAnalogF64(self.num_samples_total, -1, DAQmx_Val_GroupByChannel,
                                        buf, 2*self.num_samples_total, byref(self.read), None)
        await self.current_input.push(buf[self.num_samples_total+self.trim_len:self.num_samples_total+self.trim_len+self.num_samples_trimmed]/self.r_ref)
        await self.voltage_sample.push(buf[self.trim_len:self.trim_len+self.num_samples_trimmed]/self.preamp_gain)
        await asyncio.sleep(0.02)
