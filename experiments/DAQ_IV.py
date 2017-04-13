from PyDAQmx import *

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import WriteToHDF5

from auspex.experiment import FloatParameter, Experiment
from auspex.stream import DataAxis, OutputConnector, DataStreamDescriptor
from auspex.filters.io import WriteToHDF5
from auspex.filters.average import Averager
from auspex.filters.plot import Plotter
from auspex.log import logger

import asyncio
import numpy as np
import time

class IVExperiment(Experiment):

    # Constants (set with attribute access if you want to change these!)
    sample_rate = 1e3
    num_samples = int(4e3)

    min_daq_voltage = -0.1
    max_daq_voltage = 0.1
    sweep_amplitude = 0.1

    voltage_input  = OutputConnector()
    voltage_sample = OutputConnector()

    def init_streams(self):
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage_input'
        descrip.add_axis(DataAxis("time", np.arange(self.num_samples)/self.sample_rate))
        self.voltage_input.set_descriptor(descrip)

        descrip = DataStreamDescriptor()
        descrip.data_name='voltage_sample'
        descrip.add_axis(DataAxis("time", np.arange(self.num_samples)/self.sample_rate))
        self.voltage_sample.set_descriptor(descrip)

    def init_instruments(self):

        # Setup the NIDAQ
        self.analog_input = Task()
        self.read = int32()
        self.analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_Diff,
            self.min_daq_voltage, self.max_daq_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff,
            self.min_daq_voltage, self.max_daq_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", self.sample_rate, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , self.num_samples)
        self.analog_input.CfgInputBuffer(2*self.num_samples)
        # self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        # self.analog_input.SetStartTrigRetriggerable(1)


    async def run(self):
        """This is run for each step in a sweep."""
        self.analog_input.StartTask()
        buf = np.empty(2*self.num_samples)
        self.analog_input.ReadAnalogF64(self.num_samples, -1, DAQmx_Val_GroupByChannel,
                                        buf, 2*self.num_samples, byref(self.read), None)
        await self.voltage_input.push(buf[self.num_samples:])
        await self.voltage_sample.push(buf[:self.num_samples])

        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
