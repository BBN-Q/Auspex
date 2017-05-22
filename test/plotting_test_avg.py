# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import os
import numpy as np
import sys

from auspex.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
from auspex.experiment import Experiment, FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.plot import Plotter
from auspex.filters.average import Averager
from auspex.filters.debug import Print
from auspex.filters.channelizer import Channelizer
from auspex.filters.integrator import KernelIntegrator

from auspex.log import logger, logging
logger.setLevel(logging.INFO)

class TestInstrument(SCPIInstrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(name="enumerated mode", scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Create instances of instruments
    fake_instr_1 = TestInstrument("FAKE::RESOURE::NAME")

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")

    # DataStreams
    voltage = OutputConnector(unit="V")

    def init_instruments(self):
        pass

    def init_streams(self):
        self.voltage.add_axis(DataAxis("xs", np.arange(100)))
        self.voltage.add_axis(DataAxis("ys", np.arange(100)))
        self.voltage.add_axis(DataAxis("repeats", np.arange(500)))

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
       
        for _ in range(500):
            await asyncio.sleep(0.01)
            data = np.zeros((100,100))
            data[25:75, 25:75] = 1.0 
            data = data + 25*np.random.random((100,100))
            await self.voltage.push(data.flatten())

if __name__ == '__main__':

    exp = TestExperiment()
    avg = Averager("repeats", name="Averager")
    pl1 = Plotter(name="Partial", plot_dims=2, plot_mode="real", palette="Spectral11")
    pl2 = Plotter(name="Final", plot_dims=2, plot_mode="real", palette="Spectral11")

    edges = [
            (exp.voltage, avg.sink),
            (avg.partial_average, pl1.sink),
            (avg.final_average, pl2.sink)
            ]

    avg.update_interval = 0.2
    pl1.update_interval = 0.2
    pl2.update_interval = 0.2

    exp.set_graph(edges)
    exp.init_instruments()
    exp.run_sweeps()
