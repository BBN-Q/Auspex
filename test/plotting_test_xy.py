# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0


import os
import numpy as np
import sys

# from auspex.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
from auspex.experiment import Experiment, FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.plot import XYPlotter, Plotter
# from auspex.filters.average import Averager
# from auspex.filters.debug import Print
# from auspex.filters.channelizer import Channelizer
# from auspex.filters.integrator import KernelIntegrator

from auspex.log import logger, logging
logger.setLevel(logging.INFO)

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    field = FloatParameter(unit="Oe")
    amp   = FloatParameter(unit="A")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector(unit="V")
    current = OutputConnector(unit="A")

    # Constants
    samples = 1
    time_val = 0

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        # self.voltage.add_axis(DataAxis("samples", list(range(self.samples))))
        pass

    def __repr__(self):
        return "<SweptTestExperiment>"

    def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.02
        time.sleep(0.02)
        self.time_val += time_step
        self.voltage.push(self.amp.value*np.cos(2*np.pi*self.time_val) + 0.01*np.random.random())
        self.current.push(self.amp.value*np.sin(2*np.pi*self.time_val) + 0.01*np.random.random())
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))


if __name__ == '__main__':

    exp = TestExperiment()
    plt = Plotter(name="Normal Plotter")
    plt_xy = XYPlotter(name="XY Test", x_series=True, series="inner")
    # plt_xy = XYPlotter(name="XY Test")

    edges = [(exp.current, plt_xy.sink_x),
             (exp.voltage, plt_xy.sink_y),
             (exp.voltage, plt.sink)]

    exp.set_graph(edges)
    exp.add_sweep(exp.amp, [1,1.2,1.3])
    # exp.amp.value = 1.0
    exp.add_sweep(exp.field, np.linspace(0,100.0,10))
    exp.run_sweeps()
