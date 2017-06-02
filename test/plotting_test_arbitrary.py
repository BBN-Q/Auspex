# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import os
import time
import datetime
import sys
import itertools

import numpy as np
import h5py
import matplotlib.pyplot as plt

from auspex.experiment import Experiment, FloatParameter
from auspex.stream import OutputConnector, DataStreamDescriptor
from auspex.filters.plot import Plotter, ManualPlotter
from auspex.filters.io import WriteToHDF5, DataBuffer
from auspex.log import logger, logging
# import auspex.analysis.switching as sw
# from adapt import refine

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    amplitude = FloatParameter(unit="V")

    # DataStreams
    voltage = OutputConnector()

    def init_instruments(self):
        pass

    def init_streams(self):
        pass

    async def run(self):
        r = np.power(self.amplitude.value,2) + 0.1*np.random.random()
        await self.voltage.push(r)
        await asyncio.sleep(0.01)

if __name__ == '__main__':

    exp  = TestExperiment()
    plt  = ManualPlotter("Plot Me", x_label='X Thing', y_label='Y Thing')
    buff = DataBuffer()

    edges = [(exp.voltage, buff.sink)]
    exp.set_graph(edges)

    # Create the actual plots we'll need
    data_pts = plt.fig.diamond([],[], color='firebrick')
    fit_line = plt.fig.line([],[], color='navy')

    # Create a plotter callback
    def plot_me(fig, data_pts=data_pts, fit_line=fit_line):
        ys = buff.get_data()['voltage']
        xs = buff.descriptor.axes[0].points
        data_pts.data_source.data = dict(x=xs, y=ys)
        fit_line.data_source.data = dict(x=xs, y=xs)

    exp.add_manual_plotter(plt, callback=plot_me)

    exp.add_sweep(exp.amplitude, np.linspace(-5.0, 5.0, 100))
    exp.run_sweeps()

    ys = buff.get_data()['voltage']
    xs = buff.descriptor.axes[0].points
    data_pts.data_source.data = dict(x=xs, y=ys)
    fit_line.data_source.data = dict(x=xs, y=xs*xs)
