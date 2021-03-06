# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import os
import time
import datetime
import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt

from auspex.experiment import Experiment, FloatParameter
from auspex.stream import OutputConnector, DataStreamDescriptor
from auspex.filters.plot import Plotter, ManualPlotter
from auspex.filters.io import DataBuffer
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

    def run(self):
        r = np.power(self.amplitude.value,2) + 0.1*np.random.random()
        self.voltage.push(r)
        time.sleep(0.01)

if __name__ == '__main__':

    exp  = TestExperiment()
    exp.leave_plot_server_open = True

    # Create the plotter and the actual traces we'll need
    plt  = ManualPlotter("Manual Plotting Test", x_label='X Thing', y_label='Y Thing')
    plt.add_data_trace("Example Data")
    plt.add_fit_trace("Example Fit")

    buff = DataBuffer()

    edges = [(exp.voltage, buff.sink)]
    exp.set_graph(edges)

    # Create a plotter callback
    def plot_me(plot):
        data, desc = buff.get_data()
        ys = data
        xs = desc.axes[0].points
        plot["Example Data"] = (xs, ys)
        plot["Example Fit"]  = (xs, ys+0.1)

    exp.add_manual_plotter(plt, callback=plot_me)

    exp.add_sweep(exp.amplitude, np.linspace(-5.0, 5.0, 100))

    plt.start()
    exp.run_sweeps()
    plt.stop()

    # ys = buff.get_data()['voltage']
    ys, desc = buff.get_data()

    xs = desc.axes[0].points
    plt["Example Data"] = (xs, ys)
    plt["Example Fit"]  = (xs, ys+0.1)

    # exp.plot_server.stop()
