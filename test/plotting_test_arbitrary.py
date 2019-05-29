# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
'''
Test for plotting arbitrary data with a manual plotter
'''
import time

import numpy as np

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
        r = np.power(self.amplitude.value, 2) + 0.1*np.random.random()
        self.voltage.push(r)
        time.sleep(0.01)

if __name__ == '__main__':

    EXP = TestExperiment()

    # Create the plotter and the actual traces we'll need
    PLT = ManualPlotter("Manual Plotting Test", x_label='X Thing', \
                            y_label='Y Thing')
    PLT.add_data_trace("Example Data")
    PLT.add_fit_trace("Example Fit")

    BUFF = DataBuffer()

    EDGES = [(EXP.voltage, BUFF.sink)]
    EXP.set_graph(EDGES)

    # Create a plotter callback
    def plot_me(plot):
        ''' Manual plotter callback'''
        ys = BUFF.output_data['voltage']
        xs = BUFF.descriptor.axes[0].points
        plot["Example Data"] = (xs, ys)
        plot["Example Fit"] = (xs, ys+0.1)

    EXP.add_manual_plotter(PLT, callback=plot_me)

    EXP.add_sweep(EXP.amplitude, np.linspace(-5.0, 5.0, 100))
    EXP.run_sweeps()

    # ys = BUFF.get_data()['voltage']
    YS, DESC = BUFF.get_data()

    XS = DESC.axes[0].points
    PLT["Example Data"] = (XS, YS)
    PLT["Example Fit"] = (XS, YS+0.1)
