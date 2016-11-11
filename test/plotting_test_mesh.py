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
from auspex.filters.plot import Plotter, MeshPlotter
from auspex.filters.io import WriteToHDF5
from auspex.log import logger, logging

import auspex.analysis.switching as sw
from adapt import refine

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    amplitude = FloatParameter(unit="V")
    duration  = FloatParameter(unit="s")

    # DataStreams
    voltage = OutputConnector()

    def init_instruments(self):
        pass

    def init_streams(self):
        pass

    async def run(self):
        r = np.sqrt(np.power(self.amplitude.value,2) + np.power(self.duration.value,2))
        val = 1.0/(1.0 + np.exp(-10.0*(r-5.0)))
        await self.voltage.push(val)
        await asyncio.sleep(0.01)

if __name__ == '__main__':

    exp  = TestExperiment()
    fig1 = MeshPlotter(name="Plot The Mesh")
    wr   = WriteToHDF5("test_mesh.h5")

    edges = [(exp.voltage, wr.sink)]
    exp.set_graph(edges)
    exp.add_direct_plotter(fig1)

    # Construct the coarse grid
    coarse_ts = np.linspace(0.0, 10.0, 7)
    coarse_vs = np.linspace(0.0, 10.0, 7)
    points    = [coarse_ts, coarse_vs]
    points    = list(itertools.product(*points))

    async def refine_func(sweep_axis, max_points=500):
        vals = wr.data.value['voltage']
        amps = wr.data.value['amplitude']
        durs = wr.data.value['duration']
        points = np.array([durs,amps]).transpose()

        new_points = refine.refine_scalar_field(points, vals, all_points=False,
                                    criterion="integral", threshold = "one_sigma")
        if len(points) + len(new_points) > max_points:
            print("Reached maximum points ({}).".format(max_points))
            return False
        print("Reached {} points.".format(len(points) + len(new_points)))
        sweep_axis.add_points(new_points)
        exp.update_descriptors()

        # Plot previous mesh
        mesh, scale_factors = sw.scaled_Delaunay(points)
        xs   = durs[mesh.simplices]/scale_factors[0]
        ys   = amps[mesh.simplices]/scale_factors[1]
        avg_vals = [np.mean(row) for row in vals[mesh.simplices]]

        await exp.push_to_plot(fig1, [xs,ys,avg_vals])

        time.sleep(0.1)
        return True

    exp.add_sweep([exp.duration, exp.amplitude], points, refine_func=refine_func)
    exp.run_sweeps()
