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

# ----- fix/unitTests_1 (ST-15) delta Start...
# Added the followiing 02 Nov 2018 to test Instrument and filter metaclass load
# introspection minimization (during import)
#
from auspex import config

# Filter out Holzworth warning noise noise by citing the specific instrument[s]
# used for this test.
config.tgtInstrumentClass       = "TestInstrument"

# Filter out Channerlizer noise by citing the specific filters used for this
# test.
config.tgtFilterClass           = {"Plotter", "MeshPlotter", "WriteToHDF5"}

# Uncomment to the following to show the Instrument MetaClass __init__ arguments
# config.bEchoInstrumentMetaInit  = True
#
# ----- fix/unitTests_1 (ST-15) delta Stop.

from auspex.experiment import Experiment, FloatParameter
from auspex.stream import OutputConnector, DataStreamDescriptor
from auspex.filters.plot import Plotter, MeshPlotter
from auspex.filters.io import WriteToHDF5
from auspex.log import logger, logging
from auspex.refine import delaunay_refine_from_file
# import auspex.analysis.switching as sw
# from adapt import refine

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
    coarse_vs = np.linspace(0.0, 7.5, 7)
    points    = [coarse_ts, coarse_vs]
    points    = list(itertools.product(*points))

    refine_func = delaunay_refine_from_file(wr, 'duration', 'amplitude', 'voltage', max_points=1000, plotter=fig1)

    exp.add_sweep([exp.duration, exp.amplitude], points, refine_func=refine_func)
    exp.run_sweeps()
