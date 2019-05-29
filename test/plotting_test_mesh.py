# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
'''
Test mesh plotting with a Delaunay refinement
'''
import time
import itertools

import numpy as np

from auspex.experiment import Experiment, FloatParameter
from auspex.stream import OutputConnector
from auspex.filters.plot import MeshPlotter
from auspex.filters.io import WriteToFile
from auspex.refine import delaunay_refine_from_file
# import auspex.analysis.switching as sw
# from adapt import refine

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    amplitude = FloatParameter(unit="V")
    duration = FloatParameter(unit="s")

    # DataStreams
    voltage = OutputConnector()

    def init_instruments(self):
        pass

    def init_streams(self):
        pass

    def run(self):
        r = np.sqrt(np.power(self.amplitude.value, 2) + \
                                        np.power(self.duration.value, 2))
        val = 1.0/(1.0 + np.exp(-10.0 * (r - 5.0)))
        self.voltage.push(val)
        time.sleep(0.01)

if __name__ == '__main__':

    EXP = TestExperiment()
    FIG1 = MeshPlotter(name="Plot The Mesh")
    WR = WriteToFile("test_mesh.auspex")

    EDGES = [(EXP.voltage, WR.sink)]
    EXP.set_graph(EDGES)
    EXP.add_direct_plotter(FIG1)

    # Construct the coarse grid
    COURSE_TS = np.linspace(0.0, 10.0, 7)
    COURSE_VS = np.linspace(0.0, 7.5, 7)
    POINTS = [COURSE_TS, COURSE_VS]
    POINTS = list(itertools.product(*POINTS))

    REFINE_FUNC = delaunay_refine_from_file(WR, 'duration', 'amplitude', \
                'voltage', max_points=1000, plotter=FIG1)

    EXP.add_sweep([EXP.duration, EXP.amplitude], POINTS, \
                    refine_func=REFINE_FUNC)
    EXP.run_sweeps()
