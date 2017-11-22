# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import asyncio
import os
import numpy as np
import h5py
from adapt.refine import refine_1D

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import WriteToHDF5
from auspex.log import logger
from auspex.analysis.io import load_from_HDF5

config.load_meas_file(config.find_meas_file())

class SweptTestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    temperature = FloatParameter(unit="K")

    # DataStreams
    resistance = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        await asyncio.sleep(0.002)

        def ideal_tc(t, tc=9.0, k=20.0):
            return t*1.0/(1.0 + np.exp(-k*(t-tc)))

        await self.resistance.push(ideal_tc(self.temperature.value))
        # logger.debug("Stream pushed points {}.".format(data_row))
        # logger.debug("Stream has filled {} of {} points".format(self.resistance.points_taken, self.resistance.num_points() ))

class Adapt1DTestCase(unittest.TestCase):

    def test_writehdf5_1D_adaptive_sweep(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5_1D_adaptive-0000.h5"):
            os.remove("test_writehdf5_1D_adaptive-0000.h5")
        wr = WriteToHDF5("test_writehdf5_1D_adaptive.h5")

        edges = [(exp.resistance, wr.sink)]
        exp.set_graph(edges)

        async def rf(sweep_axis, exp):
            logger.debug("Running refinement function.")
            temps = wr.group['data']['temperature'][:]
            ress  = wr.group['data']['resistance'][:]
            logger.debug("Temps: {}".format(temps))
            logger.debug("Ress: {}".format(ress))

            new_temps = refine_1D(temps, ress, all_points=False, criterion="difference", threshold = "one_sigma")

            logger.debug("New temperature values: {}".format(new_temps))
            if new_temps.size + temps.size > 15:
                return False

            sweep_axis.add_points(new_temps)
            logger.debug("Axis points are now: {}".format(sweep_axis.points))
            return True

        exp.add_sweep(exp.temperature, np.linspace(0,20,5), refine_func=rf)
        exp.run_sweeps()
        
        self.assertTrue(os.path.exists("test_writehdf5_1D_adaptive-0000.h5"))
        
        expected_data = np.array([ 0., 5.,10.,15.,20., 7.5, 8.75, 9.375, 9.0625, 8.90625, 8.984375,12.5,17.5, 9.0234375, 8.945312])
        data, desc = load_from_HDF5(wr.filename.value, reshape=False)
        actual_data = data['main']['temperature']
        self.assertTrue(actual_data.size == expected_data.size)
        os.remove("test_writehdf5_1D_adaptive-0000.h5")

if __name__ == '__main__':
    unittest.main()
