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

from pycontrol.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment
from pycontrol.parameter import FloatParameter
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print
from pycontrol.filters.io import WriteToHDF5
from pycontrol.log import logger

import logging
logger.setLevel(logging.DEBUG)

class TestInstrument1(SCPIInstrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

class SweptTestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Create instances of instruments
    fake_instr_1 = TestInstrument1("FAKE::RESOURE::NAME")

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    def init_instruments(self):
        self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
        self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))
        self.dur.assign_method(lambda x: logger.debug("Duration got value " + str(x)))

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        self.voltage.set_descriptor(descrip)

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(5) + 0.1*np.random.random(5)
        self.time_val += time_step
        await self.voltage.push(data_row)
        logger.debug("Stream pushed points {}.".format(data_row))
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

class SweepTestCase(unittest.TestCase):

    def test_writehdf5(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5-0000.h5"):
            os.remove("test_writehdf5-0000.h5")
        wr = WriteToHDF5("test_writehdf5.h5")

        edges = [(exp.voltage, wr.data)]
        exp.set_graph(edges)

        exp.init_instruments()
        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5-0000.h5"))
        with h5py.File("test_writehdf5-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['data']['voltage'])
            self.assertTrue(np.sum(f['data']['field']) == 5*3*np.sum(np.linspace(0,100.0,4)) )
            self.assertTrue(np.sum(f['data']['freq']) == 5*4*np.sum(np.linspace(0,10.0,3)) )
            self.assertTrue(np.sum(f['data']['samples']) == 3*4*np.sum(np.linspace(0,4,5)) )
            self.assertTrue("Here the run loop merely spews" in f['data'].attrs['exp_src'])
            self.assertTrue(f['data'].attrs['time_val'] == 0)
            self.assertTrue(f['data'].attrs['unit_freq'] == "Hz")
            print(f['data']['voltage'])

        os.remove("test_writehdf5-0000.h5")

    def test_writehdf5_adaptive_sweep(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5_adaptive-0000.h5"):
            os.remove("test_writehdf5_adaptive-0000.h5")
        wr = WriteToHDF5("test_writehdf5_adaptive.h5")

        edges = [(exp.voltage, wr.data)]
        exp.set_graph(edges)

        def rf(sweep_axis, num_points):
            logger.debug("Running refinement function.")
            if sweep_axis.num_points() >= num_points:
                return False
            sweep_axis.points.append(sweep_axis.points[-1]*2)
            return True

        exp.init_instruments()
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, [1.0, 2.0], refine_func=rf, refine_args=[5])
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_adaptive-0000.h5"))
        self.assertTrue(wr.points_taken == 5*11*5)
        os.remove("test_writehdf5_adaptive-0000.h5")

    def test_writehdf5_unstructured_sweep(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5_unstructured-0000.h5"):
            os.remove("test_writehdf5_unstructured-0000.h5")
        wr = WriteToHDF5("test_writehdf5_unstructured.h5")

        edges = [(exp.voltage, wr.data)]
        exp.set_graph(edges)
        exp.init_instruments()

        coords = [[ 0, 0.1],
                  [10, 4.0],
                  [15, 2.5],
                  [40, 4.4],
                  [50, 2.5],
                  [60, 1.4],
                  [65, 3.6],
                  [66, 3.5],
                  [67, 3.6],
                  [68, 1.2]]
        exp.add_sweep([exp.field, exp.freq], coords)
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_unstructured-0000.h5"))
        self.assertTrue(wr.points_taken == 10*5)

        os.remove("test_writehdf5_unstructured-0000.h5")

if __name__ == '__main__':
    unittest.main()