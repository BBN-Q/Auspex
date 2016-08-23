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

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment
from pycontrol.parameter import FloatParameter
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print
from pycontrol.filters.io import WriteToHDF5
from pycontrol.logging import logger

import logging
logger.setLevel(logging.DEBUG)

class TestInstrument1(Instrument):
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

    def test_add_sweep(self):
        exp = SweptTestExperiment()
        self.assertTrue(len(exp.voltage.descriptor.axes) == 1)
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        self.assertTrue(len(exp.voltage.descriptor.axes) == 2)
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        self.assertTrue(len(exp.voltage.descriptor.axes) == 3)
        print(exp.voltage.descriptor.axes)

    def test_run(self):
        exp = SweptTestExperiment()
        pri = Print()

        edges = [(exp.voltage, pri.data)]
        exp.set_graph(edges)

        exp.init_instruments()
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()

        logger.debug("Run test: logger.debuger ended up with %d points.", pri.data.input_streams[0].points_taken)
        logger.debug("Run test: voltage ended up with %d points.", exp.voltage.output_streams[0].points_taken)

        self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())

    def test_run_sweep(self):
        exp = SweptTestExperiment()
        pri = Print(name="Printer")

        edges = [(exp.voltage, pri.data)]
        exp.set_graph(edges)

        exp.init_instruments()
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()

    def test_run_adaptive_sweep(self):
        exp = SweptTestExperiment()
        pri = Print(name="Printer")

        edges = [(exp.voltage, pri.data)]
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
        self.assertTrue(pri.data.input_streams[0].points_taken == 5*11*5)

    def test_unstructured_sweep(self):
        exp = SweptTestExperiment()
        pri = Print()

        edges = [(exp.voltage, pri.data)]
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
        self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())

if __name__ == '__main__':
    unittest.main()
