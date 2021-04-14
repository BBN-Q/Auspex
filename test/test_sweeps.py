# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import time
import os
import tempfile
import numpy as np

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import WriteToFile, DataBuffer
from auspex.log import logger

class SweptTestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

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
        self.voltage.add_axis(DataAxis("trials", list(range(self.samples))))

    def __repr__(self):
        return "<SweptTestExperiment>"

    def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        time.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(5) + 0.1*np.random.random(5)
        self.time_val += time_step
        self.voltage.push(data_row)
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

    def test_run(self):
        exp = SweptTestExperiment()
        pri = Print()

        edges = [(exp.voltage, pri.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()

        logger.debug("Run test: logger.debuger ended up with %d points." % pri.sink.input_streams[0].points_taken.value)
        logger.debug("Run test: voltage ended up with %d points." % exp.voltage.num_points())

        self.assertTrue(pri.sink.input_streams[0].points_taken.value == exp.voltage.num_points())

    def test_run_sweep(self):
        exp = SweptTestExperiment()
        pri = Print(name="Printer")

        edges = [(exp.voltage, pri.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()

    def test_unstructured_sweep(self):
        exp = SweptTestExperiment()
        pri = Print()

        edges = [(exp.voltage, pri.sink)]
        exp.set_graph(edges)

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
        self.assertTrue(pri.sink.input_streams[0].points_taken.value == exp.voltage.num_points())

    def test_unstructured_sweep_io(self):

        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperiment()
            pri = Print()
            buf = DataBuffer()
            wri = WriteToFile(tmpdirname+"/test.auspex")

            edges = [(exp.voltage, pri.sink), (exp.voltage, buf.sink), (exp.voltage, wri.sink)]
            exp.set_graph(edges)

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

            self.assertTrue(pri.sink.input_streams[0].points_taken.value == exp.voltage.num_points())

            data, desc, _ = wri.get_data()
            self.assertTrue(np.allclose(desc.axes[0].points, coords))

            data, desc = buf.get_data()
            self.assertTrue(np.allclose(desc.axes[0].points, coords))

if __name__ == '__main__':
    unittest.main()
