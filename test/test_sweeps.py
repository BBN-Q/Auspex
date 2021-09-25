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


pl = None
cl = None

# Set temporary output directories
awg_dir = tempfile.TemporaryDirectory()
kern_dir = tempfile.TemporaryDirectory()
import QGL
config.AWGDir = QGL.config.AWGDir = awg_dir.name
config.KernelDir = kern_dir.name

from QGL import *
from auspex.qubit import *
import bbndb

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
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken.value, self.voltage.num_points() ))

class SweepTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global cl, pl

        cl = ChannelLibrary(":memory:")
        pl = PipelineManager()

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

    def test_qubit_metafile_sweep(self):
        cl.clear()
        q1    = cl.new_qubit("q1")
        aps1  = cl.new_APS2("BBNAPS1", address="192.168.5.102")
        aps2  = cl.new_APS2("BBNAPS2", address="192.168.5.103")
        x6_1  = cl.new_X6("X6_1", address="1", record_length=512)
        holz1 = cl.new_source("Holz_1", "HolzworthHS9000", "HS9004A-009-1", power=-30)
        holz2 = cl.new_source("Holz_2", "HolzworthHS9000", "HS9004A-009-2", power=-30)
        cl.set_control(q1, aps1, generator=holz1)
        cl.set_measure(q1, aps2, x6_1[1], generator=holz2)
        cl.set_master(aps1, aps1.ch("m2"))
        cl.commit()
        pl.create_default_pipeline()
        pl.reset_pipelines()
        pl["q1"].clear_pipeline()
        pl["q1"].stream_type = "integrated"
        pl["q1"].create_default_pipeline(buffers=True)

        def mf(sigma):
            q1.pulse_params["sigma"] = sigma
            mf = RabiAmp(q1, np.linspace(-1,1,21))
            return mf

        exp = QubitExperiment(mf(5e-9), averages=5)
        exp.set_fake_data(x6_1, np.linspace(-1, 1, 21), random_mag=0.0)
        exp.add_sweep("q1_sigma", np.linspace(1e-9, 10e-9, 10), metafile_func=mf)
        exp.run_sweeps()

        buf = list(exp.qubits_by_output.keys())[0]
        ax  = buf.input_connectors["sink"].descriptor.axes[0]

        self.assertTrue(buf.done.is_set())
        
        data, desc = buf.get_data()
        self.assertTrue(np.allclose(np.linspace(1e-9, 10e-9, 10), desc.axes[0].points))
        target_dat = np.vstack([np.linspace(-1.0, 1.0, 21)]*10)
        self.assertTrue(np.allclose(target_dat, data.real))


if __name__ == '__main__':
    unittest.main()
