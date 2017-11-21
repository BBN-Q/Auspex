# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import asyncio
import time
import numpy as np

from copy import copy, deepcopy

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Print, Passthrough
from auspex.log import logger

class TestInstrument1(SCPIInstrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestInstrument2(SCPIInstrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestInstrument3(SCPIInstrument):
    power = FloatCommand(get_string="power?")
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestExperiment(Experiment):

    # Create instances of instruments
    fake_instr_1 = TestInstrument1("FAKE::RESOURE::NAME")
    fake_instr_2 = TestInstrument2("FAKE::RESOURE::NAME")
    fake_instr_3 = TestInstrument3("FAKE::RESOURE::NAME")

    # Parameters
    freq_1 = FloatParameter(unit="Hz")
    freq_2 = FloatParameter(unit="Hz")

    # DataStreams
    chan1 = OutputConnector()
    chan2 = OutputConnector()

    # Constants
    samples    = 3
    num_trials = 5
    time_val   = 0.0

    def init_instruments(self):
        self.freq_1.assign_method(lambda x: logger.debug("Set: {}".format(x)))
        self.freq_2.assign_method(lambda x: logger.debug("Set: {}".format(x)))

    def init_streams(self):
        # Add "base" data axes
        self.chan1.add_axis(DataAxis("samples", list(range(self.samples))))
        self.chan2.add_axis(DataAxis("trials", list(range(self.num_trials))))

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(self.samples) + 0.1*np.random.random(self.samples)
        self.time_val += time_step
        await self.chan1.push(data_row)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(self.num_trials) + 0.1*np.random.random(self.num_trials)
        await self.chan2.push(data_row)
        logger.debug("Stream pushed points {}.".format(data_row))
        logger.debug("Stream has filled {} of {} points".format(self.chan1.points_taken, self.chan1.num_points() ))

class ExperimentTestCase(unittest.TestCase):

    def test_parameters(self):
        """Check that parameters have been appropriately gathered"""
        self.assertTrue(hasattr(TestExperiment, "_parameters")) # should have parsed these parameters from class dir
        self.assertTrue(len(TestExperiment._parameters) == 2 ) # should have parsed these parameters from class dir
        self.assertTrue(TestExperiment._parameters['freq_1'] == TestExperiment.freq_1) # should contain this parameter
        self.assertTrue(TestExperiment._parameters['freq_2'] == TestExperiment.freq_2) # should contain this parameter

        self.assertTrue(TestExperiment._constants['samples'] == 3)
        self.assertTrue(TestExperiment._constants['num_trials'] == 5)

    def test_instruments(self):
        """Check that instruments have been appropriately gathered"""
        self.assertTrue(hasattr(TestExperiment, "_instruments")) # should have parsed these instruments from class dir
        self.assertTrue(len(TestExperiment._instruments) == 3 ) # should have parsed these instruments from class dir
        
        te = TestExperiment()
        self.assertTrue(te._instruments['fake_instr_1'] == te.fake_instr_1) # should contain this instrument
        self.assertTrue(te._instruments['fake_instr_2'] == te.fake_instr_2) # should contain this instrument
        self.assertTrue(te._instruments['fake_instr_3'] == te.fake_instr_3) # should contain this instrument

    def test_create_graph(self):
        exp         = TestExperiment()
        printer_one = Print(name="One")
        printer_two = Print(name="Two")

        edges = [(exp.chan1, printer_one.sink),
                 (exp.chan2, printer_two.sink)]

        exp.set_graph(edges)

        self.assertTrue(exp.chan1.output_streams[0] == printer_one.sink.input_streams[0])
        self.assertTrue(exp.chan2.output_streams[0] == printer_two.sink.input_streams[0])
        self.assertTrue(len(exp.nodes) == 3)
        self.assertTrue(exp in exp.nodes)
        self.assertTrue(printer_one in exp.nodes)

    def test_graph_parenting(self):
        exp  = TestExperiment()
        pt   = Passthrough()
        prnt = Print(name="One")

        edges = [(exp.chan1, pt.sink),
                 (pt.source, prnt.sink)]

        exp.set_graph(edges)

        self.assertTrue(pt.source.parent == pt)
        self.assertTrue(exp.chan1.output_streams[0].end_connector.parent == pt)
        self.assertTrue(pt.source.output_streams[0].end_connector.parent == prnt)

    def test_update_descriptors(self):
        exp  = TestExperiment()
        pt   = Passthrough()
        prnt = Print(name="One")

        edges = [(exp.chan1, pt.sink),
                 (pt.source, prnt.sink)]

        exp.set_graph(edges)
        exp.update_descriptors()
        self.assertFalse(pt.sink.descriptor is None)
        self.assertFalse(prnt.sink.descriptor is None)
        self.assertTrue(exp.chan1.descriptor == pt.sink.descriptor)

    def test_copy_descriptor(self):
        dsd = DataStreamDescriptor()
        dsd.add_axis(DataAxis("One", [1,2,3,4]))
        dsd.add_axis(DataAxis("Two", [1,2,3,4,5]))
        self.assertTrue(len(dsd.axes)==2)
        self.assertTrue("One" in [a.name for a in dsd.axes])
        dsdc = copy(dsd)
        self.assertTrue(dsd.axes == dsdc.axes)
        ax = dsdc.pop_axis("One")
        self.assertTrue(ax.name == "One")
        self.assertTrue(len(dsdc.axes)==1)
        self.assertTrue(dsdc.axes[0].name == "Two")

    def test_run_simple_graph(self):
        exp     = TestExperiment()
        printer = Print()

        edges = [(exp.chan1, printer.sink)]

        exp.set_graph(edges)
        exp.run_sweeps()

    def test_run_simple_graph_branchout(self):
        exp      = TestExperiment()
        printer1 = Print(name="One")
        printer2 = Print(name="Two")

        edges = [(exp.chan1, printer1.sink), (exp.chan1, printer2.sink)]

        exp.set_graph(edges)
        exp.run_sweeps()

    def test_compressed_streams(self):
        exp      = TestExperiment()
        printer1 = Print(name="One")
        printer2 = Print(name="Two")

        edges = [(exp.chan1, printer1.sink), (exp.chan1, printer2.sink)]

        exp.set_graph(edges)
        exp.set_stream_compression("zlib")
        exp.run_sweeps()

    def test_depth(self):
        exp         = TestExperiment()
        passthrough = Passthrough(name="Passthrough")
        printer     = Print(name="Printer")

        edges = [(exp.chan1, passthrough.sink), (passthrough.source, printer.sink)]

        exp.set_graph(edges)
        exp.run_sweeps()

if __name__ == '__main__':
    unittest.main()
