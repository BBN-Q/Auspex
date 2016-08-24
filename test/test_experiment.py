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

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment
from pycontrol.parameter import FloatParameter
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print, Passthrough
from pycontrol.filters.average import Average
from pycontrol.log import logger, logging
logger.setLevel(logging.DEBUG)

class TestInstrument1(Instrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestInstrument2(Instrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestInstrument3(Instrument):
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
        self.freq_1.assign_method(lambda x: print("Set: {}".format(x)))
        self.freq_2.assign_method(lambda x: print("Set: {}".format(x)))

    def init_streams(self):
        # Add "base" data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", list(range(self.samples))))
        descrip.add_axis(DataAxis("trials", list(range(self.num_trials))))
        self.chan1.set_descriptor(descrip)
        self.chan2.set_descriptor(descrip)

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(self.samples*self.num_trials) + 0.1*np.random.random(self.samples*self.num_trials)
        self.time_val += time_step
        await self.chan1.push(data_row)
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
        self.assertTrue(TestExperiment._instruments['fake_instr_1'] == TestExperiment.fake_instr_1) # should contain this instrument
        self.assertTrue(TestExperiment._instruments['fake_instr_2'] == TestExperiment.fake_instr_2) # should contain this instrument
        self.assertTrue(TestExperiment._instruments['fake_instr_3'] == TestExperiment.fake_instr_3) # should contain this instrument

    def test_create_graph(self):
        exp             = TestExperiment()
        printer_partial = Print(name="Partial")
        printer_final   = Print(name="Final")
        avgr            = Average('samples', name="TestAverager")

        edges = [(exp.chan1, avgr.data),
                 (avgr.partial_average, printer_partial.data),
                 (avgr.final_average, printer_final.data)]

        exp.set_graph(edges)

        self.assertTrue(exp.chan1.output_streams[0] == avgr.data.input_streams[0])
        self.assertTrue(avgr.partial_average.output_streams[0] == printer_partial.data.input_streams[0])
        self.assertTrue(avgr.final_average.output_streams[0] == printer_final.data.input_streams[0])
        self.assertTrue(len(exp.nodes) == 4)
        self.assertTrue(exp in exp.nodes)
        self.assertTrue(avgr in exp.nodes)

    def test_graph_parenting(self):
        exp             = TestExperiment()
        printer_partial = Print(name="Partial")
        printer_final   = Print(name="Final")
        avgr            = Average('samples', name="TestAverager")

        edges = [(exp.chan1, avgr.data),
                 (avgr.partial_average, printer_partial.data),
                 (avgr.final_average, printer_final.data)]

        exp.set_graph(edges)

        self.assertTrue(avgr.partial_average.parent == avgr)
        self.assertTrue(avgr.final_average.parent == avgr)
        self.assertTrue(exp.chan1.output_streams[0].end_connector.parent == avgr)
        self.assertTrue(avgr.partial_average.output_streams[0].end_connector.parent == printer_partial)

    def test_update_descriptors(self):
        exp             = TestExperiment()
        printer_partial = Print(name="Partial")
        printer_final   = Print(name="Final")
        avgr            = Average('samples', name="TestAverager")

        edges = [(exp.chan1, avgr.data),
                 (avgr.partial_average, printer_partial.data),
                 (avgr.final_average, printer_final.data)]

        exp.set_graph(edges)
        exp.init_instruments()

        self.assertFalse(avgr.data.descriptor is None)
        self.assertFalse(printer_partial.data.descriptor is None)
        self.assertTrue(exp.chan1.descriptor == avgr.data.descriptor)
        self.assertTrue(avgr.partial_average.descriptor.axes[0].name == 'Partial Averages')
        self.assertTrue(avgr.partial_average.descriptor.axes[0].points == [0,1,2])

    def test_run_simple_graph(self):
        exp     = TestExperiment()
        printer = Print()

        edges = [(exp.chan1, printer.data)]

        exp.set_graph(edges)
        exp.init_instruments()
        exp.run_sweeps()

    def test_run_simple_graph_branchout(self):
        exp      = TestExperiment()
        printer1 = Print(name="One")
        printer2 = Print(name="Two")

        edges = [(exp.chan1, printer1.data), (exp.chan1, printer2.data)]

        exp.set_graph(edges)
        exp.init_instruments()
        exp.run_sweeps()

    def test_compressed_streams(self):
        exp      = TestExperiment()
        printer1 = Print(name="One")
        printer2 = Print(name="Two")

        edges = [(exp.chan1, printer1.data), (exp.chan1, printer2.data)]

        exp.set_graph(edges)
        exp.set_stream_compression("zlib")
        exp.init_instruments()
        exp.run_sweeps()

    def test_depth(self):
        exp         = TestExperiment()
        passthrough = Passthrough(name="Passthrough")
        printer     = Print(name="Printer")

        edges = [(exp.chan1, passthrough.data_in), (passthrough.data_out, printer.data)]

        exp.set_graph(edges)
        exp.init_instruments()
        exp.run_sweeps()

if __name__ == '__main__':
    unittest.main()
