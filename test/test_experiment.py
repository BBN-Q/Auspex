import unittest
import asyncio
import time
import numpy as np

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment, FloatParameter, Quantity
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print, Passthrough
from pycontrol.filters.average import Average

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: \t%(message)s')
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

    # Quantities
    power = Quantity(unit="Watts")
    clout = Quantity(unit="Trumps")

    # DataStreams
    chan1 = OutputConnector()
    chan2 = OutputConnector()

    # Constants
    samples    = 3
    num_trials = 5

    def init_streams(self):
        # Add "base" data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", list(range(self.samples))))
        descrip.add_axis(DataAxis("trials", list(range(self.num_trials))))
        self.chan1.set_descriptor(descrip)
        self.chan2.set_descriptor(descrip)

    async def run(self):
        for c in self.output_connectors.values():
            for s in c.output_streams:
                s.reset()

        print("Data taker running")
        time_val = 0
        time_step = 0.1
        
        while True:
            #Produce fake noisy sinusoid data every 0.02 seconds until we have 1000 points
            if self.chan1.done():
                print("Data taker finished.")
                break
            await asyncio.sleep(0.001)
            
            print("Stream has filled {} of {} points".format(self.chan1.points_taken, self.chan1.num_points() ))
            data_row = np.sin(2*np.pi*1e3*time_val) + 0.1*np.random.random(self.samples)       
            time_val += time_step
            await self.chan1.push(data_row)
            await self.chan2.push(-data_row*2)

class ExperimentTestCase(unittest.TestCase):
    """
    Tests procedure class
    """

    def test_parameters(self):
        """Check that parameters have been appropriately gathered"""
        self.assertTrue(hasattr(TestExperiment, "_parameters")) # should have parsed these parameters from class dir
        self.assertTrue(len(TestExperiment._parameters) == 2 ) # should have parsed these parameters from class dir
        self.assertTrue(TestExperiment._parameters['freq_1'] == TestExperiment.freq_1) # should contain this parameter
        self.assertTrue(TestExperiment._parameters['freq_2'] == TestExperiment.freq_2) # should contain this parameter

    def test_quantities(self):
        """Check that quantities have been appropriately gathered"""
        self.assertTrue(hasattr(TestExperiment, "_quantities")) # should have parsed these quantities from class dir
        self.assertTrue(len(TestExperiment._quantities) == 2 ) # should have parsed these quantities from class dir
        self.assertTrue(TestExperiment._quantities['power'] == TestExperiment.power) # should contain this quantity
        self.assertTrue(TestExperiment._quantities['clout'] == TestExperiment.clout) # should contain this quantity

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
        avgr            = Average(name="TestAverager")

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
        avgr            = Average(name="TestAverager")

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
        avgr            = Average(name="TestAverager")

        edges = [(exp.chan1, avgr.data),
                 (avgr.partial_average, printer_partial.data),
                 (avgr.final_average, printer_final.data)]

        exp.set_graph(edges)

        self.assertFalse(avgr.data.descriptor is None)
        self.assertFalse(printer_partial.data.descriptor is None)
        self.assertTrue(exp.chan1.descriptor == avgr.data.descriptor)
        self.assertTrue(avgr.partial_average.descriptor == printer_partial.data.descriptor)

    def test_run_simple_graph(self):
        exp     = TestExperiment()
        printer = Print()

        edges = [(exp.chan1, printer.data)]

        exp.set_graph(edges)
        exp.run_loop()

    def test_run_simple_graph_branchout(self):
        exp      = TestExperiment()
        printer1 = Print(name="One")
        printer2 = Print(name="Two")

        edges = [(exp.chan1, printer1.data), (exp.chan1, printer2.data)]

        exp.set_graph(edges)
        exp.run_loop()

    def test_depth(self):
        exp         = TestExperiment()
        passthrough = Passthrough(name="Passthrough")
        printer     = Print(name="Printer")

        edges = [(exp.chan1, passthrough.data_in), (passthrough.data_out, printer.data)]

        exp.set_graph(edges)
        exp.run_loop()

    # def test_reset(self):
    #     exp = TestExperiment()
    #     loop = asyncio.get_event_loop()
    #     tasks = [exp.run()]
    #     loop.run_until_complete(asyncio.wait(tasks))

    #     exp.reset()
    #     loop = asyncio.get_event_loop()
    #     tasks = [exp.run()]
    #     loop.run_until_complete(asyncio.wait(tasks))


    # def test_streams_printing(self):
        # logger.info("Running stream printing test")
        # exp = TestExperiment()
        # pri = Print(name="TestPrint") 

        # self.assertTrue(TestExperiment._output_connectors['chan1'] == TestExperiment.chan1) # should contain this instrument
        # self.assertTrue(TestExperiment._output_connectors['chan2'] == TestExperiment.chan2) # should contain this instrument
        # self.assertTrue(len(exp.chan1.descriptor.axes) == 2)
        # self.assertTrue(len(exp.chan2.descriptor.axes) == 2)
        # self.assertTrue(exp.chan1.descriptor.num_points() == exp.samples*exp.num_trials)

        # repeats = 2
        # exp.chan1.descriptor.add_axis(DataAxis("repeats", list(range(repeats))))
        # self.assertTrue(len(exp.chan1.descriptor.axes) == 3)

        # exp.chan1.connect_to(pri.data)
        # self.assertTrue(not exp.chan1.output_streams[0].done())
        # self.assertTrue(pri.data.input_streams[0].descriptor == exp.chan1.descriptor)
        # self.assertTrue(len(pri.data.input_streams) == 1)

        # with self.assertRaises(ValueError):
        #     pri.data.connect_to(exp.chan2)

        # loop = asyncio.get_event_loop()
        # tasks = [exp.run(), pri.run()]
        # loop.run_until_complete(asyncio.wait(tasks))
        # self.assertTrue(exp.chan1.points_taken == repeats*exp.num_trials*exp.samples)
    #     # self.assertTrue(exp.chan1.output_streams[0].done())

    # def test_streams_averaging(self):
    #     logger.info("Running stream averaging test")

    #     exp             = TestExperiment()
    #     printer_partial = Print(name="Partial")
    #     printer_final   = Print(name="Final")
    #     avgr            = Average(name="TestAverager")

    #     repeats = 4
    #     exp.chan1.descriptor.add_axis(DataAxis("repeats", list(range(repeats))))
    #     self.assertTrue(len(exp.chan1.descriptor.axes) == 3)

    #     exp.chan1.connect_to(avgr.data)
    #     avgr.partial_average.connect_to(printer_partial.data)
    #     strm = avgr.final_average.connect_to(printer_final.data)
    #     self.assertTrue(isinstance(strm, DataStream))
    #     self.assertTrue(exp.chan1.output_streams[0].done() == False)

    #     avgr.axis = 2 # repeats
    #     self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.partial_average.descriptor.axes) )
    #     self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.final_average.descriptor.axes) + 1)
    #     self.assertTrue(avgr.final_average.descriptor.num_points() == exp.num_trials * exp.samples)

    #     avgr.axis = "trials"
    #     self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.partial_average.descriptor.axes) )
    #     self.assertTrue(avgr.final_average.descriptor.num_points() == exp.samples * repeats)

    #     loop = asyncio.get_event_loop()
    #     tasks = [exp.run(), avgr.run(), printer_partial.run(), printer_final.run()]
    #     loop.run_until_complete(asyncio.wait(tasks))

    #     avgr.axis = "samples"
    #     self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.partial_average.descriptor.axes)  )
    #     self.assertTrue(avgr.final_average.descriptor.num_points() == exp.num_trials * repeats)

if __name__ == '__main__':
    unittest.main()