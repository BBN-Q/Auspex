import unittest
import asyncio
import numpy as np

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment, FloatParameter, Quantity
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor
from pycontrol.filters.debug import Print
from pycontrol.filters.average import Average

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
    chan1 = DataStream()
    chan2 = DataStream()

    # Constants
    samples    = 5
    num_trials = 10

    def init_streams(self):
        # Add a "base" data axis
        # Say we are averaging 10 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        descrip.add_axis(DataAxis("trials", range(self.num_trials)))
        self.chan1.set_descriptor(descrip)
        self.chan2.set_descriptor(descrip)

    async def run(self):
        for s in self._output_streams.values():
            s.reset()

        print("Data taker running")
        time_val = 0
        time_step = 0.1
        
        while True:
            #Produce fake noisy sinusoid data every 0.02 seconds until we have 1000 points
            if self._output_streams['chan1'].done():
                print("Data taker finished.")
                break
            await asyncio.sleep(0.001)
            
            print("Stream has filled {} of {} points".format(self._output_streams['chan1'].points_taken, self._output_streams['chan1'].num_points() ))
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

    def test_streams_printing(self):
        exp     = TestExperiment()
        printer2 = Print() # Example node

        exp.init_instruments()
        self.assertTrue(TestExperiment._output_streams['chan1'] == TestExperiment.chan1) # should contain this instrument
        self.assertTrue(TestExperiment._output_streams['chan2'] == TestExperiment.chan2) # should contain this instrument
        self.assertTrue(len(exp.chan1.descriptor.axes) == 2)
        self.assertTrue(len(exp.chan2.descriptor.axes) == 2)
        self.assertTrue(exp.chan1.descriptor.num_points() == exp.samples*exp.num_trials)

        repeats = 4
        exp.chan1.descriptor.add_axis(DataAxis("repeats", range(repeats)))
        self.assertTrue(len(exp.chan1.descriptor.axes) == 3)

        printer2.data.add_input_stream(exp.chan1)
        self.assertTrue(printer2.data.input_streams[0].descriptor == exp.chan1.descriptor)
        self.assertTrue(len(printer2.data.input_streams) == 1)

        with self.assertRaises(ValueError):
            printer2.data.add_input_stream(exp.chan2)

        loop = asyncio.get_event_loop()
        tasks = [exp.run(), printer2.run()]
        loop.run_until_complete(asyncio.wait(tasks))

        self.assertTrue(exp.chan1.points_taken == repeats*exp.num_trials*exp.samples)
        
    def test_streams_averaging(self):
        exp             = TestExperiment()
        printer_partial = Print(name="Partial") # Example node
        printer_final   = Print(name="Final") # Example node
        avgr            = Average(name="TestAverager")
        strm_partial    = DataStream(name="Partial")
        # strm_final      = DataStream(name="Final")

        exp.init_instruments()

        repeats = 4
        exp.chan1.descriptor.add_axis(DataAxis("repeats", range(repeats)))
        self.assertTrue(len(exp.chan1.descriptor.axes) == 3)

        avgr.data.add_input_stream(exp.chan1)

        # Testing directly adding streams
        avgr.partial_average.add_output_stream(strm_partial)
        printer_partial.data.add_input_stream(strm_partial)

        # Test convenience functions for doing so
        strm = avgr.final_average.connect_to(printer_final.data)
        self.assertTrue(isinstance(strm, DataStream))

        avgr.axis = 2 # repeats
        avgr.update_descriptors()
        self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.partial_average.output_streams[0].descriptor.axes) )
        self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.final_average.output_streams[0].descriptor.axes) + 1)
        self.assertTrue(avgr.final_average.output_streams[0].descriptor.num_points() == exp.num_trials * exp.samples)

        avgr.axis = "trials"
        avgr.update_descriptors()
        self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.partial_average.output_streams[0].descriptor.axes) )
        self.assertTrue(avgr.final_average.output_streams[0].descriptor.num_points() == exp.samples * repeats)

        loop = asyncio.get_event_loop()
        tasks = [exp.run(), avgr.run(), printer_partial.run(), printer_final.run()]
        loop.run_until_complete(asyncio.wait(tasks))

        avgr.axis = "samples"
        avgr.update_descriptors()
        self.assertTrue(len(exp.chan1.descriptor.axes) == len(avgr.partial_average.output_streams[0].descriptor.axes)  )
        self.assertTrue(avgr.final_average.output_streams[0].descriptor.num_points() == exp.num_trials * repeats)

if __name__ == '__main__':
    unittest.main()