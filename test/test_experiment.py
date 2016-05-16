import unittest
import asyncio
import numpy as np

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment, FloatParameter, Quantity
from pycontrol.streams.stream import DataStream, DataAxis, DataStreamDescriptor
from pycontrol.streams.io import Printer

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
    samples    = 10
    num_trials = 128

    def init_instruments(self):
        # Add a "base" data axis
        # Say we are averaging 10 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        descrip.add_axis(DataAxis("trials", range(self.num_trials)))
        self.chan1.set_descriptor(descrip)
        self.chan2.set_descriptor(descrip)

    async def run(self):
        print("Data taker running")
        start_time = 0
        time_step  = 20e-6
        
        while True:
            #Produce fake noisy sinusoid data every 0.02 seconds until we have 1000 points
            if self._output_streams['chan1'].done():
                print("Data taker finished.")
                break
            await asyncio.sleep(0.1)
            
            print("Stream has filled {} of {} points".format(self._output_streams['chan1'].points_taken, self._output_streams['chan1'].num_points() ))
            print("Stream reports: {}".format(self._output_streams['chan1'].done()))
            timepts  = start_time + np.arange(0, time_step*self.num_trials, time_step)
            data_row = np.sin(2*np.pi*1e3*timepts)

            data = np.repeat(data_row, self.samples).reshape(-1, self.samples)
            print(data.shape)
            data += 0.1*np.random.random((self.num_trials, self.samples))          
            
            start_time += self.num_trials*time_step
            print("Data taker pushing data")
            await self.chan1.push(data)

class ExperimentTestCase(unittest.TestCase):
    """
    Tests procedure class
    """

    def setUp(self):
        self.exp     = TestExperiment() 
        self.printer = Printer() # Example node

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

    def test_axes(self):
        self.exp.init_instruments()
        self.assertTrue(TestExperiment._output_streams['chan1'] == TestExperiment.chan1) # should contain this instrument
        self.assertTrue(TestExperiment._output_streams['chan2'] == TestExperiment.chan2) # should contain this instrument
        self.assertTrue(len(self.exp.chan1.descriptor.axes) == 2)
        self.assertTrue(len(self.exp.chan2.descriptor.axes) == 2)
        self.assertTrue(self.exp.chan1.descriptor.num_points() == self.exp.samples*self.exp.num_trials)

        repeats = 2
        self.exp.chan1.descriptor.add_axis(DataAxis("repeats", range(repeats)))
        self.assertTrue(len(self.exp.chan1.descriptor.axes) == 3)

        self.printer.add_input_stream(self.exp.chan1)
        self.assertTrue(self.printer.input_streams[0].descriptor == self.exp.chan1.descriptor)
        self.assertTrue(len(self.printer.input_streams) == 1)

        with self.assertRaises(ValueError):
            self.printer.add_input_stream(self.exp.chan2)

        loop = asyncio.get_event_loop()
        tasks = [self.exp.run(), self.printer.run()]
        loop.run_until_complete(asyncio.wait(tasks))
        

if __name__ == '__main__':
    unittest.main()