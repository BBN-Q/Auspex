import unittest
import asyncio
import numpy as np

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment, FloatParameter, Quantity
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor

class TestInstrument1(Instrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestExperiment(Experiment):

    # Create instances of instruments
    fake_instr_1 = TestInstrument1("FAKE::RESOURE::NAME")

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")

    # DataStreams
    voltage = DataStream(unit="V")

    # Constants
    samples    = 5

    def init_instruments(self):
        self.field.assign_method(lambda x: x)
        self.freq.assign_method(lambda x: x)

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        self.voltage.set_descriptor(descrip)

    async def run(self):
        for s in self._output_streams.values():
            s.reset()

        print("Data taker running")
        time_val = 0
        time_step = 0.1
        
        while True:
            #Produce fake noisy sinusoid data every 0.02 seconds until we have 1000 points
            if self._output_streams['voltage'].done():
                print("Data taker finished.")
                break
            await asyncio.sleep(0.01)
            
            print("Stream has filled {} of {} points".format(self._output_streams['voltage'].points_taken, self._output_streams['voltage'].num_points() ))
            data_row = np.sin(2*np.pi*1e3*time_val) + 0.1*np.random.random(self.samples)       
            time_val += time_step
            await self.voltage.push(data_row)

class SweepTestCase(unittest.TestCase):
    """
    Tests sweeping
    """

    def test_add_sweep(self):
        exp = TestExperiment()
        self.assertTrue(len(exp.voltage.descriptor.axes) == 1)
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        self.assertTrue(len(exp.voltage.descriptor.axes) == 2)
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        self.assertTrue(len(exp.voltage.descriptor.axes) == 3)
        sweep_coords = (list(exp._sweep_generator))
        self.assertTrue(len(sweep_coords) == 3*11)
        self.assertTrue(len(sweep_coords[0]) == 2)
        self.assertTrue(exp.voltage.num_points() == 5*len(sweep_coords))

    def test_run(self):
        exp = TestExperiment()
        exp.init_instruments()
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()
        # printer_final = Print(name="Final") # Example node

if __name__ == '__main__':
    unittest.main()