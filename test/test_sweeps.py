import unittest
import asyncio
import os
import numpy as np
import h5py

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment, FloatParameter, Quantity
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print
from pycontrol.filters.io import WriteToHDF5

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: \t%(message)s')
logger.setLevel(logging.DEBUG)

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
    voltage = OutputConnector()

    # Constants
    samples = 5

    def init_instruments(self):
        self.field.assign_method(lambda x: x)
        self.freq.assign_method(lambda x: x)

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        self.voltage.set_descriptor(descrip)

    async def run(self):
        print("Data taker running")
        time_val = 0
        time_step = 0.1
        
        while True:
            #Produce fake noisy sinusoid data every 0.02 seconds until we have 1000 points
            if self.voltage.done():
                print("Data taker finished.")
                break
            await asyncio.sleep(0.002)
            
            data_row = np.sin(2*np.pi*time_val)*np.ones(5)
            time_val += time_step
            await self.voltage.push(data_row)
            print("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))


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
        pri = Print()

        edges = [(exp.voltage, pri.data)]
        exp.set_graph(edges)
        
        exp.init_instruments()
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_loop()

        logger.debug("Run test: printer ended up with %d points.", pri.data.input_streams[0].points_taken)
        logger.debug("Run test: voltage ended up with %d points.", exp.voltage.output_streams[0].points_taken)

        self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())

    def test_writehdf5(self):
        exp = TestExperiment()
        pr = Print()
        wr = WriteToHDF5("test_write.h5")
        self.assertTrue(os.path.exists("0000-test_write.h5"))

        edges = [(exp.voltage, pr.data), (exp.voltage, wr.data)]
        exp.set_graph(edges)

        self.assertTrue(exp.voltage.name == "voltage")

        exp.init_instruments()
        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_loop()

        with h5py.File("0000-test_write.h5", 'r') as f:
            self.assertTrue([d.label for d in f['data'].dims] == ['freq', 'field', 'samples'])
            self.assertTrue([d.keys()[0] for d in f['data'].dims] == ['freq', 'field', 'samples'])
            self.assertTrue(np.sum(f['data'].dims[0][0].value - np.linspace(0,10.0,3)) == 0.0)
            self.assertTrue(np.sum(f['data'].dims[1]['field'].value - np.linspace(0,100.0,4)) == 0.0)
            self.assertTrue(np.sum(f['data'].dims[2]['samples'].value - np.arange(0,5)) == 0.0)
            print(f['data'][:])

        os.remove("0000-test_write.h5")

if __name__ == '__main__':
    unittest.main()