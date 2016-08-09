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
from pycontrol.filters.io import WriteToHDF5_New
from pycontrol.logging import logger

import logging
logger.setLevel(logging.DEBUG)

class TestInstrument1(Instrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(scpi_string=":mode", allowed_values=["A", "B", "C"])

# class UnsweptTestExperiment(Experiment):
#     """Here the run loop merely spews data until it fills up the stream. """

#     # Create instances of instruments
#     fake_instr_1 = TestInstrument1("FAKE::RESOURE::NAME")

#     # Parameters
#     field = FloatParameter(unit="Oe")
#     freq  = FloatParameter(unit="Hz")

#     # DataStreams
#     voltage = OutputConnector()

#     # Constants
#     samples = 5

#     def init_instruments(self):
#         self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
#         self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))

#     def init_streams(self):
#         # Add a "base" data axis: say we are averaging 5 samples per trigger
#         descrip = DataStreamDescriptor()
#         descrip.add_axis(DataAxis("samples", range(self.samples)))
#         self.voltage.set_descriptor(descrip)

#     def __repr__(self):
#         return "<TestExperiment>"

#     async def run(self):
#         logger.debug("Data taker running")
#         time_val = 0
#         time_step = 0.1

#         while True:
#             #Produce fake noisy sinusoid data every 0.02 seconds until we have 1000 points
#             if self.voltage.done():
#                 logger.debug("Data taker finished.")
#                 break
#             await asyncio.sleep(0.002)

#             data_row = np.sin(2*np.pi*time_val)*np.ones(5) + 0.1*np.random.random(5)
#             time_val += time_step
#             await self.voltage.push(data_row)
#             logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

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
#     """
#     Tests sweeping
#     """

    def test_add_sweep(self):
        exp = SweptTestExperiment()
        self.assertTrue(len(exp.voltage.descriptor.axes) == 1)
        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        self.assertTrue(len(exp.voltage.descriptor.axes) == 2)
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        self.assertTrue(len(exp.voltage.descriptor.axes) == 3)
        # sweep_coords = (list(exp._sweep_generator))
        # self.assertTrue(len(sweep_coords) == 3*11)
        # self.assertTrue(len(sweep_coords[0]) == 2)
        # self.assertTrue(exp.voltage.num_points() == 5*len(sweep_coords))
        # print("Is the axis the same?", exp.voltage.)

        print(exp.voltage.descriptor.axes)

    

    # def test_run(self):
    #     exp = UnsweptTestExperiment()
    #     pri = Print()

    #     edges = [(exp.voltage, pri.data)]
    #     exp.set_graph(edges)

    #     exp.init_instruments()
    #     exp.add_sweep(exp.field, np.linspace(0,100.0,11))
    #     exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
    #     exp.run_loop()

    #     logger.debug("Run test: logger.debuger ended up with %d points.", pri.data.input_streams[0].points_taken)
    #     logger.debug("Run test: voltage ended up with %d points.", exp.voltage.output_streams[0].points_taken)

    #     self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())

    # def test_run_sweep(self):
    #     exp = SweptTestExperiment()
    #     pri = Print(name="Printer")

    #     edges = [(exp.voltage, pri.data)]
    #     exp.set_graph(edges)

    #     exp.init_instruments()
    #     exp.add_sweep(exp.field, np.linspace(0,100.0,11))
    #     exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
    #     exp.run_sweeps()

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
        self.assertTrue(pri.points_taken == 5*11*5)

        # logger.debug("Run test: printer ended up with %d points.", pri.data.input_streams[0].points_taken)
        # logger.debug("Run test: voltage ended up with %d points.", exp.voltage.output_streams[0].points_taken)

        # self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())

    # def test_unstructured_sweep(self):
    #     exp = UnsweptTestExperiment()
    #     pri = Print()

    #     edges = [(exp.voltage, pri.data)]
    #     exp.set_graph(edges)

    #     exp.init_instruments()

    #     coords = [[ 0, 0.1],
    #               [10, 4.0],
    #               [15, 2.5],
    #               [40, 4.4],
    #               [50, 2.5],
    #               [60, 1.4],
    #               [65, 3.6],
    #               [66, 3.5],
    #               [67, 3.6],
    #               [68, 1.2]]
    #     exp.add_unstructured_sweep([exp.field, exp.freq], coords)
    #     exp.run_loop()
    #     self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())

    # def test_write_unstructured_sweep(self):
    #     exp = UnsweptTestExperiment()
    #     pri = Print()
    #     if os.path.exists("test_write_unstructured-0000.h5"):
    #         os.remove("test_write_unstructured-0000.h5")
    #     wr  = WriteToHDF5("test_write_unstructured.h5")

    #     edges = [(exp.voltage, pri.data), (exp.voltage, wr.data)]
    #     exp.set_graph(edges)

    #     exp.init_instruments()

    #     coords = np.array([[ 0, 0.1],
    #               [10, 4.0],
    #               [15, 2.5],
    #               [40, 4.4],
    #               [50, 2.5],
    #               [60, 1.4],
    #               [65, 3.6],
    #               [66, 3.5],
    #               [67, 3.6],
    #               [68, 1.2]])
    #     exp.add_unstructured_sweep([exp.field, exp.freq], coords)
    #     exp.run_loop()
    #     self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())
    #     self.assertTrue(os.path.exists("test_write_unstructured-0000.h5"))
    #     with h5py.File("test_write_unstructured-0000.h5", 'r') as f:
    #         self.assertTrue([d.label for d in f['data-0000'].dims] == ['Unstructured', 'samples'])
    #         self.assertTrue([d.keys() for d in f['data-0000'].dims] == [['field', 'freq'], ['samples']])
    #         self.assertTrue(np.sum(f['data-0000'].dims[0]['freq'].value - coords[:,1]) == 0.0)
    #         self.assertTrue(np.sum(f['data-0000'].dims[0]['field'].value - coords[:,0]) == 0.0)
    #     os.remove("test_write_unstructured-0000.h5")

    # def test_run_write_unstructured_sweep(self):
    #     exp = SweptTestExperiment()
    #     pri = Print()
    #     if os.path.exists("test_run_write_unstructured-0000.h5"):
    #         os.remove("test_run_write_unstructured-0000.h5")
    #     wr  = WriteToHDF5("test_run_write_unstructured.h5")


    #     edges = [(exp.voltage, pri.data), (exp.voltage, wr.data)]
    #     exp.set_graph(edges)

    #     exp.init_instruments()

    #     coords = np.array([[ 0, 0.1],
    #               [10, 4.0],
    #               [15, 2.5],
    #               [40, 4.4],
    #               [50, 2.5],
    #               [60, 1.4],
    #               [65, 3.6],
    #               [66, 3.5],
    #               [67, 3.6],
    #               [68, 1.2]])
    #     sweep = exp.add_unstructured_sweep([exp.field, exp.freq], coords)
    #     exp.run_sweeps()

    #     self.assertTrue(pri.data.input_streams[0].points_taken == exp.voltage.num_points())


    #     coords2 = np.array([[ 1, 0.1],
    #                        [11, 4.0],
    #                        [11, 2.5],
    #                        [41, 4.4],
    #                        [51, 2.5],
    #                        [61, 1.4]])
    #     sweep.update_values(coords2)
    #     exp.dur.value = 2
    #     exp.reset()
    #     self.assertTrue(pri.data.input_streams[0].num_points() == len(coords2)*exp.samples)
    #     self.assertFalse(pri.data.input_streams[0].done())
    #     self.assertFalse(wr.data.input_streams[0].done())
    #     self.assertFalse(exp.voltage.output_streams[0].done())
    #     exp.run_sweeps()
    #     self.assertTrue(os.path.exists("test_run_write_unstructured-0000.h5"))
    #     with h5py.File("test_run_write_unstructured-0000.h5", 'r') as f:
    #         self.assertTrue([d.label for d in f['data-0000'].dims] == ['Unstructured', 'samples'])
    #         self.assertTrue([d.keys() for d in f['data-0000'].dims] == [['field', 'freq'], ['samples']])
    #         self.assertTrue(np.sum(f['data-0000'].dims[0]['freq'].value - coords[:,1]) == 0.0)
    #         self.assertTrue(np.sum(f['data-0000'].dims[0]['field'].value - coords[:,0]) == 0.0)
    #         self.assertTrue([d.label for d in f['data-0001'].dims] == ['Unstructured', 'samples'])
    #         self.assertTrue([d.keys() for d in f['data-0001'].dims] == [['field', 'freq'], ['samples']])
    #         self.assertTrue(np.sum(f['data-0001'].dims[0]['freq'].value - coords2[:,1]) == 0.0)
    #         self.assertTrue(np.sum(f['data-0001'].dims[0]['field'].value - coords2[:,0]) == 0.0)

    #     os.remove("test_run_write_unstructured-0000.h5")

    # def test_writehdf5(self):
    #     exp = SweptTestExperiment()
    #     # pr = Print()
    #     if os.path.exists("test_write-0000.h5"):
    #         os.remove("test_write-0000.h5")
    #     wr = WriteToHDF5_New("test_write.h5")

    #     # edges = [(exp.voltage, pr.data), (exp.voltage, wr.data)]
    #     edges = [(exp.voltage, wr.data)]
    #     exp.set_graph(edges)

    #     self.assertTrue(exp.voltage.name == "voltage")

    #     exp.init_instruments()
    #     exp.add_sweep(exp.field, np.linspace(0,100.0,4))
    #     exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
    #     exp.run_loop()
    #     self.assertTrue(os.path.exists("test_write-0000.h5"))
    #     # with h5py.File("test_write-0000.h5", 'r') as f:
    #     #     self.assertTrue([d.label for d in f['data-0000'].dims] == ['freq', 'field', 'samples'])
    #     #     self.assertTrue([d.keys()[0] for d in f['data-0000'].dims] == ['freq', 'field', 'samples'])
    #     #     self.assertTrue(np.sum(f['data-0000'].dims[0][0].value - np.linspace(0,10.0,3)) == 0.0)
    #     #     self.assertTrue(np.sum(f['data-0000'].dims[1]['field'].value - np.linspace(0,100.0,4)) == 0.0)
    #     #     self.assertTrue(np.sum(f['data-0000'].dims[2]['samples'].value - np.arange(0,5)) == 0.0)
    #     #     self.assertTrue("Here the run loop merely spews" in f['data-0000'].attrs['exp_src'])
    #     #     print(f['data-0000'][:])

    #     os.remove("test_write-0000.h5")

if __name__ == '__main__':
    unittest.main()
