# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import asyncio
import os
import numpy as np
import h5py

from auspex.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import WriteToHDF5
from auspex.log import logger

import logging
logger.setLevel(logging.INFO)

class SweptTestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector()
    current = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    def init_instruments(self):
        self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
        self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))
        self.dur.assign_method(lambda x: logger.debug("Duration got value " + str(x)))

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        self.voltage.add_axis(DataAxis("samples", list(range(self.samples))))

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(5) + 0.1*np.random.random(5)
        self.time_val += time_step
        await self.voltage.push(data_row)
        await self.current.push(np.sin(2*np.pi*self.time_val) + 0.1*np.random.random(1))
        logger.debug("Stream pushed points {}.".format(data_row))
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

class SweptTestExperimentMetadata(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector()
    current = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    def init_instruments(self):
        self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
        self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))
        self.dur.assign_method(lambda x: logger.debug("Duration got value " + str(x)))

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        self.voltage.add_axis(DataAxis("samples", np.array([0,1,2,np.nan,np.nan]), metadata=["data", "data", "data", "0", "1"]))

    def __repr__(self):
        return "<SweptTestExperimentMetadata>"

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(5) + 0.1*np.random.random(5)
        self.time_val += time_step
        await self.voltage.push(data_row)
        await self.current.push(np.sin(2*np.pi*self.time_val) + 0.1*np.random.random(1))
        logger.debug("Stream pushed points {}.".format(data_row))
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

class SweptTestExperiment2(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector()
    current = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    def init_instruments(self):
        self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
        self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))
        self.dur.assign_method(lambda x: logger.debug("Duration got value " + str(x)))

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        self.voltage.add_axis(DataAxis("samples", list(range(self.samples))))
        self.current.add_axis(DataAxis("samples", list(range(self.samples))))

    def __repr__(self):
        return "<SweptTestExperiment2>"

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(5) + 0.1*np.random.random(5)
        self.time_val += time_step
        await self.voltage.push(data_row)
        await self.current.push(-0.1*data_row)
        logger.debug("Stream pushed points {}.".format(data_row))
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))


class WriteTestCase(unittest.TestCase):

    def test_writehdf5(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5-0000.h5"):
            os.remove("test_writehdf5-0000.h5")
        wr = WriteToHDF5("test_writehdf5.h5")

        edges = [(exp.voltage, wr.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5-0000.h5"))
        with h5py.File("test_writehdf5-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['main']['data']['voltage'])
            self.assertTrue(np.sum(f['main']['data']['field']) == 5*3*np.sum(np.linspace(0,100.0,4)) )
            self.assertTrue(np.sum(f['main']['data']['freq']) == 5*4*np.sum(np.linspace(0,10.0,3)) )
            self.assertTrue(np.sum(f['main']['data']['samples']) == 3*4*np.sum(np.linspace(0,4,5)) )
            self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
            self.assertTrue(f['main']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['main']['data'].attrs['unit_freq'] == "Hz")

        os.remove("test_writehdf5-0000.h5")

    def test_writehdf5_metadata(self):
        exp = SweptTestExperimentMetadata()
        if os.path.exists("test_writehdf5_metadata-0000.h5"):
            os.remove("test_writehdf5_metadata-0000.h5")
        wr = WriteToHDF5("test_writehdf5_metadata.h5")

        edges = [(exp.voltage, wr.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_metadata-0000.h5"))
        with h5py.File("test_writehdf5_metadata-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['main']['data']['voltage'])
            self.assertTrue(np.sum(f['main']['data']['field']) == 5*3*np.sum(np.linspace(0,100.0,4)) )
            self.assertTrue(np.sum(f['main']['data']['freq']) == 5*4*np.sum(np.linspace(0,10.0,3)) )
            self.assertTrue(np.sum(np.isnan(f['main']['data']['samples'])) == 3*4*2 )
            self.assertTrue(np.sum(f['main']['data']['samples_metadata'] == b'data') == 4*3*3)
            self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
            self.assertTrue(f['main']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['main']['data'].attrs['unit_freq'] == "Hz")

        os.remove("test_writehdf5_metadata-0000.h5")

    def test_writehdf5_metadata_unstructured(self):
        exp = SweptTestExperimentMetadata()
        if os.path.exists("test_writehdf5_metadata_unstructured-0000.h5"):
            os.remove("test_writehdf5_metadata_unstructured-0000.h5")
        wr = WriteToHDF5("test_writehdf5_metadata_unstructured.h5")

        edges = [(exp.voltage, wr.sink)]
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
                  [np.nan, np.nan],
                  [np.nan, np.nan],
                  [np.nan, np.nan]]
        md = ["data"]*9 + ["a","b","c"]
        exp.add_sweep([exp.field, exp.freq], coords, metadata=md)
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_metadata_unstructured-0000.h5"))
        with h5py.File("test_writehdf5_metadata_unstructured-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['main']['data']['voltage'])
            self.assertTrue(np.sum(np.isnan(f['main']['data']['field'])) == 3*5 )
            self.assertTrue(np.sum(np.isnan(f['main']['data']['freq'])) == 3*5 )
            self.assertTrue(np.sum(np.isnan(f['main']['data']['samples'])) == 3*4*2 )
            self.assertTrue(np.sum(f['main']['data']['field+freq_metadata'] == b'a') == 5)
            self.assertTrue(np.sum(f['main']['data']['field+freq_metadata'] == b'b') == 5)
            self.assertTrue(np.sum(f['main']['data']['field+freq_metadata'] == b'c') == 5)
            self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
            self.assertTrue(f['main']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['main']['data'].attrs['unit_freq'] == "Hz")

        os.remove("test_writehdf5_metadata_unstructured-0000.h5")

    def test_writehdf5_metadata_unstructured_adaptive(self):
        exp = SweptTestExperimentMetadata()
        if os.path.exists("test_writehdf5_metadata_unstructured_adaptive-0000.h5"):
            os.remove("test_writehdf5_metadata_unstructured_adaptive-0000.h5")
        wr = WriteToHDF5("test_writehdf5_metadata_unstructured_adaptive.h5")

        edges = [(exp.voltage, wr.sink)]
        exp.set_graph(edges)
        coords = [[ 0, 0.1],
                  [10, 4.0],
                  [15, 2.5],
                  [40, 4.4],
                  [50, 2.5],
                  [60, 1.4],
                  [65, 3.6],
                  [66, 3.5],
                  [67, 3.6]]
        md = ["data"]*9

        async def rf(sweep_axis, exp):
            logger.debug("Running refinement function.")
            if sweep_axis.num_points() >= 12:
                return False
            sweep_axis.metadata = sweep_axis.metadata + ["a","b","c"]
            sweep_axis.add_points([
                  [np.nan, np.nan],
                  [np.nan, np.nan],
                  [np.nan, np.nan]])
            logger.debug("Sweep points now: {}.".format(sweep_axis.points))
            return True

        exp.add_sweep([exp.field, exp.freq], coords, metadata=md, refine_func=rf)
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_metadata_unstructured_adaptive-0000.h5"))
        with h5py.File("test_writehdf5_metadata_unstructured_adaptive-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['main']['data']['voltage'])
            self.assertTrue(np.sum(np.isnan(f['main']['data']['field'])) == 3*5 )
            self.assertTrue(np.sum(np.isnan(f['main']['data']['freq'])) == 3*5 )
            self.assertTrue(np.sum(np.isnan(f['main']['data']['samples'])) == 3*4*2 )
            self.assertTrue(np.sum(f['main']['data']['field+freq_metadata'] == b'a') == 5)
            self.assertTrue(np.sum(f['main']['data']['field+freq_metadata'] == b'b') == 5)
            self.assertTrue(np.sum(f['main']['data']['field+freq_metadata'] == b'c') == 5)
            self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
            self.assertTrue(f['main']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['main']['data'].attrs['unit_freq'] == "Hz")
            self.assertTrue(len(f['main']['field+freq']) == 12)

        os.remove("test_writehdf5_metadata_unstructured_adaptive-0000.h5")

    def test_samefile_writehdf5(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_samefile_writehdf5-0000.h5"):
            os.remove("test_samefile_writehdf5-0000.h5")
        wr1 = WriteToHDF5("test_samefile_writehdf5.h5", "group1")
        wr2 = WriteToHDF5("test_samefile_writehdf5.h5", "group2")

        edges = [(exp.voltage, wr1.sink), (exp.current, wr2.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_samefile_writehdf5-0000.h5"))
        with h5py.File("test_samefile_writehdf5-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['group1']['data']['voltage'])
            self.assertTrue(np.sum(f['group1']['data']['field']) == 5*3*np.sum(np.linspace(0,100.0,4)) )
            self.assertTrue(np.sum(f['group1']['data']['freq']) == 5*4*np.sum(np.linspace(0,10.0,3)) )
            self.assertTrue(np.sum(f['group1']['data']['samples']) == 3*4*np.sum(np.linspace(0,4,5)) )
            self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
            self.assertTrue(f['group1']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['group1']['data'].attrs['unit_freq'] == "Hz")
            self.assertTrue(0.0 not in f['group2']['data']['current'])
            self.assertTrue(np.sum(f['group2']['data']['field']) == 3*np.sum(np.linspace(0,100.0,4)) )
            self.assertTrue(np.sum(f['group2']['data']['freq']) == 4*np.sum(np.linspace(0,10.0,3)) )
            self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
            self.assertTrue(f['group2']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['group2']['data'].attrs['unit_freq'] == "Hz")

        os.remove("test_samefile_writehdf5-0000.h5")

    def test_writehdf5_complex(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5_complex-0000.h5"):
            os.remove("test_writehdf5_complex-0000.h5")
        wr = WriteToHDF5("test_writehdf5_complex.h5")

        edges = [(exp.voltage, wr.sink)]
        exp.set_graph(edges)
        exp.voltage.descriptor.dtype = np.complex64
        exp.update_descriptors()
        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_complex-0000.h5"))
        with h5py.File("test_writehdf5_complex-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['main']['data']['voltage'])
            self.assertTrue(np.sum(f['main']['data']['field']) == 5*3*np.sum(np.linspace(0,100.0,4)) )
            self.assertTrue(np.sum(f['main']['data']['freq']) == 5*4*np.sum(np.linspace(0,10.0,3)) )
            self.assertTrue(np.sum(f['main']['data']['samples']) == 3*4*np.sum(np.linspace(0,4,5)) )
            self.assertTrue(f['main']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['main']['data'].attrs['unit_freq'] == "Hz")

        os.remove("test_writehdf5_complex-0000.h5")

    def test_writehdf5_multiple_streams(self):
        exp = SweptTestExperiment2()
        if os.path.exists("test_writehdf5_mult-0000.h5"):
            os.remove("test_writehdf5_mult-0000.h5")
        wr = WriteToHDF5("test_writehdf5_mult.h5")

        edges = [(exp.voltage, wr.sink), (exp.current, wr.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_mult-0000.h5"))
        with h5py.File("test_writehdf5_mult-0000.h5", 'r') as f:
            self.assertTrue(0.0 not in f['main']['data']['voltage'])
            self.assertTrue(np.sum(f['main']['data']['field']) == 5*3*np.sum(np.linspace(0,100.0,4)) )
            self.assertTrue(np.sum(f['main']['data']['freq']) == 5*4*np.sum(np.linspace(0,10.0,3)) )
            self.assertTrue(np.sum(f['main']['data']['samples']) == 3*4*np.sum(np.linspace(0,4,5)) )
            self.assertTrue(f['main']['data'].attrs['time_val'] == 0)
            self.assertTrue(f['main']['data'].attrs['unit_freq'] == "Hz")

        os.remove("test_writehdf5_mult-0000.h5")

    def test_writehdf5_adaptive_sweep(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5_adaptive-0000.h5"):
            os.remove("test_writehdf5_adaptive-0000.h5")
        wr = WriteToHDF5("test_writehdf5_adaptive.h5")

        edges = [(exp.voltage, wr.sink)]
        exp.set_graph(edges)

        async def rf(sweep_axis, exp):
            num_points = 5
            logger.debug("Running refinement function.")
            if sweep_axis.num_points() >= num_points:
                return False
            # sweep_axis.points.append(sweep_axis.points[-1]*2)
            sweep_axis.add_points(sweep_axis.points[-1]*2)
            return True

        exp.add_sweep(exp.field, np.linspace(0,100.0,11))
        exp.add_sweep(exp.freq, [1.0, 2.0], refine_func=rf)
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_adaptive-0000.h5"))
        self.assertTrue(wr.points_taken == 5*11*5)
        with h5py.File("test_writehdf5_adaptive-0000.h5", 'r') as f:
            self.assertTrue(len(f['main']['freq']) == 5)

        os.remove("test_writehdf5_adaptive-0000.h5")

    def test_writehdf5_unstructured_sweep(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5_unstructured-0000.h5"):
            os.remove("test_writehdf5_unstructured-0000.h5")
        wr = WriteToHDF5("test_writehdf5_unstructured.h5")

        edges = [(exp.voltage, wr.sink)]
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
        self.assertTrue(os.path.exists("test_writehdf5_unstructured-0000.h5"))
        self.assertTrue(wr.points_taken == 10*5)

        os.remove("test_writehdf5_unstructured-0000.h5")

    def test_writehdf5_adaptive_unstructured_sweep(self):
        exp = SweptTestExperiment()
        if os.path.exists("test_writehdf5_adaptive_unstructured-0000.h5"):
            os.remove("test_writehdf5_adaptive_unstructured-0000.h5")
        wr = WriteToHDF5("test_writehdf5_adaptive_unstructured.h5")

        edges = [(exp.voltage, wr.sink)]
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

        async def rf(sweep_axis, exp):
            logger.debug("Running refinement function.")
            if sweep_axis.num_points() >= 30:
                return False

            first_points = np.array(sweep_axis.points[-10:])
            new_points   = first_points.copy()
            new_points[:,0] += 100
            new_points[:,1] += 10

            sweep_axis.add_points(new_points)
            logger.debug("Sweep points now: {}.".format(sweep_axis.points))
            return True

        exp.add_sweep([exp.field, exp.freq], coords, refine_func=rf)
        exp.run_sweeps()
        self.assertTrue(os.path.exists("test_writehdf5_adaptive_unstructured-0000.h5"))
        self.assertTrue(wr.points_taken == 10*5*3)

        os.remove("test_writehdf5_adaptive_unstructured-0000.h5")

if __name__ == '__main__':
    unittest.main()
