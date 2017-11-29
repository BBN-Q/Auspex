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

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import DataBuffer
from auspex.log import logger

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
        self.current.add_axis(DataAxis("samples", list(range(self.samples))))

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(5) + 0.1*np.random.random(5)
        self.time_val += time_step
        await self.voltage.push(data_row)
        await self.current.push(data_row*2.0)
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

class BufferTestCase(unittest.TestCase):

    def test_buffer(self):
        exp = SweptTestExperiment()
        db  = DataBuffer()

        edges = [(exp.voltage, db.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()

        data = db.get_data()
        self.assertTrue(len(data) == 4*3*5)
        self.assertTrue(len(data['field']) == 4*3*5)

    def test_buffer_multi(self):
        exp = SweptTestExperiment()
        db  = DataBuffer()

        edges = [(exp.voltage, db.sink), (exp.current, db.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()

        data = db.get_data()
        self.assertTrue(len(data) == 4*3*5)
        self.assertTrue(len(data['current']) == 4*3*5)
        self.assertTrue(len(data['voltage']) == 4*3*5)
        self.assertTrue(len(data['field']) == 4*3*5)

    def test_buffer_metadata(self):
        exp = SweptTestExperimentMetadata()
        db  = DataBuffer()

        edges = [(exp.voltage, db.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.field, np.linspace(0,100.0,4))
        exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
        exp.run_sweeps()

        data = db.get_data()
        self.assertTrue(len(data) == 4*3*5)
        self.assertTrue(len(data['samples_metadata']) == 4*3*5)

if __name__ == '__main__':
    unittest.main()