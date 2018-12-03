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

_bNO_METACLASS_INTROSPECTION_CONSTRAINTS = True  # Use original dummy flag logic
#_bNO_METACLASS_INTROSPECTION_CONSTRAINTS = False # Enable instrument and filter introspection constraints

if _bNO_METACLASS_INTROSPECTION_CONSTRAINTS:
    #
    # The original unittest quieting logic
    import auspex.config as config
    config.auspex_dummy_mode = True
    #
else:
    # ----- fix/unitTests_1 (ST-15) delta Start...
    # Added the followiing 05 Nov 2018 to test Instrument and filter metaclass load
    # introspection minimization (during import)
    #
    from auspex import config

    # Filter out Holzworth warning noise noise by citing the specific instrument[s]
    # used for this test.
    config.tgtInstrumentClass       = ""  # No Instruments

    # Filter out Channerlizer noise by citing the specific filters used for this
    # test.
    # ...Actually Print, Channelizer, and KernelIntegrator are NOT used in this test;
    # hence commented them out, below, as well.
    config.tgtFilterClass           = {"Print", "Passthrough", "DataBuffer", "Averager"}

    # Uncomment to the following to show the Instrument MetaClass __init__ arguments
    # config.bEchoInstrumentMetaInit  = True
    #
    # ----- fix/unitTests_1 (ST-15) delta Stop.


from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print, Passthrough
from auspex.filters.io import DataBuffer
from auspex.filters.average import Averager
from auspex.log import logger

class TestExperiment(Experiment):

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
        self.chan1.add_axis(DataAxis("samples", list(range(self.samples))))
        self.chan1.add_axis(DataAxis("trials", list(range(self.num_trials))))
        self.chan2.add_axis(DataAxis("samples", list(range(self.samples))))
        self.chan2.add_axis(DataAxis("trials", list(range(self.num_trials))))

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.ones(self.samples*self.num_trials) + 0.1*np.random.random(self.samples*self.num_trials)
        self.time_val += time_step
        await self.chan1.push(data_row)
        logger.debug("Stream pushed points {}.".format(data_row))
        logger.debug("Stream has filled {} of {} points".format(self.chan1.points_taken, self.chan1.num_points() ))

class VarianceExperiment(Experiment):

    # DataStreams
    chan1 = OutputConnector()

    # Constants
    samples = 3
    trials  = 5
    repeats = 10
    idx     = 0

    # For variance comparison
    vals = np.random.random((samples*trials*repeats))

    def init_streams(self):
        self.chan1.add_axis(DataAxis("samples", list(range(self.samples))))
        self.chan1.add_axis(DataAxis("trials", list(range(self.trials))))
        self.chan1.add_axis(DataAxis("repeats", list(range(self.repeats))))

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        await asyncio.sleep(0.002)
        data_row = self.vals[self.idx:self.idx+(self.samples*self.trials*self.repeats)]
        self.idx += (self.samples*self.trials*self.repeats)
        await self.chan1.push(data_row)
        logger.debug("Stream pushed points {}.".format(data_row))
        logger.debug("Stream has filled {} of {} points".format(self.chan1.points_taken, self.chan1.num_points() ))

class AverageTestCase(unittest.TestCase):

    def test_final_average_runs(self):
        exp             = TestExperiment()
        printer_final   = Print(name="Final")
        avgr            = Averager('trials', name="TestAverager")

        edges = [(exp.chan1, avgr.sink),
                 (avgr.final_average, printer_final.sink)]

        exp.set_graph(edges)
        exp.run_sweeps()

    def test_final_variance_runs(self):
        exp             = VarianceExperiment()
        printer_final   = Print(name="Final")
        avgr            = Averager('repeats', name="TestAverager")
        var_buff        = DataBuffer(name='Variance Buffer')
        mean_buff       = DataBuffer(name='Mean Buffer')

        edges = [(exp.chan1,           avgr.sink),
                 (avgr.final_variance, printer_final.sink),
                 (avgr.final_variance, var_buff.sink),
                 (avgr.final_average,  mean_buff.sink)]

        exp.set_graph(edges)
        exp.run_sweeps()

        var_data  = var_buff.get_data()['Variance'].reshape(var_buff.descriptor.data_dims())
        mean_data = mean_buff.get_data()['chan1'].reshape(mean_buff.descriptor.data_dims())
        orig_data = exp.vals.reshape(exp.chan1.descriptor.data_dims())

        self.assertTrue(np.abs(np.sum(mean_data - np.mean(orig_data, axis=0))) <= 1e-3)
        self.assertTrue(np.abs(np.sum(var_data - np.var(orig_data, axis=0, ddof=1))) <= 1e-3)


    def test_partial_average_runs(self):
        exp             = TestExperiment()
        printer_partial = Print(name="Partial")
        avgr            = Averager('trials', name="TestAverager")

        edges = [(exp.chan1, avgr.sink),
                 (avgr.partial_average, printer_partial.sink)]

        exp.set_graph(edges)
        exp.run_sweeps()

    def test_sameness(self):
        exp             = TestExperiment()
        printer_partial = Print(name="Partial")
        printer_final   = Print(name="Final")
        avgr            = Averager(name="TestAverager", axis='freq_1')

        edges = [(exp.chan1, avgr.sink),
                 (avgr.partial_average, printer_partial.sink),
                 (avgr.final_average, printer_final.sink)]
        exp.set_graph(edges)

        exp.add_sweep(exp.freq_1, np.linspace(0,9,10))
        exp.run_sweeps()

if __name__ == '__main__':
    unittest.main()
