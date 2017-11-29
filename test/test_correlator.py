
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import asyncio
import time
import numpy as np

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.experiment import Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print, Passthrough
from auspex.filters.correlator import Correlator
from auspex.filters.io import DataBuffer
from auspex.log import logger

class CorrelatorExperiment(Experiment):

    # DataStreams
    chan1 = OutputConnector()
    chan2 = OutputConnector()

    # Constants
    samples = 100
    idx_1   = 0
    idx_2   = 0

    # For correlator verification
    vals = 2.0 + np.linspace(0, 10*np.pi, samples)

    def init_streams(self):
        self.chan1.add_axis(DataAxis("samples", list(range(self.samples))))
        self.chan2.add_axis(DataAxis("samples", list(range(self.samples))))

    async def run(self):
        logger.debug("Data taker running (inner loop)")

        while self.idx_1 < self.samples or self.idx_2 < self.samples:

            # Generate random number of samples:
            new_1 = np.random.randint(1,5)
            new_2 = np.random.randint(1,5)

            if self.chan1.points_taken < self.chan1.num_points():
                if self.chan1.points_taken + new_1 > self.chan1.num_points():
                    new_1 = self.chan1.num_points() - self.chan1.points_taken
                await self.chan1.push(self.vals[self.idx_1:self.idx_1+new_1])
                self.idx_1 += new_1
            if self.chan2.points_taken < self.chan2.num_points():
                if self.chan2.points_taken + new_2 > self.chan2.num_points():
                    new_2 = self.chan2.num_points() - self.chan2.points_taken
                await self.chan2.push(self.vals[self.idx_2:self.idx_2+new_2])
                self.idx_2 += new_2

            await asyncio.sleep(0.002)
            logger.debug("Idx_1: %d, Idx_2: %d", self.idx_1, self.idx_2)

class CorrelatorTestCase(unittest.TestCase):

    def test_correlator(self):
        exp   = CorrelatorExperiment()
        corr  = Correlator()
        buff  = DataBuffer()

        edges = [(exp.chan1,   corr.sink),
                 (exp.chan2,   corr.sink),
                 (corr.source, buff.sink)]

        exp.set_graph(edges)
        exp.run_sweeps()

        corr_data     = buff.get_data()['Correlator']
        expected_data = exp.vals*exp.vals
        self.assertTrue(np.abs(np.sum(corr_data - expected_data)) <= 1e-4)


if __name__ == '__main__':
    unittest.main()
