import unittest
import os
import asyncio
import time
import numpy as np

# Dummy mode
import auspex.globals
auspex.globals.auspex_dummy_mode = True

# Trick QGL and Auspex into using our local config
import QGL.config
import auspex.config
curr_dir = os.path.dirname(__file__)
awg_dir  = os.path.join(curr_dir, "AWG" )
cfg_file = os.path.join(curr_dir, "test_config.yml")

QGL.config.configFile    = cfg_file
auspex.config.configFile = cfg_file
QGL.config.AWGDir        = awg_dir
auspex.config.AWGDir     = awg_dir

# Create the AWG directory if it doesn't exist
if not os.path.exists(awg_dir):
    os.makedirs(awg_dir)

from auspex.exp_factory import QubitExpFactory
from QGL import *

class QubitExpFactoryTestCase(unittest.TestCase):

    qubits = ["q1"]
    instrs = ['BBNAPS1', 'BBNAPS2', 'X6-1', 'Holz1', 'Holz2']
    filts  = ['Demod-q1', 'Int-q1', 'avg-q1', 'final-avg-buff', 'partial-avg-buff']

    def test_create(self):
        q = QubitFactory("q1")
        exp = QubitExpFactory.create(PulsedSpec(q))
        self.assertTrue(set(self.instrs).issubset(exp._instruments.keys())) # All instruments were loaded
        self.assertTrue(set(self.filts).issubset(exp.filters.keys())) # All filters were loaded
        self.assertTrue(set(self.qubits).issubset(exp.qubits))
        self.assertTrue(len(exp._output_connectors["q1-RawSS"].descriptor.axes) == 2)
        
    def test_add_qubit_sweep(self):
        q = QubitFactory("q1")
        exp = QubitExpFactory.create(PulsedSpec(q))
        exp.add_qubit_sweep("q1 measure frequency", np.linspace(6e9, 6.5e9, 500))
        self.assertTrue(len(exp._output_connectors["q1-RawSS"].descriptor.axes[0].points) == 500)
        self.assertTrue(exp._output_connectors["q1-RawSS"].descriptor.axes[0].points[-1] == 6.5e9)

    def test_run_direct(self):
        q = QubitFactory("q1")
        exp = QubitExpFactory.run(RabiAmp(q, np.linspace(-1,1,21)))

    # Figure out how to buffer a partial average for testing...
    @unittest.expectedFailure
    def test_final_vs_partial_avg(self):
        q = QubitFactory("q1")
        exp = QubitExpFactory.run(RabiAmp(q, np.linspace(-1,1,21)))
        fab = exp.filters['final-avg-buff'].get_data()['Data']
        pab = exp.filters['partial-avg-buff'].get_data()['Data']
        self.assertTrue(np.abs(np.sum(fab-pab)) < 1e-8)

if __name__ == '__main__':
    unittest.main()