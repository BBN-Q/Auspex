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
QGL.config.AWGdir = awg_dir

# Create the AWG directory if it doesn't exist
if not os.path.exists(awg_dir):
    os.makedirs(awg_dir)

from auspex.exp_factory import QubitExpFactory
from QGL import *

class QubitExpFactoryTestCase(unittest.TestCase):

    def test_create(self):
        q = QubitFactory("q1")
        exp = QubitExpFactory.create(PulsedSpec(q))
        import ipdb; ipdb.set_trace()
        
    def test_add_qubit_sweep(self):
        q = QubitFactory("q1")
        exp = QubitExpFactory.create(PulsedSpec(q))
        exp.add_qubit_sweep("q1 measure frequency", np.linspace(6e9, 6.5e9, 500))

    def test_run_direct(self):
        q = QubitFactory("q1")
        exp = QubitExpFactory.run(PulsedSpec(q))


if __name__ == '__main__':
    unittest.main()