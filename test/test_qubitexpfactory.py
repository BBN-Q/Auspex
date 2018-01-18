import unittest
import os
import asyncio
import time
import numpy as np
from QGL import *
import QGL.config

# Trick QGL and Auspex into using our local config
# from QGL import config_location
curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = curr_dir.replace('\\', '/')  # use unix-like convention
awg_dir  = os.path.abspath(os.path.join(curr_dir, "AWG" ))
cfg_file = os.path.abspath(os.path.join(curr_dir, "test_measure.yml"))

ChannelLibrary(library_file=cfg_file)
import auspex.config
# Dummy mode
import auspex.config as config
config.auspex_dummy_mode = True

auspex.config.meas_file  = cfg_file
auspex.config.AWGDir     = awg_dir
QGL.config.AWGDir        = awg_dir

# Create the AWG directory if it doesn't exist
if not os.path.exists(awg_dir):
    os.makedirs(awg_dir)

from auspex.exp_factory import QubitExpFactory

class QubitExpFactoryTestCase(unittest.TestCase):

    qubits = ["q1"]
    instrs = ['BBNAPS1', 'BBNAPS2', 'X6-1', 'Holz1', 'Holz2']
    filts  = ['avg-q1-int', 'q1-WriteToHDF5'] #'partial-avg-buff'
    nbr_round_robins = 20

    def test_create(self):
        qq = QubitFactory("q1")
        exp = QubitExpFactory.create(PulsedSpec(qq), save_data = False)
        self.assertTrue(set(self.instrs).issubset(exp._instruments.keys())) # All instruments were loaded
        self.assertTrue(set(self.filts).issubset(exp.filters.keys())) # All filters were loaded
        self.assertTrue(set(self.qubits).issubset(exp.qubits))
        self.assertTrue(len(exp._output_connectors["q1-IntegratedSS"].descriptor.axes) == 1)
        self.assertTrue(len(exp._output_connectors["q1-IntegratedSS"].descriptor.axes[0].points) == self.nbr_round_robins)

    def test_add_qubit_sweep(self):
        qq = QubitFactory("q1")
        exp = QubitExpFactory.create(PulsedSpec(qq), save_data = False)
        exp.add_qubit_sweep("q1 measure frequency", np.linspace(6e9, 6.5e9, 500))
        self.assertTrue(len(exp._output_connectors["q1-IntegratedSS"].descriptor.axes[0].points) == 500)
        self.assertTrue(exp._output_connectors["q1-IntegratedSS"].descriptor.axes[0].points[-1] == 6.5e9)

    def test_run_direct(self):
        qq = QubitFactory("q1")
        exp = QubitExpFactory.run(RabiAmp(qq, np.linspace(-1,1,21)), save_data = False)
        buf = exp.buffers[0]
        ax = buf.descriptor.axes[0]
        self.assertTrue(buf.finished_processing.is_set())
        self.assertTrue(len(buf.out_queue.get()) == 21)
        self.assertTrue((ax.points == np.linspace(-1,1,21)).all())
        self.assertTrue(ax.name == 'amplitude')

    # Figure out how to buffer a partial average for testing...
    @unittest.skip("Partial average for buffers to be fixed")
    def test_final_vs_partial_avg(self):
        qq = QubitFactory("q1")
        exp = QubitExpFactory.run(RabiAmp(qq, np.linspace(-1,1,21)))
        fab = exp.filters['final-avg-buff'].out_queue.get()['Data']
        pab = exp.filters['partial-avg-buff'].out_queue.get()['Data']
        self.assertTrue(np.abs(np.sum(fab-pab)) < 1e-8)

if __name__ == '__main__':
    unittest.main()
