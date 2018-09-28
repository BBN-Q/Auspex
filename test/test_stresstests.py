import unittest
import os
import glob
import shutil
import time
import numpy as np
from QGL import *
import QGL.config
import h5py

# Trick QGL and Auspex into using our local config
# from QGL import config_location
curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = curr_dir.replace('\\', '/')  # use unix-like convention
awg_dir  = os.path.abspath(os.path.join(curr_dir, "AWG" ))
cfg_file = os.path.abspath(os.path.join(curr_dir, "test_measure_stress.yml"))

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

def clear_test_data():
    for file in glob.glob("stresstest-*.h5"):
        os.remove(file)
    for direc in glob.glob("stresstest-*"):
        shutil.rmtree(direc)

class StressTestIO(unittest.TestCase):

    qubits = ["q1"]
    instrs = ['BBNAPS1', 'BBNAPS2', 'X6-1', 'Holz1', 'Holz2']
    filts  = ['avg-q1-int', 'q1-WriteToHDF5'] #'partial-avg-buff'
    nbr_round_robins = 250

    def run_for_num_segs(self, num_segs, filters_to_keep):
        clear_test_data()
        qq = QubitFactory("q1")
        exp = QubitExpFactory.create(RabiAmp(qq, np.linspace(-1,1,num_segs)), save_data = True)

        exp.output_connectors['q1-RawSS'].output_streams = [os for os in exp.output_connectors['q1-RawSS'].output_streams if os.end_connector.parent.filter_name in filters_to_keep]
        exp.nodes = [n for n in exp.nodes if not hasattr(n, 'filter_name') or n.filter_name in filters_to_keep]
        exp.graph.edges = [e for e in exp.graph.edges if e.end_connector.parent.filter_name in filters_to_keep]

        # exp.dashboard = True
        exp.run_sweeps()
        return exp

    def run_buf_test(self, num_segs):
        exp = self.run_for_num_segs(num_segs, ['q1-raw-buff'])
        buf = exp.buffers[0]

        self.assertTrue(buf.done.is_set())
        self.assertTrue(len(buf.output_data) == num_segs*250*8192//4)
        self.assertTrue(len(np.nonzero(buf.output_data["Data"])[0]) == len(buf.output_data))
        clear_test_data()

    def run_hdf5_test(self, num_segs):
        exp = self.run_for_num_segs(num_segs, ['q1-WriteRaw'])

        with h5py.File(exp.writers[0].filename.value, 'r') as f:
            buf = f['q1-main/data/Data'][:]

        self.assertTrue(len(buf) == num_segs*250*8192//4)
        self.assertTrue(len(np.nonzero(buf)[0]) == len(buf))
        clear_test_data()


    def test_buffer_010segments(self):
        """Generates 100 MB of data in buffer"""
        self.run_buf_test(10)

    def test_buffer_050segments(self):
        """Generates 500 MB of data in buffer"""
        self.run_buf_test(50)

    def test_buffer_100segments(self):
        """Generates 1000 MB of data in buffer"""
        self.run_buf_test(100)

    def test_buffer_200segments(self):
        """Generates 2000 MB of data in buffer"""
        self.run_buf_test(200)

    def test_hdf5_010segments(self):
        """Generates 100 MB of data in hdf5"""
        self.run_hdf5_test(10)

    def test_hdf5_050segments(self):
        """Generates 500 MB of data in hdf5"""
        self.run_hdf5_test(50)

    def test_hdf5_100segments(self):
        """Generates 1000 MB of data in hdf5"""
        self.run_hdf5_test(100)

    def test_hdf5_200segments(self):
        """Generates 2000 MB of data in hdf5"""
        self.run_hdf5_test(200)


if __name__ == '__main__':
    unittest.main()
