import unittest
import os
import glob
import shutil
import time
import tempfile
import numpy as np

pl = None
cl = None


import QGL.config
import auspex.config
auspex.config.auspex_dummy_mode = True

# Set temporary output directories
awg_dir = tempfile.TemporaryDirectory()
kern_dir = tempfile.TemporaryDirectory()
auspex.config.AWGDir = QGL.config.AWGDir = awg_dir.name
auspex.config.KernelDir = kern_dir.name

from QGL import *
from auspex.qubit import *
import bbndb

def clear_test_data():
    for file in glob.glob("test_*.h5"):
        os.remove(file)
    for direc in glob.glob("test_writehdf5*"):
        shutil.rmtree(direc)

class PipelineTestCase(unittest.TestCase):

    qubits = ["q1"]
    instrs = ['BBNAPS1', 'BBNAPS2', 'X6-1', 'Holz1', 'Holz2']
    filts  = ['avg-q1-int', 'q1-WriteToHDF5'] #'partial-avg-buff'
    nbr_round_robins = 50

    @classmethod
    def setUpClass(cls):
        global cl, pl

        cl = ChannelLibrary(db_resource_name=":memory:")
        pl = PipelineManager()

    def test_create(self):
        cl.clear()
        q1    = cl.new_qubit("q1")
        q2    = cl.new_qubit("q2")
        aps1  = cl.new_APS2("BBNAPS1", address="192.168.5.102")
        aps2  = cl.new_APS2("BBNAPS2", address="192.168.5.103")
        aps3  = cl.new_APS2("BBNAPS3", address="192.168.5.104")
        aps4  = cl.new_APS2("BBNAPS4", address="192.168.5.105")
        x6_1  = cl.new_X6("X6_1", address="1", record_length=512)
        x6_2  = cl.new_X6("X6_2", address="1", record_length=512)
        holz1 = cl.new_source("Holz_1", "HolzworthHS9000", "HS9004A-009-1", power=-30)
        holz2 = cl.new_source("Holz_2", "HolzworthHS9000", "HS9004A-009-2", power=-30)
        holz3 = cl.new_source("Holz_3", "HolzworthHS9000", "HS9004A-009-3", power=-30)
        holz4 = cl.new_source("Holz_4", "HolzworthHS9000", "HS9004A-009-4", power=-30)

        cl.set_control(q1, aps1, generator=holz1)
        cl.set_measure(q1, aps2, x6_1[1], generator=holz2)
        cl.set_control(q2, aps3, generator=holz3)
        cl.set_measure(q2, aps4, x6_2[1], generator=holz4)
        cl.set_master(aps1, aps1.ch("m2"))
        cl.commit()

        pl.create_default_pipeline()
        pl["q1"].clear_pipeline()
        pl["q1"].stream_type = "raw"
        pl.reset_pipelines()

        exp = QubitExperiment(PulsedSpec(q1), averages=5)

        # These should only be related to q1
        self.assertTrue([q1] == exp.measured_qubits)
        self.assertTrue([q1] == exp.controlled_qubits)
        self.assertTrue(set(exp.transmitters) == set([aps1, aps2]))
        self.assertTrue(set(exp.instrument_proxies) == set([aps1, aps2, x6_1, holz1, holz2]))
        self.assertTrue(set(exp.generators) == set([holz1, holz2]))
        self.assertTrue(set(exp.receivers) == set([x6_1]))
        self.assertTrue(len(exp.output_connectors["q1-raw"].descriptor.axes) == 2)
        self.assertTrue(len(exp.output_connectors["q1-raw"].descriptor.axes[0].points) == 5)

    def test_create_transceiver(self):
        cl.clear()
        q1    = cl.new_qubit("q1")
        q2    = cl.new_qubit("q2")
        rack  = cl.new_APS2_rack("APS2Rack", [f"192.168.5.10{i}" for i in range(4)])
        x6_1  = cl.new_X6("X6_1", address="1", record_length=512)
        x6_2  = cl.new_X6("X6_2", address="1", record_length=512)
        holz1 = cl.new_source("Holz_1", "HolzworthHS9000", "HS9004A-009-1", power=-30)
        holz2 = cl.new_source("Holz_2", "HolzworthHS9000", "HS9004A-009-2", power=-30)
        holz3 = cl.new_source("Holz_3", "HolzworthHS9000", "HS9004A-009-3", power=-30)
        holz4 = cl.new_source("Holz_4", "HolzworthHS9000", "HS9004A-009-4", power=-30)

        self.assertTrue(rack.tx("1").label == 'APS2Rack_U1')

        cl.set_control(q1, rack.tx("1"), generator=holz1)
        cl.set_measure(q1, rack.tx("2"), x6_1[1], generator=holz2)
        cl.set_control(q2, rack.tx("3"), generator=holz3)
        cl.set_measure(q2, rack.tx("4"), x6_2[1], generator=holz4)
        cl.set_master(rack.tx("1"), rack.tx("1").ch("m2"))
        cl.commit()

        pl.create_default_pipeline()
        pl["q1"].clear_pipeline()
        pl["q1"].stream_type = "raw"
        pl.reset_pipelines()

        exp = QubitExperiment(PulsedSpec(q1), averages=5)

        # These should only be related to q1
        self.assertTrue([q1] == exp.measured_qubits)
        self.assertTrue([q1] == exp.controlled_qubits)
        self.assertTrue(set(exp.transmitters) == set([rack.tx("1"), rack.tx("2")]))
        self.assertTrue(set(exp.generators) == set([holz1, holz2]))
        self.assertTrue(set(exp.receivers) == set([x6_1]))
        self.assertTrue(len(exp.output_connectors["q1-raw"].descriptor.axes) == 2)
        self.assertTrue(len(exp.output_connectors["q1-raw"].descriptor.axes[0].points) == 5)

    def test_add_qubit_sweep(self):
        cl.clear()
        q1    = cl.new_qubit("q1")
        aps1  = cl.new_APS2("BBNAPS1", address="192.168.5.102")
        aps2  = cl.new_APS2("BBNAPS2", address="192.168.5.103")
        x6_1  = cl.new_X6("X6_1", address="1", record_length=512)
        holz1 = cl.new_source("Holz_1", "HolzworthHS9000", "HS9004A-009-1", power=-30)
        holz2 = cl.new_source("Holz_2", "HolzworthHS9000", "HS9004A-009-2", power=-30)
        cl.set_control(q1, aps1, generator=holz1)
        cl.set_measure(q1, aps2, x6_1[1], generator=holz2)
        cl.set_master(aps1, aps1.ch("m2"))
        pl.create_default_pipeline()
        cl.commit()

        exp = QubitExperiment(PulsedSpec(q1), averages=5)
        exp.add_qubit_sweep(q1, "measure", "frequency", np.linspace(6e9, 6.5e9, 500))
        self.assertTrue(len(exp.output_connectors["q1-integrated"].descriptor.axes[0].points) == 500)
        self.assertTrue(exp.output_connectors["q1-integrated"].descriptor.axes[0].points[-1] == 6.5e9)

    def test_multiple_streamselectors_per_qubit(self):
        cl.clear()
        q1    = cl.new_qubit("q1")
        aps1  = cl.new_APS2("BBNAPS1", address="192.168.5.102")
        aps2  = cl.new_APS2("BBNAPS2", address="192.168.5.103")
        x6_1  = cl.new_X6("X6_1", address="1", record_length=512)
        holz1 = cl.new_source("Holz_1", "HolzworthHS9000", "HS9004A-009-1", power=-30)
        holz2 = cl.new_source("Holz_2", "HolzworthHS9000", "HS9004A-009-2", power=-30)
        cl.set_control(q1, aps1, generator=holz1)
        cl.set_measure(q1, aps2, x6_1[1], generator=holz2)
        cl.set_master(aps1, aps1.ch("m2"))
        pl.create_default_pipeline(buffers=True)
        pl.add_qubit_pipeline("q1", "demodulated", buffers=True)
        cl.commit()

        self.assertTrue(pl["q1 integrated"])
        self.assertTrue(pl["q1 demodulated"])

        exp = QubitExperiment(RabiAmp(q1, np.linspace(-1,1,21)), averages=5)
        exp.set_fake_data(x6_1, np.random.random(21))
        exp.run_sweeps()

        self.assertTrue(len(exp.buffers)==2)
 
    def test_run_pipeline(self):
        cl.clear()
        q1    = cl.new_qubit("q1")
        aps1  = cl.new_APS2("BBNAPS1", address="192.168.5.102")
        aps2  = cl.new_APS2("BBNAPS2", address="192.168.5.103")
        x6_1  = cl.new_X6("X6_1", address="1", record_length=512)
        holz1 = cl.new_source("Holz_1", "HolzworthHS9000", "HS9004A-009-1", power=-30)
        holz2 = cl.new_source("Holz_2", "HolzworthHS9000", "HS9004A-009-2", power=-30)
        cl.set_control(q1, aps1, generator=holz1)
        cl.set_measure(q1, aps2, x6_1[1], generator=holz2)
        cl.set_master(aps1, aps1.ch("m2"))
        cl.commit()
        pl.create_default_pipeline()
        pl.reset_pipelines()
        pl["q1"].clear_pipeline()
        pl["q1"].stream_type = "raw"
        pl["q1"].create_default_pipeline(buffers=True)
        exp = QubitExperiment(RabiAmp(q1, np.linspace(-1,1,21)), averages=5)
        exp.set_fake_data(x6_1, np.random.random(21))
        exp.run_sweeps()
        
        buf = list(exp.qubits_by_output.keys())[0]
        ax  = buf.input_connectors["sink"].descriptor.axes[0]

        # self.assertTrue(buf.done.is_set())
        data, desc = buf.get_data()
        self.assertTrue(len(data) == 21) # Record length * segments * averages (record decimated by 4x)
        self.assertTrue(np.all(np.array(ax.points) == np.linspace(-1,1,21)))
        self.assertTrue(ax.name == 'amplitude')

if __name__ == '__main__':
    unittest.main()
