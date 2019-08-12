from multiprocessing import Queue, Process, Event, Value
from auspex.instruments import AlazarATS9870, AlazarChannel

import time
import unittest
import tempfile
import numpy as np
from QGL import *
import QGL.config
import auspex.config

# Dummy mode
auspex.config.auspex_dummy_mode = True

# Set temporary output directories
awg_dir = tempfile.TemporaryDirectory()
kern_dir = tempfile.TemporaryDirectory()
auspex.config.AWGDir = QGL.config.AWGDir = awg_dir.name
auspex.config.KernelDir = kern_dir.name

from auspex.qubit import *
import bbndb

class AlazarTestCase(unittest.TestCase):

    def basic(self, delay=1.0, averages=10, segments=20, record=1024):
        oc   = Queue()
        exit = Event()
        run  = Event()
        alz  = AlazarATS9870(resource_name="1")
        ch   = AlazarChannel()
        ch.phys_channel = 1
        alz.add_channel(ch)
        alz.gen_fake_data = True
        alz.connect()

        config_dict = {
            'acquireMode': 'digitizer',
            'bandwidth': "Full" ,
            'clockType': "int",
            'delay': 0.0,
            'enabled': True,
            'label': 'Alazar',
            'recordLength': record,
            'nbrSegments': segments,
            'nbrWaveforms': 1,
            'nbrRoundRobins': averages,
            'samplingRate': 500e6,
            'triggerCoupling': "DC",
            'triggerLevel': 100,
            'triggerSlope': "rising",
            'triggerSource': "Ext",
            'verticalCoupling': "DC",
            'verticalOffset': 0.0,
            'verticalScale': 1.0
        }
        alz._lib.setAll(config_dict)
        alz.record_length           = record
        alz.number_acquisitions     = alz._lib.numberAcquisitions
        alz.samples_per_acquisition = alz._lib.samplesPerAcquisition
        alz.number_segments         = segments
        alz.number_waveforms        = 1
        alz.number_averages         = averages
        alz.ch1_buffer              = alz._lib.ch1Buffer
        alz.ch2_buffer              = alz._lib.ch2Buffer
        # print("asdasd", alz.number_averages)

        class OC(object):
            def __init__(self):
                self.queue = Queue()
                self.points_taken = Value('i', 0)
            def push(self, data):
                # print(f"Got data {len(data)}")
                self.queue.put(data)
                self.points_taken.value += data.size
        oc = OC()
        ready = Value('i', 0)
        proc = Process(target=alz.receive_data, args=(ch, oc, exit, ready, run))
        proc.start()
        while ready.value < 1:
            time.sleep(delay)

        run.set()
        alz.wait_for_acquisition(run, timeout=5, ocs=[oc])

        exit.set()
        time.sleep(0.1)
        proc.join(3.0)
        if proc.is_alive():
            proc.terminate()
        alz.disconnect()

        self.assertTrue(oc.points_taken.value == averages*segments*record)

    def test_1sec(self):
        self.basic(delay=1.0)
    def test_100msec(self):
        self.basic(delay=0.1)
    def test_10msec(self):
        self.basic(delay=0.01)
    def test_1sec_100avg(self):
        self.basic(delay=1.0, averages=100)

    def test_qubit_experiment(self):
        cl = ChannelLibrary(db_resource_name=":memory:")
        pl = PipelineManager()
        cl.clear()
        q1     = cl.new_qubit("q1")
        aps2_1 = cl.new_APS2("BBNAPS1", address="192.168.5.101")
        aps2_2 = cl.new_APS2("BBNAPS2", address="192.168.5.102")
        dig_1  = cl.new_Alazar("Alz1", address="1")
        h1     = cl.new_source("Holz1", "HolzworthHS9000", "HS9004A-009-1", power=-30, reference='10MHz')
        h2     = cl.new_source("Holz2", "HolzworthHS9000", "HS9004A-009-2", power=-30, reference='10MHz')
        cl.set_control(q1, aps2_1, generator=h1)
        cl.set_measure(q1, aps2_2, dig_1.ch("1"), generator=h2)
        cl.set_master(aps2_1, aps2_1.ch("m2"))
        pl.create_default_pipeline(buffers=True)
        pl["q1"]["Demodulate"].decimation_factor = 16
        pl["q1"]["Demodulate"]["Integrate"].box_car_stop = 1e-6
        exp = QubitExperiment(RabiAmp(cl["q1"], np.linspace(-1, 1, 51)), averages=100)
        exp.set_fake_data(dig_1, np.cos(np.linspace(-np.pi, np.pi, 51)))
        exp.run_sweeps()
        data, desc = exp.buffers[0].get_data()
        self.assertAlmostEqual(np.abs(data).sum(),459.1,places=0)

if __name__ == '__main__':
    unittest.main() 
