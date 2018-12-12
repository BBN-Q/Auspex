from multiprocessing import Queue, Process, Event, Value
from auspex.instruments import AlazarATS9870, AlazarChannel

import time
import unittest

class AlazarTestCase(unittest.TestCase):

    def basic(self, delay):
        oc   = Queue()
        exit = Event()
        
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
            'recordLength': 1024,
            'nbrSegments': 20,
            'nbrWaveforms': 1,
            'nbrRoundRobins': 10,
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
        alz.record_length           = 1024
        alz.number_acquisitions     = alz._lib.numberAcquisitions
        alz.samples_per_acquisition = alz._lib.samplesPerAcquisition
        alz.number_segments         = 20
        alz.number_waveforms        = 1
        alz.number_averages         = 10
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

        proc = Process(target=alz.receive_data, args=(ch, oc, exit))
        proc.start()
        time.sleep(delay)

        alz.wait_for_acquisition(2.0, [oc])

        exit.set()
        time.sleep(1)
        proc.join(3.0)
        if proc.is_alive():
            proc.terminate()
        alz.disconnect()

    def test_1sec(self):
        self.basic(1.0)
    def test_100msec(self):
        self.basic(0.1)
    def test_10msec(self):
        self.basic(0.01)

if __name__ == '__main__':
    unittest.main()