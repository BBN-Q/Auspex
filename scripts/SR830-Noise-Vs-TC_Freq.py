from __future__ import print_function, division
import time
import logging
import sys
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

import numpy as np
import scipy as sp
import pandas as pd

from instruments.stanford import SR830
from sweep import Sweep
from procedure import FloatParameter, Quantity, Procedure


class FieldTest(Procedure):
    frequency = FloatParameter("Lockin Frequency", unit="Hz")
    time_constant = FloatParameter("Time Constant", unit="s")
    noise = Quantity("Noise", unit="V/Hz^1/2")

    lock = SR830("Lockin Amplifier", "GPIB1::9::INSTR")

    def instruments_init(self):
        self.averages = 5
        


        self.lock.channel_1_type = 'X Noise'

        def lockin_measure():
            self.tc_delay = 9*self.lock.tc
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.ch1 for i in range(self.averages)] )

        self.frequency.set_method(self.lock.set_frequency)
        self.time_constant.set_method(self.lock.set_tc)
        self.noise.set_method(lockin_measure)

    def instruments_shutdown(self):
        self.lock.channel_1_type = 'X'   

if __name__ == '__main__':

    proc = FieldTest()

    # Define a sweep over prarameters
    sw = Sweep(proc)
    values = np.append( np.append(np.arange(10,105,5), np.arange(200,1100,50)), np.arange(2000,7000,500)).tolist()
    sw.add_parameter_hack(proc.frequency, values)
    sw.add_parameter_hack(proc.time_constant, [1e-3, 3e-3, 10e-3, 30e-3, 100e-3, 300e-3])

    # Define a writer
    sw.add_writer('SweepFrequencyTC.h5', 'NoiseVsFreqAndTC', proc.noise)

    proc.instruments_init()
    for i in sw:
        logging.info("Freq, TC, Noise: %f, %g, %g" % (proc.frequency.value, proc.time_constant.value, proc.noise.value) )
    proc.instruments_shutdown()
