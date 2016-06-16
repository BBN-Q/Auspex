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
    frequency = FloatParameter(name="Lockin Frequency", unit="Hz")
    noise = Quantity(name="Noise", unit="V/Hz^1/2")

    lock = SR830("GPIB1::9::INSTR")

    def instruments_init(self):
        self.tc_delay = 9*self.lock.tc
        self.averages = 5
        
        self.lock.channel_1_type = 'X Noise'

        def lockin_measure():
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.ch1 for i in range(self.averages)] )

        self.frequency.set_method(self.lock.set_frequency)
        self.noise.set_method(lockin_measure)

    def instruments_shutdown(self):
        self.lock.channel_1_type = 'X'   

if __name__ == '__main__':

    proc = FieldTest()

    # Define a sweep over prarameters
    sw = Sweep(proc)
    values = np.append( np.append(np.arange(0.1,105,5), np.arange(200,1100,50)), np.arange(2000,7000,500)).tolist()
    sw.add_parameter_hack(proc.frequency, values)

    # Define a writer
    sw.add_writer('SweepFrequency.h5', 'NoiseVsFreq-30ms', proc.noise)

    proc.instruments_init()
    for i in sw:
        logging.info("Freq, Noise: %f, %g" % (proc.frequency.value, proc.noise.value) )
    proc.instruments_shutdown()
