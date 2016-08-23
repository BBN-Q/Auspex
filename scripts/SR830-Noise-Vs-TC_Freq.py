# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

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


class NoiseTest(Procedure):
    frequency = FloatParameter(name="Lockin Frequency", unit="Hz")
    time_constant = FloatParameter(name="Time Constant", unit="s")
    noise = Quantity(name="Noise", unit="V/Hz^1/2")

    lock = SR830("GPIB1::9::INSTR")

    def init_instruments(self):
        self.averages = 1
        self.lock.channel_1_type = 'X Noise'

        def lockin_measure():
            time.sleep(self.time_constant.value*50)
            return np.mean( [self.lock.ch1 for i in range(self.averages)] )

        self.frequency.assign_method(self.lock.set_frequency)
        self.time_constant.assign_method(self.lock.set_tc)
        self.noise.assign_method(lockin_measure)

    def run(self):
        for param in self._parameters:
            self._parameters[param].push()
        for quant in self._quantities:
            self._quantities[quant].measure()

    def shutdown_instruments(self):
        self.lock.channel_1_type = 'R'

if __name__ == '__main__':

    proc = NoiseTest()

    # Define a sweep over prarameters
    sw = Sweep(proc)
    sw.add_parameter(proc.frequency, np.logspace(2, 5, 100))
    proc.time_constant.value = 100e-3

    # Define a writer
    sw.add_plotter('X Noise vs. Frequency', proc.frequency, proc.noise, color="navy", line_width=2)
    #sw.add_writer('SweepFrequencyTC.h5', 'NoiseVsFreqAndTC', proc.noise)

    sw.run()

    # proc.instruments_init()
    # for i in sw:
    #     logging.info("Freq, TC, Noise: %f, %g, %g" % (proc.frequency.value, proc.time_constant.value, proc.noise.value) )
    # proc.instruments_shutdown()
