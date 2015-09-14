from __future__ import print_function, division
import time
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd

from instruments.kepco import BOP2020M
from instruments.stanford import SR830
from sweep import Sweep
from procedure import FloatParameter, Quantity, Procedure



class FieldTest(Procedure):
    current = FloatParameter("Supply Current", unit="A")
    voltage = Quantity("Magnitude", unit="V")

    def run(self):
        for param in self._parameters:
            self._parameters[param].push()
            time.sleep(0.3*5)
        for quant in self._quantities:
            self._quantities[quant].measure()

if __name__ == '__main__':
    bop = BOP2020M("Kepco Power Supply", "GPIB1::1::INSTR")
    lock = SR830("Lockin Amplifier", "GPIB1::9::INSTR")

    bop.output = True
    proc = FieldTest()
    proc.current.set_method(bop.set_current)
    proc.voltage.set_method(lock.get_magnitude)

    # Define a sweep over prarameters
    sw = Sweep(proc)
    values = np.append(np.arange(-15, 15.1, 0.5), np.arange(14.99,-15,-0.5)).tolist()
    sw.add_parameter_hack(proc.current, values)

    # Define a writer
    sw.add_writer('SweepField.h5', 'VvsH', proc.voltage)

    for i in sw:
    	logging.info("Current, Lockin Magnitude: %f" % (proc.current.value) )

    # proc.current.value = 0.0
    