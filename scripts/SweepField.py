from __future__ import print_function, division
import time
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

import numpy as np
import scipy as sp
import pandas as pd

from instruments.kepco import BOP2020M
from instruments.stanford import SR830
from instruments.magnet import Electromagnet
from instruments.hall_probe import HallProbe
from sweep import Sweep
from procedure import FloatParameter, Quantity, Procedure


class FieldTest(Procedure):
    set_field = FloatParameter("Set Field", unit="G")
    field   = Quantity("Field", unit="G")
    voltage = Quantity("Magnitude", unit="V")

    bop  = BOP2020M("Kepco Power Supply", "GPIB1::1::INSTR")
    lock = SR830("Lockin Amplifier", "GPIB1::9::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)

    def instruments_init(self):
        self.tc_delay = 9*self.lock.tc
        self.averages = 10
        self.bop.output = True

        def lockin_measure():
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.averages)] )

        self.set_field.set_method(self.mag.set_field)
        self.field.set_method(self.mag.get_field)
        self.voltage.set_method(lockin_measure)

    def run(self):
        """This is run for each step in a sweep."""
        for param in self._parameters:
            self._parameters[param].push()
        for quant in self._quantities:
            self._quantities[quant].measure()

    def instruments_shutdown(self):
        self.bop.current = 0.0

if __name__ == '__main__':

    proc = FieldTest()

    # Define a sweep over prarameters
    sw = Sweep(proc)
    values = np.append(np.arange(-1100, 1100.1, 25), np.arange(1100.0,-1100.1,-25)).tolist()
    sw.add_parameter(proc.set_field, values)

    # Define a writer
    sw.add_writer('data/FieldLoops.h5', 'SWS2129(2,0)G-(011,09)', 'MajorLoop', proc.field, proc.voltage)

    proc.instruments_init()
    for i in sw:
        logging.info("Field, Lockin Magnitude: {:f}, {:g}".format(proc.field.value, proc.voltage.value) )
    proc.instruments_shutdown()

    # proc.current.value = 0.0
    