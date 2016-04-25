from __future__ import print_function, division
import time
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

import numpy as np
import scipy as sp
import pandas as pd

from instruments.kepco import BOP2020M
from instruments.stanford import SR865, SR830
from instruments.magnet import Electromagnet
from instruments.hall_probe import HallProbe
from sweep import Sweep
from procedure import FloatParameter, Quantity, Procedure


class CurrentLoop(Procedure):
    field        = FloatParameter(name="Magnetic Field Setpoint", unit="G")
    dc_bias      = FloatParameter(name="DC Bias", unit="V")

    actual_field = Quantity(name="Magnitude Field", unit="G")
    voltage      = Quantity(name="Magnitude", unit="V")

    bop       = BOP2020M("GPIB1::1::INSTR")
    lock      = SR830("GPIB1::9::INSTR")
    fast_lock = SR865("USB0::0xB506::0x2000::002638::INSTR")
    hp        = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag       = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)

    def instruments_init(self):
        self.tc_delay = 9*self.lock.tc
        self.averages = 10
        self.bop.output = True

        def lockin_measure():
            time.sleep(self.tc_delay)
            vals = []
            for i in range(self.averages):
                vals.append(self.fast_lock.r)
                time.sleep(0.03)
            return np.mean(vals)

        self.field.assign_method(self.mag.set_field)
        self.actual_field.assign_method(self.mag.get_field)
        self.dc_bias.assign_method(self.fast_lock.set_offset)
        self.voltage.assign_method(lockin_measure)

    def run(self):
        """This is run for each step in a sweep."""
        for param in self._parameters:
            self._parameters[param].push()
        for quant in self._quantities:
            self._quantities[quant].measure()

    def instruments_shutdown(self):
        self.bop.current = 0.0

if __name__ == '__main__':

    proc = CurrentLoop()
    proc.field.value = 390.0

    # Define a sweep over prarameters
    sw = Sweep(proc)
    values = np.append(np.arange(-3.0, 3.01, 0.1), np.arange(3.0,-3.01,-0.1)).tolist()
    sw.add_parameter(proc.dc_bias, values)

    # Define a writer
    sw.add_writer('data/CurrentLoops.h5', 'SWS2129(2,0)G-(011,09)', 'Loop', proc.voltage)
    # sw.add_writer('data/CurrentLoops.h5', 'Test', proc.voltage)

    proc.instruments_init()
    for i in sw:
        logging.info("Bias (V), Lockin Magnitude (V): {:f}, {:g}".format(proc.dc_bias.value, proc.voltage.value) )
    proc.instruments_shutdown()
