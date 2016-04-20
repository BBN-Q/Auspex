from __future__ import print_function, division
import time
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

import numpy as np

from pycontrol.instruments.yokogawa import YokogawaGS200
from pycontrol.sweep import Sweep
from pycontrol.procedure import FloatParameter, Quantity, Procedure

class RampCurrent(Procedure):
    current = FloatParameter("Current", unit="A")
    voltage = Quantity("Voltage", unit="V")

    yoko = YokogawaGS200("Current source", "GPIB0::30::INSTR")

    def init_instruments(self):
        self.yoko.function = "current"
        self.yoko.level = 0.0
        self.yoko.output = True

        self.current.assign_method(self.yoko.set_level)
        self.voltage.assign_method(self.yoko.get_sense_value)

        for param in self._parameters:
            self._parameters[param].push()

    def run(self):
        """This is run for each step in a sweep."""
        for param in self._parameters:
            self._parameters[param].push()
        time.sleep(0.5)
        for quant in self._quantities:
            self._quantities[quant].measure()

    def shutdown_instruments(self):
        self.yoko.level = 0.0
        self.yoko.output = False

if __name__ == '__main__':

    proc = RampCurrent()

    # Define a sweep over prarameters
    sw = Sweep(proc)

    sw.add_parameter(proc.current, np.linspace(0, 10e-6, 100))

    # Define a writer
    # sw.add_writer('data/C4R2-FieldSweep.h5', 'CSHE-C4R2', 'Neg-4K', proc.resistance)

    # Define a plotter
    sw.add_plotter("Voltage vs. Current", proc.current, proc.voltage, color="firebrick", line_width=2)

    sw.run()
