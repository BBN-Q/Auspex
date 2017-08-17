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
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

import numpy as np

from auspex.instruments.yokogawa import YokogawaGS200
from auspex.sweep import Sweep
from auspex.experiment import FloatParameter, Quantity, Procedure

class RampCurrent(Procedure):
    voltage = FloatParameter(unit="V")
    current = Quantity(unit="A")

    yoko = YokogawaGS200("GPIB0::30::INSTR")

    def init_instruments(self):
        self.yoko.function = "voltage"
        self.yoko.level = 0.0
        self.yoko.output = True

        self.voltage.assign_method(self.yoko.set_level)
        self.current.assign_method(self.yoko.get_sense_value)

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

    sw.add_parameter(proc.voltage, np.linspace(0, 20, 100))

    # Define a writer
    # sw.add_writer('data/C4R2-FieldSweep.h5', 'CSHE-C4R2', 'Neg-4K', proc.resistance)

    # Define a plotter
    sw.add_plotter("Voltage vs. Current", proc.current, proc.voltage, color="firebrick", line_width=2)

    sw.run()
