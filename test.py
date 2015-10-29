from __future__ import print_function, division
import logging
import time
logging.basicConfig(format='%(levelname)s: \t%(asctime)s: \t%(message)s', level=logging.WARNING)

import numpy as np
import scipy as sp
import pandas as pd

from instruments.instrument import Instrument, Command
from instruments.picosecond import Picosecond10070A
from sweep import Sweep
from procedure import FloatParameter, Quantity, Procedure

class Magnet(Instrument):
    field = Command("field", get_string=":field?", set_string=":field %g Oe;")

class Keithley(Instrument):
    resistance = Command("resistance", get_string=":res?", set_string=":res %g Oe;")
    testing = Command("testing", get_string=":test?", set_string=":test %g Oe;")

class TestProcedure(Procedure):

    # Create instances of instruments
    mag = Magnet("GMW", "FAKE::RESOURE::NAME")
    keith1 = Keithley("Keithley Transverse", "FAKE::RESOURE::NAME")
    keith2 = Keithley("Keithley Longitudinal", "FAKE::RESOURE::NAME")

    # Parameters
    field_x = FloatParameter("Field X", unit="G")
    field_y = FloatParameter("Field Y", unit="G")

    # Quantities
    resistance_trans = Quantity("Transverse Resistance", unit="Ohm")
    resistance_long = Quantity("Longitudinal Resistance", unit="Ohm")

    def init_instruments(self):
        self.field_x.assign_method(lambda x: time.sleep(0.1))
        self.field_y.assign_method(lambda x: time.sleep(0.05))
        self.resistance_trans.assign_method(lambda: self.field_x.value - self.field_y.value + 20*np.random.random())
        self.resistance_long.assign_method(lambda: self.field_x.value - self.field_y.value + 40*np.random.random())

    def run(self):
        for quant in self._quantities:
            self._quantities[quant].measure()


if __name__ == '__main__':

    # Create an instance of the procedure
    proc = TestProcedure()    
  
    # Define a sweep over prarameters
    sweep1 = Sweep(proc)
    sweep1.add_parameter(proc.field_y, np.arange(-100, 101, 10))
    sweep1.add_parameter(proc.field_x, np.arange(-100, 101, 10))
    
    p1 = sweep1.add_plotter("ResistanceL Vs Field", proc.field_x, proc.resistance_long, color="firebrick", line_width=2)
    p2 = sweep1.add_plotter("ResistanceT Vs Field", proc.field_x, proc.resistance_trans, color="navy", line_width=2)

    proc.field_y.add_post_push_hook(p1.clear)

    sweep1.run()

