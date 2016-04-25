from __future__ import print_function, division
import logging
import time
from functools import partial
logging.basicConfig(format='%(levelname)s: \t%(asctime)s: \t%(message)s', level=logging.WARNING)

import numpy as np
import scipy as sp
import pandas as pd

from pycontrol.instruments.instrument import Instrument, StringCommand
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.sweep import Sweep
from pycontrol.procedure import FloatParameter, Quantity, Procedure

class Magnet(Instrument):
    field = StringCommand(get_string=":field?", set_string=":field %g Oe;")

class Keithley(Instrument):
    resistance = StringCommand(get_string=":res?", set_string=":res %g Oe;")
    testing = StringCommand(get_string=":test?", set_string=":test %g Oe;")

class TestProcedure(Procedure):

    # Create instances of instruments
    mag    = Magnet("FAKE::RESOURCE::NAME")
    keith1 = Keithley("FAKE::RESOURCE::NAME")
    keith2 = Keithley("FAKE::RESOURCE::NAME")

    # Parameters
    field_x = FloatParameter(name="Field X", unit="G")
    field_y = FloatParameter(name="Field Y", unit="G")

    # Quantities
    resistance_trans = Quantity(name="Transverse Resistance", unit="Ohm")
    resistance_long = Quantity(name="Longitudinal Resistance", unit="Ohm")

    def init_instruments(self):
        self.field_x.assign_method(lambda x: time.sleep(0.01))
        self.field_y.assign_method(lambda x: time.sleep(0.01))
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
    field_x = sweep1.add_parameter(proc.field_y, np.arange(-100, 101, 10))
    field_y = sweep1.add_parameter(proc.field_x, np.arange(-100, 101, 10))

    plot1 = sweep1.add_plotter("ResistanceL Vs Field", proc.field_x, proc.resistance_long, color="firebrick", line_width=2)
    plot2 = sweep1.add_multiplotter("Resistances Vs Field", [proc.field_x, proc.field_x], [proc.resistance_trans, proc.resistance_long],
                                    line_color=["firebrick","navy"], line_width=2)

    # Have to pass sweep parmaters here in order that the plotter knows the x,y grid
    plot3 = sweep1.add_plotter2d("A Whole New Dimension", field_x, field_y, proc.resistance_trans, palette="Spectral11")

    # Hooks for clearing the plots at the end of a sub-sweep
    proc.field_y.add_post_push_hook(plot1.clear)
    proc.field_y.add_post_push_hook(plot2.clear)

    # Make sure we update the data at the end of a trace
    proc.field_y.add_pre_push_hook(partial(plot1.update, force=True))
    proc.field_y.add_pre_push_hook(partial(plot2.update, force=True))

    sweep1.run()
