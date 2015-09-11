from __future__ import print_function, division
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd

from instrument import Instrument, Command
from sweep import Sweep
from procedure import FloatParameter, Quantity, Procedure

class SignalGenerator(Instrument):
    power = Command("power", get_string=":pow?", set_string=":pow %g dbm;")
    frequency = Command("frequency", get_string=":freq?", set_string=":freq %g Hz;")
    center_frequency = Command("center_frequency", get_string=":SOUR:FREQ:CENT?", set_string=":SOUR:FREQ:CENT %e HZ")
    start_frequency = Command("start_frequency", get_string=":SOUR:FREQ:STAR?", set_string=":SOUR:FREQ:STAR %e HZ")
    stop_frequency = Command("stop_frequency", get_string=":SOUR:FREQ:STOP?", set_string=":SOUR:FREQ:STOP %e HZ")
    start_power = Command("start_power", get_string=":SOUR:POW:STAR?", set_string=":SOUR:POW:STAR %e DBM")
    stop_power = Command("stop_power", get_string=":SOUR:POW:STOP?", set_string=":SOUR:POW:STOP %e DBM")
    dwell_time = Command("dwell_time", get_string=":SOUR:SWE:DWEL1?", set_string=":SOUR:SWE:DWEL1 %.3f")
    step_points = Command("step_points", get_string=":SOUR:SWE:POIN?")
    output = Command("output", get_string=":output?;", set_string=":output %s;", value_map={True: "on", False: "off"})

class Magnet(Instrument):
    field = Command("field", get_string=":field?", set_string=":field %g Oe;")

class Keithley(Instrument):
    resistance = Command("resistance", get_string=":res?", set_string=":res %g Oe;")
    testing = Command("testing", get_string=":test?", set_string=":test %g Oe;")

class TestProcedure(Procedure):

    # Parameters
    field       = FloatParameter("Field", unit="G")
    frequency   = FloatParameter("Frequency", unit="Hz")

    # Quantities
    resistance_trans  = Quantity("Transverse Resistance", unit="Ohm")
    resistance_long   = Quantity("Longitudinal Resistance", unit="Ohm")

    def __init__(self):
        super(TestProcedure,self).__init__()


if __name__ == '__main__':

    # Create instances of instruments
    sg = SignalGenerator('SG1', "GPIB0::4::INSTR")
    mag = Magnet("GMW", "GPIB0::5::INSTR")
    keith1 = Keithley("Keithley Transverse", "GPIB0::6::INSTR")
    keith2 = Keithley("Keithley Longitudinal", "GPIB0::7::INSTR")

    # Create an instance of the procedure
    proc = TestProcedure()
    
    # Define methods for setting 
    proc.field.set_method(mag.set_field)
    proc.frequency.set_method(sg.set_frequency)
    
    # Define methods for getting
    proc.resistance_trans.set_method(keith1.get_resistance)
    proc.resistance_long.set_method(keith2.get_resistance)

    # Define a sweep over prarameters
    sweep1 = Sweep(proc)
    sweep1.add_parameter(proc.frequency, 1e9, 5e9, interval=1e9)
    sweep1.add_parameter(proc.field, 0, 1000, interval=200)
    sweep1.generate_sweep()

    for i in sweep1:

        # This runs the procedure with the now updated parameters
        proc.run()

        # This returns the values obtained during the loop
        logging.info("Fake results: %s,\t%s" % (proc.resistance_trans, proc.resistance_long) )


