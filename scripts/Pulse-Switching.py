from __future__ import print_function, division
import time
import logging
# logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

import numpy as np
import pandas as pd

from instruments.kepco import BOP2020M
from instruments.magnet import Electromagnet
from instruments.hall_probe import HallProbe
from instruments.stanford import SR830
from instruments.picosecond import Picosecond10070A
from sweep import Sweep
from procedure import FloatParameter, IntParameter, Quantity, Procedure

class Switching(Procedure):
    pulse_voltage  = FloatParameter("Pulse Amplitude", unit="V")
    pulse_duration = FloatParameter("Pulse Duration", unit="s")
    attempt_number = IntParameter("Switching Attempt Index")

    field         = Quantity("Field", unit="G")
    initial_state = Quantity("Initial Voltage", unit="V")
    final_state   = Quantity("Final Voltage", unit="V")

    bop  = BOP2020M("Kepco Power Supply", "GPIB1::1::INSTR")
    lock = SR830("Lockin Amplifier", "GPIB1::9::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)
    pspl = Picosecond10070A("Pulse Generator", "GPIB1::24::INSTR")

    def instruments_init(self):
        self.tc_delay = 9*self.lock.tc
        self.averages = 2
        self.pspl.output = True

        def lockin_measure_initial():
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.averages)] )

        def lockin_measure_final():
            self.pspl.trigger()
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.averages)] )

        self.field.assign_method(self.mag.set_field)
        self.initial_state.assign_method(lockin_measure_initial)
        self.final_state.assign_method(lockin_measure_final)

        self.pulse_voltage.assign_method(self.pspl.set_amplitude)
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.attempt_number.assign_method(lambda x: 42)

    def run(self):
        """This is run for each step in a sweep."""
        for param in self._parameters:
            self._parameters[param].push()

        self.initial_state.measure()
        self.final_state.measure()

    def instruments_shutdown(self):
        self.bop.current = 0.0
        self.pspl.output = False

if __name__ == '__main__':

    proc = Switching()
    proc.field.value = -405
    proc.pulse_duration.value = 0.2e-9

    # Define a sweep over prarameters
    sw = Sweep(proc)
    sw.add_parameter(proc.pulse_voltage, -7.5*np.power(10,-np.arange(4,3,-1)/20.0))
    # sw.add_parameter(proc.pulse_duration, np.arange(0.2e-9, 0.31e-9, 0.1e-9))
    sw.add_parameter(proc.attempt_number, np.arange(0,50))

    # Define a writer
    sw.add_writer('data/SwitchingAttempts.h5', 'SWS2129(2,0)G-(011,09)', 'SwitchingTest', proc.initial_state, proc.final_state)
    sw.add_plotter('Intial Voltage vs. Attempt number', proc.attempt_number, proc.initial_state, color="firebrick", line_width=2)
    sw.add_plotter('Final Voltage vs. Attempt number', proc.attempt_number, proc.final_state, color="navy", line_width=2)
    sw.add_plotter('Pulse Voltage', proc.pulse_voltage, proc.attempt_number, color="green", line_width=2)

    proc.instruments_init()
    sw.run()
    proc.instruments_shutdown()
    