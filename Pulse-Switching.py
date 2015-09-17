from __future__ import print_function, division
import time
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd

from instruments.kepco import BOP2020M
from instruments.stanford import SR830
from instruments.picosecond import Picosecond10070A
from sweep import Sweep
from procedure import FloatParameter, IntParameter, Quantity, Procedure

import ipdb

class Switching(Procedure):
    pulse_voltage = FloatParameter("Pulse Amplitude", unit="A")
    pulse_duration = FloatParameter("Pulse Duration", unit="s")
    attempt_number = IntParameter("Switching Attempt Index")
    supply_current = FloatParameter("Supply Current", unit="A")
    
    initial_state = Quantity("Initial Voltage State", unit="V")
    final_state = Quantity("Final Voltage State", unit="V")

    bop = BOP2020M("Kepco Power Supply", "GPIB1::1::INSTR")
    lock = SR830("Lockin Amplifier", "GPIB1::9::INSTR")
    pspl = Picosecond10070A("Pulse Generator", "GPIB1::24::INSTR")

    def instruments_init(self):
        self.tc_delay = 9*self.lock.tc
        self.averages = 5
        self.bop.output = True
        self.pspl.output = True

        def lockin_measure_initial():
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.averages)] )

        def lockin_measure_final():
            self.pspl.trigger()
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.averages)] )

        self.supply_current.set_method(self.bop.set_current)
        self.initial_state.set_method(lockin_measure_initial)
        self.final_state.set_method(lockin_measure_final)

        self.pulse_voltage.set_method(self.pspl.set_amplitude)
        self.pulse_duration.set_method(self.pspl.set_duration)
        self.attempt_number.set_method(lambda x: 42)

    def run(self):
        """This is run for each step in a sweep."""
        for param in self._parameters:
            self._parameters[param].push()

        self.initial_state.measure()
        self.final_state.measure()

    def instruments_shutdown(self):
        self.bop.current = 0.0
        self.bop.output = False
        self.pspl.output = False

if __name__ == '__main__':

    proc = Switching()
    proc.supply_current.value = 5.0

    # Define a sweep over prarameters
    sw = Sweep(proc)
    sw.add_parameter_hack(proc.pulse_voltage, 7.5*np.power(10,-np.arange(34,33,-1)/20.0))
    sw.add_parameter_hack(proc.pulse_duration, np.arange(0.1e-9, 0.51e-9, 0.1e-9))
    sw.add_parameter_hack(proc.attempt_number, np.arange(0,10))

    # Define a writer
    # sw.add_writer('SwitchingAttempts.h5', 'Testing-CheckBeforeSet-6', proc.initial_state, proc.final_state)

    proc.instruments_init()
    for i in sw:
        logging.info("Voltage, Duration, Initial V, Final V: {:g}, {:g}, {:g}, {:g}".format(proc.pulse_voltage.value, 
                proc.pulse_duration.value, proc.initial_state.value, proc.final_state.value) )
    
    proc.instruments_shutdown()
    