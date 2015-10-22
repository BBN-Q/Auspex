from __future__ import print_function, division
import time
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.WARNING)

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

    instrument_settings = {"lock":{"amp":0.1, "freq":167}}

    attempt_number             = IntParameter("Switching Attempt Index")
    field_setpoint             = FloatParameter("Field Setpoint", unit="G")
    middle_voltage             = FloatParameter("Middle Voltage Value", unit="V")
    high_probability_duration  = FloatParameter("High Probability Duration", unit="s")
    initialization_probability = FloatParameter("Initialization Probability", default=1.0)
    pulse_voltage              = FloatParameter("Pulse Amplitude", unit="V")
    pulse_duration             = FloatParameter("Pulse Duration", unit="s")

    field         = Quantity("Field", unit="G")
    initial_state = Quantity("Initial Voltage", unit="V")
    final_state   = Quantity("Final Voltage", unit="V")

    bop  = BOP2020M("Kepco Power Supply", "GPIB1::1::INSTR")
    lock = SR830("Lockin Amplifier", "GPIB1::9::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)
    pspl = Picosecond10070A("Pulse Generator", "GPIB1::24::INSTR")

    def instruments_init(self):
        self.tc_delay = self.lock.measure_delay()
        self.averages = 2
        self.pspl.output = True

        def lockin_measure_initial():
            if np.random.random() <= self.initialization_probability:
                self.pspl.duration = self.high_probability_duration.value
                time.sleep(0.09) # Required to let the duration settle
                self.pspl.trigger()
                self.pspl.duration = self.pulse_duration.value
                time.sleep(0.09) # Required to let the duration settle
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.averages)] )

        def lockin_measure_final():
            self.pspl.trigger()
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.averages)] )

        def find_reset_duration():
            # Search over pulse durations for highest probability reversal
            logging.warning("Finding optimal duration for this pulse amplitude...")

            num_pulses = 40
            durs = np.linspace(0.15e-9, 1.0e-9, 10)
            probs = []
            
            for dur in durs:
                self.pspl.duration = dur
                ivs = []
                fvs = []
                for i in range(num_pulses):
                    time.sleep(self.tc_delay)
                    v_initial = np.mean( [self.lock.r for i in range(self.averages)] )
                    ivs.append(v_initial)
                    self.pspl.trigger()
                    time.sleep(self.tc_delay)
                    v_final = np.mean( [self.lock.r for i in range(self.averages)] )
                    fvs.append(v_final)
                ivs = np.array(ivs)
                fvs = np.array(fvs)
                initial_states = (ivs-self.middle_voltage.value) > 0.0
                final_states = (fvs-self.middle_voltage.value) > 0.0
                switched = np.logical_xor(initial_states, final_states)
                num_attempts_APtoP = np.sum(initial_states > 0)
                num_attempts_PtoAP = np.sum(initial_states == 0)
                switched_APtoP = np.sum( np.logical_and(switched, initial_states) )
                switched_PtoAP = np.sum( np.logical_and(switched, np.logical_not(initial_states)) )
                if num_attempts_APtoP == 0:
                    prob_APtoP = 0.0
                else:
                    prob_APtoP = switched_APtoP/num_attempts_APtoP
                if num_attempts_PtoAP == 0:
                    prob_PtoAP = 0.0
                else:
                    prob_PtoAP = switched_PtoAP/num_attempts_PtoAP
                probs.append(0.5*(prob_PtoAP + prob_APtoP))
                logging.warning("Found probability {:f} for duration {:g}".format(probs[-1], dur))

            # Set the reset duration to the best available average probability
            self.high_probability_duration.value = durs[np.argmax(probs)]
            self.initialization_probability = 0.5/np.argmax(probs)
            logging.warning("Selected duration {:g}".format(self.high_probability_duration.value))

            # Reset the PSPL to the original duration
            self.pspl.duration = self.pulse_duration.value

        self.field_setpoint.assign_method(lambda x: setattr(self.mag, 'field', x))
        self.field.assign_method(lambda : getattr(self.mag, 'field'))
        self.high_probability_duration.assign_method(lambda x: 42)
        self.initial_state.assign_method(lockin_measure_initial)
        self.final_state.assign_method(lockin_measure_final)

        self.pulse_voltage.assign_method(self.pspl.set_amplitude)
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.attempt_number.assign_method(lambda x: 42)

        self.pulse_voltage.add_post_push_routine(find_reset_duration)

        for param in self._parameters:
            self._parameters[param].push()

    def run(self):
        """This is run for each step in a sweep."""
        # for param in self._parameters:
        #     self._parameters[param].push()

        self.initial_state.measure()
        self.final_state.measure()

    def instruments_shutdown(self):
        self.bop.current = 0.0
        self.pspl.output = False

if __name__ == '__main__':

    proc = Switching()
    proc.field_setpoint.value = -364
    proc.pulse_voltage.value = -7.5*np.power(10,-13.0/20.0)
    proc.high_probability_duration.value = 0.5e-9
    proc.middle_voltage.value = 0.001275

    # Define a sweep over prarameters
    sw = Sweep(proc)
    # sw.add_parameter(proc.pulse_voltage, 7.5*np.power(10,-np.arange(13,12,-1)/20.0))
    # sw.add_parameter(proc.field_setpoint, [-550,-500,-450,-400,-350,-300,-250])
    sw.add_parameter(proc.pulse_duration, np.arange(0.10e-9, 1.41e-9, 0.2e-9))
    sw.add_parameter(proc.attempt_number, np.arange(0,100))

    # Define a writer
    sw.add_writer('data/SwitchingJunk.h5', 'SWS2129(2,0)G-(009,05)', 'Switching-3.3K-13dB-Neg', proc.initial_state, proc.final_state)
    sw.add_plotter('Intial Voltage vs. Attempt number', proc.attempt_number, proc.initial_state, color="firebrick", line_width=2)
    sw.add_plotter('Final Voltage vs. Attempt number', proc.attempt_number, proc.final_state, color="navy", line_width=2)
    sw.add_plotter('Pulse Voltage', proc.pulse_duration, proc.attempt_number, color="green", line_width=2)

    proc.instruments_init()
    sw.run()
    proc.instruments_shutdown()
    