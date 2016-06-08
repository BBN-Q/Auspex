from __future__ import print_function, division
import time
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import beta
from scipy.interpolate import interp1d

from PyDAQmx import *

from pycontrol.instruments.kepco import BOP2020M
from pycontrol.instruments.magnet import Electromagnet
from pycontrol.instruments.hall_probe import HallProbe
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.keysight import *

from pycontrol.sweep import Sweep
from pycontrol.experiment import FloatParameter, IntParameter, Quantity, Procedure

class Trace(object):
    """Object for storing trace data"""
    def __init__(self, unit=None, length=None):
        super(Trace, self).__init__()
        self.length = length
        self.buffer = np.empty(shape=(0 if length is None else length))
        self.

    def register_output()


class SwitchingAWG(Procedure):
    """Simple square pulse switching, to and fro, using only AWG pulses."""

    # Parameters
    attempt_number = IntParameter("Switching attempts", default=1024, abstract=True)
    field_setpoint = FloatParameter("Field Setpoint", unit="G")
    pulse_voltage  = FloatParameter("Pulse Amplitude", unit="V")
    pulse_duration = FloatParameter("Pulse Duration", unit="s")

    # Instrument resources
    bop  = BOP2020M("GPIB0::1::INSTR")
    lock = SR865("USB0::0xB506::0x2000::002638::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)
    pspl = Picosecond10070A("GPIB0::24::INSTR")

    # Quantities
    field = Quantity("Field", unit="G")

    # Traces
    raw_trace = Trace("P")

    # Filters
    clusterer   = Clusterer(input=raw_trace)
    probability = CollapseProbability(input=clusterer)



    def init_instruments(self):

        # AWG Initialization
        arb.set_output(True, channel=1)
        arb.set_output(False, channel=2)
        arb.sample_freq = 12.0e9
        arb.waveform_output_mode = "WSPEED"

        arb.abort()
        arb.delete_all_waveforms()
        arb.reset_sequence_table()

        arb.set_output_route("DC", channel=1)
        arb.voltage_amplitude = 1.0

        arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
        arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

        arb.continuous_mode = False
        arb.gate_mode = False
        arb.sequence_mode = "SCENARIO"
        arb.scenario_advance_mode = "SINGLE"


class SwitchingPSPL(Procedure):

    # instrument_settings = {"lock":{"amp":0.1, "freq":167}}

    # These don't correspond to actual instrument parameters, and are thus "abstract"
    attempt_number             = IntParameter("Switching Attempt Index", abstract=True)
    middle_voltage             = FloatParameter("Middle Voltage Value", unit="V", abstract = True)
    high_probability_duration  = FloatParameter("High Probability Duration", unit="s", abstract=True)
    initialization_probability = FloatParameter("Initialization Probability", default=1.0, abstract=True)
    resistance_averages        = IntParameter("Resistance Averages", default=2, abstract=True)

    # These do correspond to instrument parameters
    field_setpoint             = FloatParameter("Field Setpoint", unit="G")
    pulse_voltage              = FloatParameter("Pulse Amplitude", unit="V")
    pulse_duration             = FloatParameter("Pulse Duration", unit="s")

    # Quantities to be measured
    field         = Quantity("Field", unit="G")
    initial_state = Quantity("Initial Voltage", unit="V")
    final_state   = Quantity("Final Voltage", unit="V")

    # Instrument resources
    bop  = BOP2020M("Kepco Power Supply", "GPIB1::1::INSTR")
    lock = SR830("Lockin Amplifier", "GPIB1::9::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)
    pspl = Picosecond10070A("Pulse Generator", "GPIB1::24::INSTR")

    def init_instruments(self):
        self.tc_delay = self.lock.measure_delay()
        self.pspl.output = True

        def lockin_measure_initial():
            if np.random.random() <= self.initialization_probability.value:
                self.pspl.duration = self.high_probability_duration.value
                time.sleep(0.09) # Required to let the duration settle
                self.pspl.trigger()
                self.pspl.duration = self.pulse_duration.value
                time.sleep(0.09) # Required to let the duration settle
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.resistance_averages.value)] )

        def lockin_measure_final():
            self.pspl.trigger()
            time.sleep(self.tc_delay)
            return np.mean( [self.lock.r for i in range(self.resistance_averages.value)] )

        # Associate quantities and parameters with their respective methods
        self.field_setpoint.assign_method(lambda x: setattr(self.mag, 'field', x))
        self.field.assign_method(lambda : getattr(self.mag, 'field'))
        self.initial_state.assign_method(lockin_measure_initial)
        self.final_state.assign_method(lockin_measure_final)
        self.pulse_voltage.assign_method(self.pspl.set_amplitude)
        self.pulse_duration.assign_method(self.pspl.set_duration)

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
                    v_initial = np.mean( [self.lock.r for i in range(self.resistance_averages.value)] )
                    ivs.append(v_initial)
                    self.pspl.trigger()
                    time.sleep(self.tc_delay)
                    v_final = np.mean( [self.lock.r for i in range(self.resistance_averages.value)] )
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
            self.initialization_probability.value = 0.5/np.max(probs)
            logging.warning("Best probability {:f}".format(self.initialization_probability.value))
            logging.warning("Selected duration {:g}".format(self.high_probability_duration.value))

            # Reset the PSPL to the original duration
            self.pspl.duration = self.pulse_duration.value

        self.pulse_voltage.add_post_push_hook(find_reset_duration)

        for param in self._parameters:
            logging.warning("Pushing parameter {:s}".format(self._parameters[param].name))
            self._parameters[param].push()

    def shutdown_instruments(self):
        self.bop.current = 0.0
        self.pspl.output = False

    def run(self):
        self.initial_state.measure()
        self.final_state.measure()


if __name__ == '__main__':

    proc = Switching()
    proc.field_setpoint.value = -364
    # proc.pulse_voltage.value = 7.5*np.power(10,-13.0/20.0)
    proc.high_probability_duration.value = 0.5e-9
    proc.middle_voltage.value = 0.001273

    # Define a sweep over prarameters
    sw = Sweep(proc)
    sw.add_parameter(proc.pulse_voltage, 7.5*np.power(10,-np.arange(15,11,-1)/20.0))
    # sw.add_parameter(proc.field_setpoint, [-550,-500,-450,-400,-350,-300,-250])
    sw.add_parameter(proc.pulse_duration, np.arange(0.10e-9, 1.21e-9, 0.025e-9))
    sw.add_parameter(proc.attempt_number, np.arange(0,200))

    # Define a writer
    sw.add_writer('data/Switching-COSTM-Seed.h5', 'SWS2129(2,0)G-(009,05)', 'Switching-3.3K-AmpSweep', proc.initial_state, proc.final_state)
    sw.add_plotter('Intial Voltage vs. Attempt number', proc.attempt_number, proc.initial_state, color="firebrick", line_width=2)
    sw.add_plotter('Final Voltage vs. Attempt number', proc.attempt_number, proc.final_state, color="navy", line_width=2)
    sw.add_plotter('Pulse Voltage', proc.pulse_duration, proc.attempt_number, color="green", line_width=2)

    sw.run()
