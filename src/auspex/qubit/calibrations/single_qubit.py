__all__ = ["CavityTuneup", "QubitTuneup", "RabiAmpCalibration", "RamseyCalibration", "PiCalibration",
            "Pi2Calibration", "DRAGCalibration"]

try:
    from QGL import *
    from QGL import config as QGLconfig
    from QGL.BasicSequences.helpers import create_cal_seqs, delay_descriptor, cal_descriptor
except:
    print("Could not find QGL")


import auspex.config as config
from auspex.log import logger
from copy import copy, deepcopy
# from adapt.refine import refine_1D
import os
import uuid
import pandas as pd
import networkx as nx
import scipy as sp
import subprocess
import zmq
import json
import datetime
from copy import copy

import time
import bbndb
from auspex.filters import DataBuffer
from auspex.qubit.qubit_exp import QubitExperiment
from auspex.qubit import pipeline
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.analysis.fits import *
from auspex.analysis.CR_fits import *
from auspex.analysis.qubit_fits import *
from auspex.analysis.helpers import normalize_buffer_data
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
from itertools import product

from .calibrations import QubitCalibration
from .helpers import *

import bbndb

class CavityTuneup(QubitCalibration):
    def __init__(self, qubit, frequencies, averages=750, **kwargs):
        self.start_frequencies = frequencies
        kwargs['averages'] = averages
        super(CavityTuneup, self).__init__(qubit, **kwargs)
        self.cw_mode = True

    def sequence(self):
        return [[Id(self.qubit), MEAS(self.qubit)]]

    def exp_config(self, exp):
        exp.add_qubit_sweep(self.qubit, "measure", "frequency", self.new_frequencies)
        self.quad_fun = lambda x: x

    def _calibrate(self):
        # all_data = np.empty(dtype=np.complex128)
        self.new_frequencies = self.start_frequencies
        self.frequencies = np.empty(0, dtype=np.complex128)
        self.group_delays = np.empty(0, dtype=np.complex128)
        self.datas = np.empty(0, dtype=np.complex128)
        # orig_avg = self.kwargs['averages']
        # Adaptive refinement to find cavity feature
        # for i in range(self.iterations + 1):
        self.data, _      = self.run_sweeps()
        self.datas        = np.append(self.datas, self.data)
        self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])

        ord = np.argsort(self.frequencies)
        self.datas = self.datas[ord]
        self.frequencies = self.frequencies[ord]

        self.phases = np.unwrap(np.angle(self.datas))
        self.group_delays = -np.diff(self.phases)/np.diff(self.frequencies)
        phase_poly = np.poly1d(np.polyfit(self.frequencies, self.phases, 6))
        # group_delay_poly = phase_poly.deriv()
        # fine_freqs = np.linspace(self.frequencies[0], self.frequencies[-1], self.iterations*len(self.frequencies))
        subtracted = self.phases - phase_poly(self.frequencies)
        group_delay = np.diff(subtracted)/np.diff(self.frequencies)

        # ordering = np.argsort(self.frequencies[:-1])
        self.plot1["Phase"] = (self.frequencies, self.phases)
        self.plot1["Phase Fit"] = (self.frequencies,phase_poly(self.frequencies))
        self.plot1B["Group Delay"] = (self.frequencies[:-1],group_delay)
        self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))

        guess = np.abs(self.frequencies[np.argmax(np.abs(group_delay))])
        self.new_frequencies = np.arange(guess-15e6, guess+15e6, 1e6)
        self.frequencies = np.empty(0, dtype=np.complex128)
        self.group_delays = np.empty(0, dtype=np.complex128)
        self.datas = np.empty(0, dtype=np.complex128)

        self.data, _      = self.run_sweeps()
        self.datas        = np.append(self.datas, self.data)
        self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])

        ord = np.argsort(self.frequencies)
        self.datas = self.datas[ord]
        self.frequencies = self.frequencies[ord]

        self.phases = np.unwrap(np.angle(self.datas))
        self.group_delays = -np.diff(self.phases)/np.diff(self.frequencies)
        phase_poly = np.poly1d(np.polyfit(self.frequencies, self.phases, 6))
        # group_delay_poly = phase_poly.deriv()
        # fine_freqs = np.linspace(self.frequencies[0], self.frequencies[-1], self.iterations*len(self.frequencies))
        subtracted = self.phases - phase_poly(self.frequencies)
        group_delay = np.diff(subtracted)/np.diff(self.frequencies)

        # ordering = np.argsort(self.frequencies[:-1])
        self.plot1["Phase"] = (self.frequencies, self.phases)
        self.plot1["Phase Fit"] = (self.frequencies,phase_poly(self.frequencies))
        self.plot1B["Group Delay"] = (self.frequencies[:-1],group_delay)
        self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))

        guess = np.abs(self.frequencies[np.argmax(np.abs(group_delay))])
        self.new_frequencies = np.arange(guess-4e6, guess+4e6, 0.2e6)
        self.frequencies = np.empty(0, dtype=np.complex128)
        self.group_delays = np.empty(0, dtype=np.complex128)
        self.datas = np.empty(0, dtype=np.complex128)

        self.data, _      = self.run_sweeps()
        self.datas        = np.append(self.datas, self.data)
        self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])

        ord = np.argsort(self.frequencies)
        self.datas = self.datas[ord]
        self.frequencies = self.frequencies[ord]

        self.phases = np.unwrap(np.angle(self.datas))
        self.group_delays = -np.diff(self.phases)/np.diff(self.frequencies)
        phase_poly = np.poly1d(np.polyfit(self.frequencies, self.phases, 6))
        # group_delay_poly = phase_poly.deriv()
        # fine_freqs = np.linspace(self.frequencies[0], self.frequencies[-1], self.iterations*len(self.frequencies))
        subtracted = self.phases - phase_poly(self.frequencies)
        group_delay = np.diff(subtracted)/np.diff(self.frequencies)

        # ordering = np.argsort(self.frequencies[:-1])
        self.plot1["Phase"] = (self.frequencies, self.phases)
        self.plot1["Phase Fit"] = (self.frequencies,phase_poly(self.frequencies))
        self.plot1B["Group Delay"] = (self.frequencies[:-1],group_delay)

        self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))

        shifted_cav = np.real(self.datas) - np.mean(np.real(self.datas))
        guess = np.abs(self.frequencies[np.argmax(np.abs(shifted_cav))])
            # self.kwargs['averages'] = 2000

            # import pdb; pdb.set_trace()
            #
            # self.new_frequencies = refine_1D(self.frequencies, subtracted, all_points=False,
            #                             criterion="difference", threshold = "one_sigma")
            # logger.info(f"new_frequencies {self.new_frequencies}")

        # n, bins = sp.histogram(np.abs(self.frequencies), bins="auto")
        # f_start = bins[np.argmax(n)]
        # f_stop  = bins[np.argmax(n)+1]
        # logger.info(f"Looking in bin from {f_start} to {f_stop}")

        # # self.kwargs['averages'] = orig_avg
        # self.new_frequencies = np.arange(f_start, f_stop, 2e6)
        # self.frequencies = np.empty(0, dtype=np.complex128)
        # self.group_delays = np.empty(0, dtype=np.complex128)
        # self.datas = np.empty(0, dtype=np.complex128)
        #
        # for i in range(self.iterations + 3):
        #     self.data, _      = self.run_sweeps()
        #     self.datas        = np.append(self.datas, self.data)
        #     self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])
        #
        #     ord = np.argsort(self.frequencies)
        #     self.datas = self.datas[ord]
        #     self.frequencies = self.frequencies[ord]
        #
        #     self.group_delays = -np.diff(np.unwrap(np.angle(self.datas)))/np.diff(self.frequencies)
        #     # self.group_delays = group_del
        #
        #     # ordering = np.argsort(self.frequencies[:-1])
        #     self.plot3["Group Delay"] = (self.frequencies[1:],self.group_delays)
        #     # self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))
        #     # self.kwargs['averages'] = 2000
        #
        #     self.new_frequencies = refine_1D(self.frequencies[:-1], self.group_delays, all_points=False,
        #                                 criterion="integral", threshold = "one_sigma")
        #     logger.info(f"new_frequencies {self.new_frequencies}")
        # #

        # # self.data, _ = self.run_sweeps()
        # # group_delay = -np.diff(np.unwrap(np.angle(self.data)))/np.diff(self.new_frequencies)
        # # self.plot3["Group Delay"] = (self.new_frequencies[1:],group_delay)
        #
        # def lor_der(x, a, x0, width, offset):
        #     return offset-(x-x0)*a/((4.0*((x-x0)/width)**2 + a**2)**2)
        # f0 = np.abs(self.frequencies[np.argmax(np.abs(self.group_delays))])
        # p0 = [np.max(np.abs(self.group_delays))*1e-18, np.abs(f0), 200e6, np.abs(self.group_delays)[0]]
        # popt, pcov = curve_fit(lor_der, np.abs(self.frequencies[1:]), np.abs(self.group_delays), p0=p0)
        # self.plot3["Group Delay Fit"] = ( np.abs(self.frequencies[1:]),  lor_der( np.abs(self.frequencies[1:]), *popt))


    def init_plots(self):
        plot1 = ManualPlotter("Phase", x_label='Frequency (GHz)', y_label='Group Delay')
        plot1.add_data_trace("Phase", {'color': 'C1'})
        plot1.add_fit_trace("Phase Fit", {'color': 'C2'})

        plot1B = ManualPlotter("Group Delay", x_label='Frequency (GHz)', y_label='Group Delay')
        plot1B.add_data_trace("Group Delay", {'color': 'C1'})
        # plot1B.add_fit_trace("Phase Fit", {'color': 'C2'})

        plot2 = ManualPlotter("Amplitude", x_label='Frequency (GHz)', y_label='Amplitude (Arb. Units)')
        plot2.add_data_trace("Amplitude", {'color': 'C2'})

        # plot3 = ManualPlotter("First refined sweep", x_label='Frequency (GHz)', y_label='Group Delay')
        # plot3.add_data_trace("Group Delay", {'color': 'C3'})
        # plot3.add_fit_trace("Group Delay Fit", {'color': 'C4'})
        self.plot1 = plot1
        self.plot1B = plot1B
        self.plot2 = plot2
        # self.plot3 = plot3
        return [plot1, plot1B, plot2] #, plot3]

class QubitTuneup(QubitCalibration):
    def __init__(self, qubit, f_start=5e9, f_stop=6e9, coarse_step=0.1e9, fine_step=1.0e6, averages=500, amp=1.0, **kwargs):
        self.coarse_frequencies = np.arange(f_start, f_stop, coarse_step) - 10.0e6 # Don't stray too close to the carrier tone
        self.fine_frequencies   = np.arange(10.0e6, coarse_step+10.0e6, fine_step)
        self.f_start = f_start
        self.f_stop = f_stop
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.amp = amp
        kwargs['averages'] = averages
        super(QubitTuneup, self).__init__(qubit, **kwargs)

    def sequence(self):
        return [[X(self.qubit, frequency=f, amp=self.amp), MEAS(self.qubit)] for f in self.fine_frequencies]

    def exp_config(self, exp):
        exp.add_qubit_sweep(self.qubit, "control", "frequency", self.coarse_frequencies)
        self.quad_fun = lambda x: x

    def _calibrate(self):
        self.data, _ = self.run_sweeps()
        freqs = np.arange(self.f_start, self.f_stop, self.fine_step)
        self.plot["Data"] = (freqs, self.data)

    def init_plots(self):
        plot = ManualPlotter("Qubit Search", x_label='Frequency (Hz)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data", {'color': 'C1'})
        plot.add_fit_trace("Fit", {'color': 'C1'})
        self.plot = plot
        return [plot]

class RabiAmpCalibration(QubitCalibration):

    amp2offset = 0.5

    def __init__(self, qubit, num_steps=40, **kwargs):
        if num_steps % 2 != 0:
            raise ValueError("Number of steps for RabiAmp calibration must be even!")
        #for now, only do one qubit at a time
        self.num_steps = num_steps
        self.amps = np.hstack((np.arange(-1, 0, 2./num_steps),
                               np.arange(2./num_steps, 1+2./num_steps, 2./num_steps)))
        super(RabiAmpCalibration, self).__init__(qubit, **kwargs)
        self.filename = 'Rabi/Rabi'

    def sequence(self):
        return ([[Xtheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps] +
                [[Ytheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps])

    def _calibrate(self):
        data, _ = self.run_sweeps()
        N = len(data)
        I_fit = RabiAmpFit(self.amps, data[N//2:])
        Q_fit = RabiAmpFit(self.amps, data[:N//2])
        #Arbitary extra division by two so that it doesn't push the offset too far.
        self.pi_amp = I_fit.pi_amp
        self.pi2_amp = I_fit.pi_amp/2.0
        self.i_offset = I_fit.fit_params["phi"]*self.amp2offset
        self.q_offset = Q_fit.fit_params["phi"]*self.amp2offset
        logger.info("Found X180 amplitude: {}".format(self.pi_amp))
        logger.info("Shifting I offset by: {}".format(self.i_offset))
        logger.info("Shifting Q offset by: {}".format(self.q_offset))
        finer_amps = np.linspace(np.min(self.amps), np.max(self.amps), 4*len(self.amps))
        self.plot["I Data"] = (self.amps, data[:N//2])
        self.plot["Q Data"] = (self.amps, data[N//2:])
        self.plot["I Fit"] = (finer_amps, I_fit.model(finer_amps))
        self.plot["Q Fit"] = (finer_amps, Q_fit.model(finer_amps))

        if self.pi_amp <= 1.0 and self.pi2_amp <= 1.0:
            self.succeeded = True

    def init_plots(self):
        plot = ManualPlotter("Rabi Amplitude Cal", x_label="I/Q Amplitude", y_label="{} (Arb. Units)".format(self.quad))
        plot.add_data_trace("I Data", {'color': 'C1'})
        plot.add_data_trace("Q Data", {'color': 'C2'})
        plot.add_fit_trace("I Fit", {'color': 'C1'})
        plot.add_fit_trace("Q Fit", {'color': 'C2'})
        self.plot = plot
        return [plot]

    def update_settings(self):
        s = round(self.pi_amp, 5)
        self.qubit.pulse_params['pi2Amp'] = round(self.pi2_amp, 5)
        self.qubit.pulse_params['piAmp'] = round(self.pi_amp, 5)
        awg_chan   = self.qubit.phys_chan
        amp_factor = self.qubit.phys_chan.amp_factor
        awg_chan.I_channel_offset += round(amp_factor*self.amp2offset*self.i_offset, 5)
        awg_chan.Q_channel_offset += round(amp_factor*self.amp2offset*self.i_offset, 5)

        if self.sample:
            c1 = bbndb.calibration.Calibration(value=self.pi2_amp, sample=self.sample, name="Pi2Amp", category="Rabi")
            c2 = bbndb.calibration.Calibration(value=self.pi_amp, sample=self.sample, name="PiAmp", category="Rabi")
            c1.date = c2.date = datetime.datetime.now()
            bbndb.get_cl_session().add_all([c1, c2])
            bbndb.get_cl_session().commit()

class RamseyCalibration(QubitCalibration):
    def __init__(self, qubit, delays=np.linspace(0.0, 20.0, 41)*1e-6,
                two_freqs=False, added_detuning=150e3, set_source=True, AIC=True, **kwargs):
        self.delays         = delays
        self.two_freqs      = two_freqs
        self.added_detuning = added_detuning
        self.set_source     = set_source
        self.AIC            = AIC #Akaike information criterion for model choice

        super(RamseyCalibration, self).__init__(qubit, **kwargs)
        self.filename = 'Ramsey/Ramsey'

    def descriptor(self):
        return [delay_descriptor(self.delays)]

    def sequence(self):
        return [[X90(self.qubit), Id(self.qubit, delay), X90(self.qubit), MEAS(self.qubit)] for delay in self.delays]

    def init_plots(self):
        plot = ManualPlotter("Ramsey Fits", x_label='Time (us)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data 1", {'color': 'black'})
        plot.add_fit_trace("Fit 1", {'color': 'red'})
        plot.add_data_trace("Data 2", {'color': 'green'})
        plot.add_fit_trace("Fit 2", {'color': 'blue'})
        self.plot = plot
        return [plot]

    def exp_config(self, exp):
        rcvr = self.qubit.measure_chan.receiver_chan.receiver
        if self.first_ramsey:
            if self.set_source:
                self.source_proxy = self.qubit.phys_chan.generator # DB object
                self.qubit_source = exp._instruments[self.source_proxy.label] # auspex instrument
                self.orig_freq    = self.source_proxy.frequency
                self.source_proxy.frequency = round(self.orig_freq + self.added_detuning, 10)
                self.qubit_source.frequency = self.source_proxy.frequency
            else:
                self.orig_freq = self.qubit.frequency

    def _calibrate(self):
        self.first_ramsey = True

        if not self.set_source:
            self.qubit.frequency += float(self.added_detuning)
        data, _ = self.run_sweeps()
        try:
            ramsey_fit = RamseyFit(self.delays, data, two_freqs=self.two_freqs, AIC=self.AIC)
            fit_freqs = ramsey_fit.fit_params["f"]
        except Exception as e:
            raise Exception(f"Exception {e} while fitting in {self}")

        # Plot the results
        self.plot["Data 1"] = (self.delays, data)
        finer_delays = np.linspace(np.min(self.delays), np.max(self.delays), 4*len(self.delays))
        self.plot["Fit 1"] = (finer_delays, ramsey_fit.model(finer_delays))

        #TODO: set conditions for success
        fit_freq_A = np.mean(fit_freqs) #the fit result can be one or two frequencies
        if self.set_source:
            self.source_proxy.frequency = round(self.orig_freq + self.added_detuning + fit_freq_A/2, 10)
            self.qubit_source.frequency = self.source_proxy.frequency
        else:
            self.qubit.frequency += float(fit_freq_A/2)

        self.first_ramsey = False

        # if self.plot:
        #     [self.add_manual_plotter(p) for p in self.plot] if isinstance(self.plot, list) else self.add_manual_plotter(self.plot)
        # self.start_manual_plotters()
        data, _ = self.run_sweeps()

        try:
            ramsey_fit = RamseyFit(self.delays, data, two_freqs=self.two_freqs, AIC=self.AIC)
            fit_freqs = ramsey_fit.fit_params["f"]
        except Exception as e:
            raise Exception(f"Exception {e} while fitting in {self}")

        # Plot the results
        self.plot["Data 2"] = (self.delays, data)
        self.plot["Fit 2"]  = (finer_delays, ramsey_fit.model(finer_delays))

        fit_freq_B = np.mean(fit_freqs)
        if fit_freq_B < fit_freq_A:
            self.fit_freq = round(self.orig_freq + self.added_detuning + 0.5*(fit_freq_A + 0.5*fit_freq_A + fit_freq_B), 10)
        else:
            self.fit_freq = round(self.orig_freq + self.added_detuning - 0.5*(fit_freq_A - 0.5*fit_freq_A + fit_freq_B), 10)
        logger.info(f"Found qubit Frequency {self.fit_freq}") #TODO: print actual qubit frequency, instead of the fit
        self.succeeded = True #TODO: add bounds

    def update_settings(self):
        if self.set_source:
            self.source_proxy.frequency = float(round(self.fit_freq))
            self.qubit_source.frequency = self.source_proxy.frequency
        else:
            self.qubit.frequency += float(round(self.fit_freq - self.orig_freq))
        # update edges where this is the target qubit
        for edge in self.qubit.edge_target:
            edge_source = edge.phys_chan.generator
            edge.frequency = self.source_proxy.frequency + self.qubit_source.frequency - edge_source.frequency
        #         # TODO: fix this for db backend

        # qubit_set_freq = self.saved_settings['instruments'][qubit_source]['frequency'] + self.saved_settings['qubits'][self.qubit.label]['control']['frequency']
        # logger.info("Qubit set frequency = {} GHz".format(round(float(qubit_set_freq/1e9),5)))
        # return ('frequency', qubit_set_freq)

class PhaseEstimation(QubitCalibration):

    amp2offset = 0.5

    def __init__(self, qubit, num_pulses= 1, amplitude= 0.1, direction = 'X',
                    target=np.pi/2, epsilon=1e-2, max_iter=5, **kwargs):
        #for now, only do one qubit at a time
        self.num_pulses = num_pulses
        self.amplitude = amplitude
        self.direction = direction

        self.target = target
        self.epsilon = epsilon
        self.max_iter = max_iter

        super(PhaseEstimation, self).__init__(qubit, **kwargs)

        self.filename = 'PhaseCal/PhaseCal'

    def sequence(self):
        # Determine whether it is a single- or a two-qubit pulse calibration
        if isinstance(self.qubit, list):
            qubit = self.qubit[1]
            cal_pulse = [ZX90_CR(*self.qubit, amp=self.amplitude)]
        else:
            qubit = self.qubit
            cal_pulse = [Xtheta(self.qubit, amp=self.amplitude)]

        # Exponentially growing repetitions of the target pulse, e.g.
        # (1, 2, 4, 8, 16, 32, 64, 128, ...) x X90
        seqs = [cal_pulse*n for n in 2**np.arange(self.num_pulses+1)]
        # measure each along Z or Y
        seqs = [s + m for s in seqs for m in [ [MEAS(qubit)], [X90m(qubit), MEAS(qubit)] ]]
        # tack on calibrations to the beginning
        seqs = [[Id(qubit), MEAS(qubit)], [X(qubit), MEAS(qubit)]] + seqs
        # repeat each
        return [copy(s) for s in seqs for _ in range(2)]

    def _calibrate(self):

        ct = 0
        done = 0

        start_amp = self.amplitude

        phase_error = []

        while not done and ct < self.max_iter:
            ct += 1
            data, var = self.run_sweeps()
            phase, sigma = phase_estimation(data, var)
            self.amplitude, done, error = phase_to_amplitude(phase, sigma, self.amplitude,
                                                self.target, epsilon=self.epsilon)
            phase_error.append(error)

            self.data_plot['data'] = (np.array(range(1, len(data)+1)), data)
            self.plot["angle_estimate"] = (np.array(range(1, len(phase_error)+1)), np.array(phase_error))

        if done == -1:
            self.succeeded = False
        elif done == 1:
            self.succeeded = True
        else:
            raise Exception()

    def init_plots(self):
        data_plot = ManualPlotter("Phase Cal", x_label="Sequence Number", y_label="{} (Arb. Units)".format(self.quad))
        data_plot.add_data_trace("data", {'color': 'C1'})
        plot = ManualPlotter("Phase Angle Error", x_label="Iteration", y_label="Angle (rad.)")
        plot.add_data_trace("angle_estimate", {'color': 'C1'})
        self.plot = plot
        self.data_plot = data_plot
        return [data_plot, plot]

    def update_settings(self):
        logger.warning("Nothing to update.")


class Pi2Calibration(PhaseEstimation):

    def __init__(self, qubit, num_pulses= 1, direction = 'X',
                    epsilon=1e-2, max_iter=5, **kwargs):
        super(Pi2Calibration, self).__init__(qubit, num_pulses=num_pulses,
                        amplitude=qubit.pulse_params['pi2Amp'], direction =direction,
                        target=np.pi/2, epsilon=epsilon, max_iter=max_iter, **kwargs)

    def update_settings(self):
        self.qubit.pulse_params['pi2Amp'] = round(self.amplitude, 5)

        if self.sample:
            c = bbndb.calibration.Calibration(value=self.amplitude, sample=self.sample, name="Pi2Amp", category="PhaseEstimation")
            c.date = datetime.datetime.now()
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

class PiCalibration(PhaseEstimation):

    def __init__(self, qubit, num_pulses= 1, direction = 'X',
                    epsilon=1e-2, max_iter=5, **kwargs):
        super(PiCalibration, self).__init__(qubit, num_pulses=num_pulses,
                        amplitude=qubit.pulse_params['piAmp'], direction =direction,
                        target=np.pi, epsilon=epsilon, max_iter=max_iter, **kwargs)

    def update_settings(self):
        self.qubit.pulse_params['piAmp'] = round(self.amplitude, 5)

        if self.sample:
            c = bbndb.calibration.Calibration(value=self.amplitude, sample=self.sample, name="PiAmp", category="PhaseEstimation")
            c.date = datetime.datetime.now()
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

class DRAGCalibration(QubitCalibration):
    def __init__(self, qubit, deltas = np.linspace(-1,1,21), num_pulses = np.arange(8, 48, 4), **kwargs):
        self.filename = 'DRAG/DRAG'
        self.deltas = deltas
        self.num_pulses = num_pulses
        super(DRAGCalibration, self).__init__(qubit, **kwargs)

    def sequence(self):
        seqs = []
        for n in self.num_pulses:
            seqs += [[X90(self.qubit, drag_scaling = d), X90m(self.qubit, drag_scaling = d)]*n + [X90(self.qubit, drag_scaling = d), MEAS(self.qubit)] for d in self.deltas]
        seqs += create_cal_seqs((self.qubit,),2)
        return seqs

    def init_plots(self):
        plot = ManualPlotter("DRAG Cal", x_label=['DRAG parameter', 'Number of pulses'], y_label=['Amplitude (Arb. Units)', 'Fit DRAG parameter'], numplots = 2)
        cmap = cm.viridis(np.linspace(0, 1, len(self.num_pulses)))
        for n in range(len(self.num_pulses)):
            plot.add_data_trace('Data_{}'.format(n), {'color': list(cmap[n]), 'linestyle': 'None'})
            plot.add_fit_trace('Fit_{}'.format(n), {'color': list(cmap[n])})
        plot.add_data_trace('Data_opt', subplot_num = 1) #TODO: error bars
        self.plot = plot
        return [plot]

    def exp_config(self, exp):
        rcvr = self.qubit.measure_chan.receiver_chan.receiver
        exp._instruments[rcvr.label].exp_step = self.step #where from?

    def _calibrate(self):
        # run twice for different DRAG parameter ranges
        for k in range(2):
            self.step = k
            data, _ = self.run_sweeps()
            finer_deltas = np.linspace(np.min(self.deltas), np.max(self.deltas), 4*len(self.deltas))
            #normalize data with cals
            data = quick_norm_data(data)
            try:
                opt_drag, error_drag, popt_mat = fit_drag(data, self.deltas, self.num_pulses)
                if k==1:
                    self.succeeded = True
            except Exception as e:
                raise Exception(f"Exception {e} while fitting in {self}")

            norm_data = data.reshape((len(self.num_pulses), len(self.deltas)))
            for n in range(len(self.num_pulses)):
                self.plot['Data_{}'.format(n)] = (self.deltas, norm_data[n, :])
                finer_deltas = np.linspace(np.min(self.deltas), np.max(self.deltas), 4*len(self.deltas))
                self.plot['Fit_{}'.format(n)] = (finer_deltas, quadf(finer_deltas, *popt_mat[:, n]))
            self.plot["Data_opt"] = (self.num_pulses, opt_drag) #TODO: add error bars

            if k==0:
                #generate sequence with new pulses and drag parameters
                new_drag_step = 0.25*(max(self.deltas) - min(self.deltas))
                self.deltas = np.linspace(opt_drag[-1] - new_drag_step, opt_drag[-1] + new_drag_step, len(self.deltas))
                new_pulse_step = int(np.floor(2*(max(self.num_pulses)-min(self.num_pulses))/len(self.num_pulses)))
                self.num_pulses = np.arange(max(self.num_pulses) - new_pulse_step, max(self.num_pulses) + new_pulse_step*(len(self.num_pulses)-1), new_pulse_step)

            if not self.leave_plots_open:
                self.plot.set_quit()
        self.opt_drag = round(float(opt_drag[-1]), 5)

    def update_settings(self):
        logger.info(f'{self.qubit.label} DRAG parameter set to {self.opt_drag}')
        self.qubit.pulse_params['drag_scaling'] = self.opt_drag

        if self.sample:
            c = bbndb.calibration.Calibration(value=self.opt_drag, sample=self.sample, name="drag_scaling")
            c.date = datetime.datetime.now()
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()
