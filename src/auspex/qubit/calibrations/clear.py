__all__  = ["CLEARCalibration"]

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
from scipy.optimize import minimize

from .calibrations import QubitCalibration
from .helpers import *

import bbndb


class CLEARCalibration(QubitCalibration):
    '''Calibration of cavity reset pulse.

    Args:
        kappa: Cavity linewith (angular frequency: 1/s).
        chi: Half of the dispersive shift (anguler frequency: 1/s).
        t_empty: Time for active depletion (s).
        alpha: Scaling factor.
        T1factor: decay due to T1 between end of measurement and start of Ramsey.
        T2: Measured T2*
        nsteps: number of calibration steps
        ramsey_delays: List of times to use for Ramsey experiment.
        ramsey_freq: Ramsey offset frequency.
        meas_delay: Delay after end of measurement pulse
        preramsey_delay: Delay before start of Ramsey sequence.
        eps1: 1st CLEAR parameter. if set to `None` will use theory values as default for eps1 and eps2.
        eps2: 2nd CLEAR parameter.
        cal_steps: Steps over which to sweep calibration.
    '''

    def __init__(self, qubit, kappa = 2*np.pi*2e6, chi = -2*np.pi*1e6, t_empty = 400e-9,
                ramsey_delays=np.linspace(0.0, 50.0, 51)*1e-6, ramsey_freq = 100e3, meas_delay = 0,
                preramsey_delay=0, alpha = 1, T1factor = 1, T2 = 30e-6, nsteps = 11,
                eps1 = None, eps2 = None, cal_steps = (1,1,1), **kwargs):

        self.kappa = kappa
        self.chi = chi
        self.ramsey_delays = ramsey_delays
        self.ramsey_freq = ramsey_freq
        self.meas_delay = meas_delay
        self.preramsey_delay = preramsey_delay
        self.tau = t_empty/2.0
        self.alpha = alpha
        self.T1factor = T1factor
        self.T2 = T2
        self.nsteps = nsteps

        #use theory values as defaults
        if not eps1:
            self.eps1 = ((1 - 2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
                        /(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2)))
            self.eps2 = 1/(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
            logger.info(f' Using theoretical CLEAR amplitudes: {self.eps1} (eps1), {self.eps2} (eps2)')
        else:
            self.eps1 = eps1
            self.eps2 = eps2

        self.cal_steps = cal_steps

        self.seq_params = {}

        kwargs['disable_plotters'] = True
        super().__init__(qubit, **kwargs)
        self.filename = 'CLEAR/CLEAR'

    def descriptor(self):
        return [delay_descriptor(self.ramsey_delays), cal_descriptor(tuple(self.qubits), 2)]

    def sequence(self):
        if self.seq_params['state']:
            prep = X(self.qubit)
        else:
            prep = Id(self.qubit)

        amp1 = self.alpha * self.seq_params['eps1']
        amp2 = self.alpha * self.seq_params['eps2']


        clear_meas = MEASCLEAR(self.qubit, amp1=amp1, amp2=amp2, step_length=self.seq_params['tau'])
        seqs = [[prep, clear_meas, Id(self.qubit, self.preramsey_delay), X90(self.qubit), Id(self.qubit,d),
                    U90(self.qubit,phase = 2*pi*self.ramsey_freq*d), Id(self.qubit, self.meas_delay), MEAS(self.qubit)]
                        for d in self.ramsey_delays]

        seqs += create_cal_seqs((self.qubit,), 2, delay = self.meas_delay)

        return seqs

    def init_plots(self):
        plot_ramsey = ManualPlotter("CLEAR Ramsey", x_label='Time (us)', y_label='<Z>')
        plot_clear = ManualPlotter("CLEAR Calibration", x_label='epsilon 1', y_label='epsilon 2')
        plot_n = ManualPlotter("Residula Photons", x_label='iteration', y_label="Residual Photons")

        plot_ramsey.add_data_trace("Data - 0 State")
        plot_ramsey.add_fit_trace("Fit - 0 State")
        plot_ramsey.add_data_trace("Data - 1 State")
        plot_ramsey.add_fit_trace("Fit - 1 State")

        plot_clear.add_data_trace("epsilons")
        plot_n.add_data_trace("n0")
        plot_n.add_data_trace("n1")

        self.plot_ramsey = plot_ramsey
        self.plot_clear = plot_clear
        self.plot_n = plot_n

        return [plot_ramsey, plot_clear, plot_n]

    def exp_config(self, exp):
        pass #??
        # rcvr = self.qubit.measure_chan.receiver_chan.receiver
        # if self.first_ramsey:
        #     if self.set_source:
        #         self.source_proxy = self.qubit.phys_chan.generator # DB object
        #         self.qubit_source = exp._instruments[self.source_proxy.label] # auspex instrument
        #         self.orig_freq    = self.source_proxy.frequency
        #         self.source_proxy.frequency = round(self.orig_freq + self.added_detuning, 10)
        #         self.qubit_source.frequency = self.source_proxy.frequency
        #     else:
        #         self.orig_freq = self.qubit.frequency

    def _calibrate_one_point(self):
        n0_0 = 0.0
        n0_1 = 0.0
        for state in [0,1]:
            self.seq_params['state'] = state
            data, _ = self.run_sweeps()
            norm_data = quick_norm_data(data)

            if self.fit_ramsey_freq is None:
                fit = RamseyFit(self.ramsey_delays, norm_data)
                self.fit_ramsey_freq = fit.fit_params["f"]
                logger.info(f"Found Ramsey Frequency of :{self.fit_ramsey_freq/1e3:.3f} kHz.")

            state_data = 0.5*(1 - norm_data) #renormalize data to match convention in CLEAR paper from IBM

            fit = PhotonNumberFit(self.ramsey_delays, state_data, self.T2, self.fit_ramsey_freq*2*np.pi, self.kappa,
                                self.chi, self.T1factor, state)

            self.plot_ramsey[f"Data - {state} State"] = (self.ramsey_delays, state_data)
            self.plot_ramsey[f"Fit - {state} State"] = (self.ramsey_delays, fit.model(self.ramsey_delays))

            if state == 1:
                n0_1 = fit.fit_params["n0"]
            else:
                n0_0 = fit.fit_params["n0"]

        return n0_0, n0_1

    def _calibrate(self):

        self.fit_ramsey_freq = None
        self.seq_params["tau"] = self.tau

        self.epsilon1 = []
        self.epsilon2 = []
        self.iteration = 0
        self.n0 = []
        self.n1 = []

        def objective_function(x):
            self.seq_params['eps1'] = x[0]
            self.seq_params['eps2'] = x[1]

            self.epsilon1.append(x[0])
            self.epsilon2.append(x[1])

            logger.info(f"eps1 = {x[0]}, eps2 = {x[1]}")
            n0_0, n0_1 = self._calibrate_one_point()

            self.iteration +=1
            self.n0.append(n0_0)
            self.n1.append(n0_1)

            self.plot_clear['epsilons'] = (np.array(self.epsilon1), np.array(self.epsilon2))
            self.plot_n['n0'] = (np.arange(self.iteration), np.array(self.n0))
            self.plot_n['n1'] = (np.arange(self.iteration), np.array(self.n1))


            return np.sqrt(n0_0**2 + n0_1**2)

        x0 = [self.eps1, self.eps2]
        minim = minimize(objective_function, x0, method='Nelder-Mead', tol=0.01, options={'maxiter': 100})
        self.eps1 = minim.x[0]
        self.eps2 = minim.x[1]


        # xpoints = np.linspace(0.0, 2*self.eps1, self.nsteps)
        # self.seq_params['eps2'] = self.eps2
        # n0vec = np.zeros(self.nsteps)
        # n1vec = np.zeros(self.nsteps)
        # for k, xp in enumerate(xpoints):
        #     self.seq_params['eps1'] = xp
        #     n0vec[k], n1vec[k] = self._calibrate_one_point()
        #     self.plot_clear['Sweep 0, State 0'] = (xpoints, n0vec)
        #     self.plot_clear['Sweep 0, State 1'] = (xpoints, n1vec)
        #
        # fit0 = QuadraticFit(xpoints, n0vec)
        # fit1 = QuadraticFit(xpoints, n1vec)
        # finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
        # self.plot_clear[f'Fit Sweep 0, State 0'] = (finer_xpoints, fit0.model(finer_xpoints))
        # self.plot_clear[f'Fit Sweep 0, State 1'] = (finer_xpoints, fit1.model(finer_xpoints))
        # best_guess = 0.5*(fit0.fit_params["x0"]+ fit1.fit_params["x0"])
        # logger.info(f"Found best epsilon1 = {best_guess:.6f}")
        # self.seq_params['eps1'] = best_guess
        #
        # xpoints = np.linspace(0.0, 2*self.eps2, self.nsteps)
        # n0vec = np.zeros(self.nsteps)
        # n1vec = np.zeros(self.nsteps)
        # for k, xp in enumerate(xpoints):
        #     self.seq_params['eps2'] = xp
        #     n0vec[k], n1vec[k] = self._calibrate_one_point()
        #     self.plot_clear['Sweep 1, State 0'] = (xpoints, n0vec)
        #     self.plot_clear['Sweep 1, State 1'] = (xp# xpoints = np.linspace(0.0, 2*self.eps1, self.nsteps)
        # self.seq_params['eps2'] = self.eps2
        # n0vec = np.zeros(self.nsteps)
        # n1vec = np.zeros(self.nsteps)
        # for k, xp in enumerate(xpoints):
        #     self.seq_params['eps1'] = xp
        #     n0vec[k], n1vec[k] = self._calibrate_one_point()
        #     self.plot_clear['Sweep 0, State 0'] = (xpoints, n0vec)
        #     self.plot_clear['Sweep 0, State 1'] = (xpoints, n1vec)
        #
        # fit0 = QuadraticFit(xpoints, n0vec)
        # fit1 = QuadraticFit(xpoints, n1vec)
        # finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
        # self.plot_clear[f'Fit Sweep 0, State 0'] = (finer_xpoints, fit0.model(finer_xpoints))
        # self.plot_clear[f'Fit Sweep 0, State 1'] = (finer_xpoints, fit1.model(finer_xpoints))
        # best_guess = 0.5*(fit0.fit_params["x0"]+ fit1.fit_params["x0"])
        # logger.info(f"Found best epsilon1 = {best_guess:.6f}")
        # self.seq_params['eps1'] = best_guess
        #
        # xpoints = np.linspace(0.0, 2*self.eps2, self.nsteps)
        # n0vec = np.zeros(self.nsteps)
        # n1vec = np.zeros(self.nsteps)
        # for k, xp in enumerate(xpoints):
        #     self.seq_params['eps2'] = xp
        #     n0vec[k], n1vec[k] = self._calibrate_one_point()
        #     self.plot_clear['Sweep 1, State 0'] = (xpoints, n0vec)
        #     self.plot_clear['Sweep 1, State 1'] = (xpoints, n1vec)
        # fit0 = QuadraticFit(xpoints, n0vec)
        # fit1 = QuadraticFit(xpoints, n1vec)
        # finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
        # self.plot_clear[f'Fit Sweep 1, State 0'] = (finer_xpoints, fit0.model(finer_xpoints))
        # self.plot_clear[f'Fit Sweep 1, State 1'] = (finer_xpoints, fit1.model(finer_xpoints))
        # best_guess = 0.5*(fit0.fit_params["x0"]+ fit1.fit_params["x0"])
        # logger.info(f"Found best epsilon2 = {best_guess:.6f}")
        # self.seq_params['eps2'] = best_guess
        #
        # xpoints = np.linspace(0.0, 2*self.seq_params["eps1"], self.nsteps)
        # n0vec = np.zeros(self.nsteps)
        # n1vec = np.zeros(self.nsteps)
        # for k, xp in enumerate(xpoints):
        #     self.seq_params['eps1'] = xp
        #     n0vec[k], n1vec[k] = self._calibrate_one_point()
        #     self.plot_clear['Sweep 2, State 0'] = (xpoints, n0vec)
        #     self.plot_clear['Sweep 2, State 1'] = (xpoints, n1vec)
        # fit0 = QuadraticFit(xpoints, n0vec)
        # fit1 = QuadraticFit(xpoints, n1vec)
        # finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
        # self.plot_clear[f'Fit Sweep 2, State 0'] = (finer_xpoints, fit0.model(finer_xpoints))
        # self.plot_clear[f'Fit Sweep 2, State 1'] = (finer_xpoints, fit1.model(finer_xpoints))
        # best_guess = 0.5*(fit0.fit_params["x0"]+ fit1.fit_params["x0"])
        # logger.info(f"Found best epsilon1 = {best_guess:.6f}")
        # self.seq_params['eps1'] = best_guessoints, n1vec)
        # fit0 = QuadraticFit(xpoints, n0vec)
        # fit1 = QuadraticFit(xpoints, n1vec)
        # finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
        # self.plot_clear[f'Fit Sweep 1, State 0'] = (finer_xpoints, fit0.model(finer_xpoints))
        # self.plot_clear[f'Fit Sweep 1, State 1'] = (finer_xpoints, fit1.model(finer_xpoints))
        # best_guess = 0.5*(fit0.fit_params["x0"]+ fit1.fit_params["x0"])
        # logger.info(f"Found best epsilon2 = {best_guess:.6f}")
        # self.seq_params['eps2'] = best_guess
        #
        # xpoints = np.linspace(0.0, 2*self.seq_params["eps1"], self.nsteps)
        # n0vec = np.zeros(self.nsteps)
        # n1vec = np.zeros(self.nsteps)
        # for k, xp in enumerate(xpoints):
        #     self.seq_params['eps1'] = xp
        #     n0vec[k], n1vec[k] = self._calibrate_one_point()
        #     self.plot_clear['Sweep 2, State 0'] = (xpoints, n0vec)
        #     self.plot_clear['Sweep 2, State 1'] = (xpoints, n1vec)
        # fit0 = QuadraticFit(xpoints, n0vec)
        # fit1 = QuadraticFit(xpoints, n1vec)
        # finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
        # self.plot_clear[f'Fit Sweep 2, State 0'] = (finer_xpoints, fit0.model(finer_xpoints))
        # self.plot_clear[f'Fit Sweep 2, State 1'] = (finer_xpoints, fit1.model(finer_xpoints))
        # best_guess = 0.5*(fit0.fit_params["x0"]+ fit1.fit_params["x0"])
        # logger.info(f"Found best epsilon1 = {best_guess:.6f}")
        # self.seq_params['eps1'] = best_guess


        self.eps1 = round(float(self.eps1), 5)
        self.eps2 = round(float(self.eps2), 5)

        logger.info("Found best CLEAR pulse parameters: eps1 = {self.eps1}, eps2 = {self.eps2}")

        self.succeeded = True #TODO: add bounds

    def update_settings(self):
        self.qubit.measure_chan.pulse_params['amp1'] = self.eps1
        self.qubit.measure_chan.pulse_params['amp2'] = self.eps2
        self.qubit.measure_chan.pulse_params['step_length'] = round(float(self.tau), 9)
