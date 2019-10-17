__all__ = ["CRLenCalibration", "CRPhaseCalibration", "CRAmpCalibration"]

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

'''Two-qubit gate calibrations'''
class CRCalibration(QubitCalibration):
    def __init__(self,
                 edge,
                 lengths = np.linspace(20, 1020, 21)*1e-9,
                 phases = [0],
                 amps = [0.8],
                 rise_fall = 40e-9,
                 **kwargs):
        self.lengths   = lengths
        self.phases    = phases
        self.amps      = amps
        self.rise_fall = rise_fall
        self.filename  = 'CR/CR'

        self.edge      = edge
        qubits = [edge.source, edge.target]
        super().__init__(qubits, **kwargs)

    def init_plots(self):
        plot = ManualPlotter("CR"+str.lower(self.cal_type.name)+"Fit", x_label=str.lower(self.cal_type.name), y_label='$<Z_{'+self.edge.target.label+'}>$', y_lim=(-1.02,1.02))
        plot.add_data_trace("Data 0", {'color': 'C1'})
        plot.add_fit_trace("Fit 0", {'color': 'C1'})
        plot.add_data_trace("Data 1", {'color': 'C2'})
        plot.add_fit_trace("Fit 1", {'color': 'C2'})
        self.plot = plot
        return [plot]

    def _calibrate(self):
        # run and load normalized data
        qt = self.edge.target
        qs = self.edge.source
        self.qubit = qt

        self.norm_points = {qs.label: (0, 1), qt.label: (0, 1)}
        data, _ =  self.run_sweeps()

        data_t = data[qt.label]
        # fit
        self.opt_par, all_params_0, all_params_1 = fit_CR([self.lengths, self.phases, self.amps], data_t, self.cal_type)
        # plot the result
        xaxis = self.lengths if self.cal_type==CR_cal_type.LENGTH else self.phases if self.cal_type==CR_cal_type.PHASE else self.amps
        finer_xaxis = np.linspace(np.min(xaxis), np.max(xaxis), 4*len(xaxis))

        self.plot["Data 0"] = (xaxis,       data_t[:len(data_t)//2])
        self.plot["Fit 0"] =  (finer_xaxis, np.polyval(all_params_0, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, **all_params_0))
        self.plot["Data 1"] = (xaxis,       data_t[len(data_t)//2:])
        self.plot["Fit 1"] =  (finer_xaxis, np.polyval(all_params_1, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, **all_params_1))

        # Optimal parameter within range of original data!
        if self.opt_par > np.min(xaxis) and self.opt_par < np.max(xaxis):
            self.succeeded = True

    def update_settings(self):
        print("updating settings...")
        self.edge.pulse_params[str.lower(self.cal_type.name)] = float(self.opt_par)
        super(CRCalibration, self).update_settings()

class CRLenCalibration(CRCalibration):
    cal_type = CR_cal_type.LENGTH

    def __init__(self, edge, lengths=np.linspace(20, 1020, 21)*1e-9, phase=0, amp=0.8, rise_fall=40e-9, **kwargs):
        super().__init__(edge, lengths=lengths, phases=[phase], amps=[amp], rise_fall=rise_fall, **kwargs)

    def sequence(self):
        qc, qt = self.qubits
        seqs = [[Id(qc)] + echoCR(qc, qt, length=l, phase = self.phases[0], amp=self.amps[0], riseFall=self.rise_fall).seq + [Id(qc), MEAS(qt)*MEAS(qc)] for l in self.lengths]
        seqs += [[X(qc)] + echoCR(qc, qt, length=l, phase= self.phases[0], amp=self.amps[0], riseFall=self.rise_fall).seq + [X(qc), MEAS(qt)*MEAS(qc)] for l in self.lengths]
        seqs += create_cal_seqs((qt,qc), 2, measChans=(qt,qc))
        return seqs

    def descriptor(self):
         return [
            delay_descriptor(np.concatenate((self.lengths, self.lengths))),
            cal_descriptor(tuple(self.qubits), 2)
        ]


class CRPhaseCalibration(CRCalibration):
    cal_type = CR_cal_type.PHASE

    def __init__(self, edge, length=None, phases=np.linspace(0,2*np.pi,21), amp=0.8, rise_fall=40e-9, **kwargs):
        if not length:
            length = edge.pulse_params['length']
        super().__init__(edge, lengths=[length], phases=phases, amps=[amp], rise_fall=rise_fall, **kwargs)

    def sequence(self):
        qc, qt = self.qubits
        seqs = [[Id(qc)] + echoCR(qc, qt, length=self.lengths[0], phase=ph, amp=self.amps[0], riseFall=self.rise_fall).seq + [X90(qt)*Id(qc), MEAS(qt)*MEAS(qc)] for ph in self.phases]
        seqs += [[X(qc)] + echoCR(qc, qt, length=self.lengths[0], phase=ph, amp=self.amps[0], riseFall=self.rise_fall).seq + [X90(qt)*X(qc), MEAS(qt)*MEAS(qc)] for ph in self.phases]
        seqs += create_cal_seqs((qt,qc), 2, measChans=(qt,qc))
        return seqs

    def descriptor(self):
        return [
            {
                'name': 'phase',
                'unit': 'radians',
                'points': list(self.phases)+list(self.phases),
                'partition': 1
            },
            cal_descriptor(tuple(self.qubits), 2)
        ]

class CRAmpCalibration(CRCalibration):
    cal_type = CR_cal_type.AMP

    def __init__(self, edge, amp_range = 0.4, amp = 0.8, rise_fall = 40e-9, num_CR = 1, **kwargs):
        self.num_CR = num_CR
        length = edge.pulse_params['length']
        phase  = edge.pulse_params['phase']
        if num_CR % 2 == 0:
            logger.error('The number of ZX90 must be odd')
        amps = np.linspace((1-amp_range/2)*amp, (1+amp_range/2)*amp, 21)
        super().__init__(edge, lengths=[length], phases=[phase], amps=amps, rise_fall=rise_fall, **kwargs)

    def sequence(self):
        qc, qt = self.qubits
        seqs = [[Id(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths, phase=self.phases, amp=a, riseFall=self.rise_fall).seq + [Id(qc), MEAS(qt)*MEAS(qc)]
        for a in self.amps]+ [[X(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths, phase= self.phases, amp=a, riseFall=self.rise_fall).seq + [X(qc), MEAS(qt)*MEAS(qc)]
        for a in self.amps] + create_cal_seqs((qt,qc), 2, measChans=(qt,qc))
        return seqs

    def descriptor(self):
        return [{'name': 'amplitude',
                 'unit': None,
                 'points': list(self.amps)+list(self.amps),
                 'partition': 1
                },
                cal_descriptor(tuple(self.qubit), 2)]
