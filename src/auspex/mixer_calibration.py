# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from QGL import *
from QGL import config as QGLconfig
# from QGL.BasicSequences.helpers import create_cal_seqs, time_descriptor, cal_descriptor
import auspex.config as config
from copy import copy
import os
import json

from scipy.optimize import curve_fit

from auspex.exp_factory import QubitExpFactory
from auspex.analysis.io import load_from_HDF5
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.analysis.fits import *
from auspex.analysis.helpers import normalize_data

from JSONLibraryUtils import LibraryCoders

def find_null_offset(xpts, powers):
    """Finds the offset corresponding to the minimum power using a fit to the measured data"""
    def model(x, a, b, c):
        return 10*np.log10(a*(x - b)**2 + c)
    min_idx = np.argmin(powers)
    fit = curve_fit(xpts, powers, p0=[1, xpts[min_idx], pow(10, powers[min_idx]/10)])
    best_offset = np.real(fit[1])
    best_offset = np.minimum(best_offset, xpts[-1])
    best_offset = np.maximum(best_offset, xpts[0])
    fit_pts = np.array([np.real(model(x, *fit)) for x in xpts])
    return best_offset, fit_pts




class MixerCalibration(Experiment):

    def __init__(self, qubit):
        super(MixerCalibration, self).__init__()

        self.settings = config.yaml_load(config.configFile)
        #search for spectrum analyzer
        self.sa_settings = [instr for instr in self.settings['instruments'].items() if instr[1]['type'] == 'SpectrumAnalyzer']
        if len(self.sa_settings) != 1:
            raise ValueError("More than one spectrum analyzer is defined in the configuration file.")
        self.sa_settings = self.sa_settings[0]
        try:
            self.qubit_settings = self.settings['qubits'][qubit]
        except KeyError as ex:
            raise ValueError("Could not find qubit {} in the qubit configuration file.".format(qubit)) from ex
