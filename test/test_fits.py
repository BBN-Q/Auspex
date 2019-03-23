# Copyright 2018 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest

import os
import numpy as np
from auspex.analysis import fits, qubit_fits

import matplotlib.pyplot as plt


class FitAssertion(object):

    def assertFitInterval(self, p0, name, fit, tol=5):
        low = fit.fit_params[name] - tol*fit.fit_errors[name]
        high = fit.fit_params[name] + tol*fit.fit_errors[name]
        test =  (low < p0 < high)
        if not test:
            raise AssertionError(f"Fit parameter {name}: {p0} is outside of interval ({low}, {high}).")



class TestFitMethods(unittest.TestCase, FitAssertion):

    def test_LorentzFit(self):

        p0 = [3, 0.25, 0.4, 1.0]
        x = np.linspace(-1, 1, 201)
        y = fits.LorentzFit._model(x, *p0)
        noise = np.random.randn(y.size) * np.max(y)/10
        y += noise
        fit = fits.LorentzFit(x, y, make_plots=False)
        self.assertFitInterval(p0[0], "A", fit)
        self.assertFitInterval(p0[1], "b", fit)
        self.assertFitInterval(p0[2], "c", fit)

    def test_T1Fit(self):

        p0 = [2.0, 15, -1]
        x = np.linspace(0, 80, 201)
        y = qubit_fits.T1Fit._model(x, *p0)
        noise = np.random.randn(y.size)/10
        y += noise 
        fit = qubit_fits.T1Fit(x, y, make_plots=False)
        self.assertFitInterval(p0[0], "A", fit)
        self.assertFitInterval(p0[1], "T1", fit)
        self.assertFitInterval(p0[2], "A0", fit)

    def test_RabiAmpFit(self):

        p0 = [0.3, 0.76, 1.0, 0.01]
        x = np.linspace(-0.5, 0.5, 201)
        y = qubit_fits.RabiAmpFit._model(x, *p0)
        noise = np.random.randn(y.size) * np.max(y)/10
        y += noise
        fit = qubit_fits.RabiAmpFit(x, y, make_plots=False)
        self.assertFitInterval(p0[1], "Api", fit)

    def test_RabiWidthFit(self):

        p0 = [0.1, 0.34, 9, 0.2, 0.02]
        x = np.linspace(0, 20, 201)
        y = qubit_fits.RabiWidthFit._model(x, *p0)
        noise = np.random.randn(y.size) * np.max(y)/10
        y += noise
        fit = qubit_fits.RabiWidthFit(x, y, make_plots=False)
        self.assertFitInterval(p0[2], "T", fit)

    def test_RamseyFit_1f(self):
        #x, f, A, tau, phi, y0

        p0 = [0.22, 1.0, 11.3, 0.01, 0.1]
        x = np.linspace(0, 30, 201)
        y = qubit_fits.RamseyFit._model_1f(x, *p0)
        noise = np.random.randn(y.size) * np.max(y)/10
        y += noise

        fit = qubit_fits.RamseyFit(x, y, two_freqs=False, make_plots=False)
        self.assertFitInterval(p0[0], "f", fit)
        self.assertFitInterval(p0[2], "tau", fit)

    @unittest.skip("Need better test case for 2-frequency Ramsey fit.")
    def test_RamseyFit_2f(self):
        #x, f, A, tau, phi, y0

        p0 = [0.22, 0.1, 1.0, 0.6, 11.3, 16.4, 0.01, 0.02, 0.1, 0.08]
        x = np.linspace(0, 30, 201)
        y = qubit_fits.RamseyFit._model_2f(x, *p0)
        noise = np.random.randn(y.size) * np.max(y)/10
        y += noise

        fit = qubit_fits.RamseyFit(x, y, two_freqs=True, AIC=True, make_plots=False)

    @unittest.skip("Fit not particularly stable?")
    def test_SingleQubitRB(self):
        p0 = [0.99, 5e-3, 0.2]

        x = np.array([2**n for n in range(10)])
        y = qubit_fits.SingleQubitRBFit._model(x, *p0)
        noise = np.random.randn(y.size) * p0[0]/100
        y += noise 

        fit = qubit_fits.SingleQubitRBFit(x, y, make_plots=False)
        self.assertFitInterval(p0[1], "r", fit, tol=20)
        


if __name__ == '__main__':
    unittest.main()
