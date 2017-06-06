# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['SingleShotMeasurement']

import numpy as np
from scipy.signal import hilbert
from scipy.stats import gaussian_kde, norm

from .filter import Filter
from auspex.parameter import Parameter, FloatParameter, IntParameter, BoolParameter
from auspex.stream import DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger

class SingleShotMeasurement(Filter):

    save_kernel = BoolParameter()
    optimal_integration_time = BoolParameter()
    set_threshold = BoolParameter()
    zero_mean = BoolParameter()
    logistic_regression = BoolParameter()

    TOLERANCE = 1e-3

    def __init__(self, **kwargs):
        super(SingleShotMeasurement, self).__init__(**kwargs)
        if len(kwargs) > 0:
            self.save_kernel.value = kwargs['save_kernel']
            self.optimal_integration_time.value = kwargs['optimal_integration_time']
            self.zero_mean.value = kwargs['zero_mean']
            self.set_threshold.value = kwargs['set_threshold']
            self.logistic_regression.value = kwargs['logistic_regression']

    def update_descriptors(self):
        pass

    async def process_data(self, data):
        #get excited and ground state data
        ground_mean = np.mean(ground_data, axis=1)
        excited_mean = np.mean(excited_data, axis=1)
        distance = np.abs(np.mean(ground_mean - excited_mean))
        bias = np.mean(ground_mean + excited_mean) / distance
        logger.info("Found single-shot measurement distance: {} and bias {}.".format(distance, bias))
        #construct matched filter kernel
        kernel = np.nan_to_num(np.divide(np.conj(ground_mean - excited_mean), np.var(ground_data, axis=2)))
        #sets kernel to zero when difference is too small, and prevents
        #kernel from diverging when var->0 at beginning of record_length
        kernel = np.multiply(kernel, np.greater(np.abs(ground_mean, excited_mean), self.TOLERANCE * distance))
        #subtract offset to cancel low-frequency fluctuations when integrating
        #raw data (not demod)
        if self.zero_mean:
            kernel = kernel - np.mean(kernel)
        logger.info("Found single shot filter norm: {}.".format(np.sum(np.abs(kernel))))
        if np.any(np.isreal(kernel)):
            #construct analytic signal from Hilbert transform
            kernel = hilbert(kernel)
        #normalize between -1 and 1
        kernel = kernel / np.maximum(np.vstack(np.abs(np.real(kernel)), np.abs(np.imag(kernel))))
        #apply matched filter
        weighted_ground = np.multiply(ground_data, kernel)
        weighted_excited = np.multiply(excited_data, kernel)

        if self.optimal_integration_time:
            #take cumulative sum up to each time step
            ground_I = np.real(weighted_ground)
            ground_Q = np.imag(weighted_ground)
            excited_I = np.real(weighted_excited)
            excited_Q = np.imag(weighted_excited)
            int_ground_I = np.cumsum(ground_I)
            int_ground_Q = np.cumsum(ground_Q)
            int_excited_I = np.cumsum(excited_I)
            int_excited_Q = np.cumsum(excited_Q)
            I_mins = np.amin(np.minimum(int_ground_I, int_excited_I), axis=1)
            I_maxes = np.amax(np.maximum(int_ground_I, int_excited_I), axis=1)
            num_times = int_ground_I.shape[0]
            fidelities = np.zeros((num_times, ))
            #Loop through each integration point; estimate the CDF and
            #then calculate best measurement fidelity
            for pt in range(1,num_times-1,2):
                bins = np.linspace(I_mins[pt], I_maxes[pt], 100)
                g_PDF = np.histogram(int_ground_I[pt, :], bins)[0]
                e_PDF = np.histogram(int_excited_I[pt,:], bins)[0]
                fidelities[pt] = np.sum(np.abs(g_PDF - e_PDF)) / np.sum(g_PDF + e_PDF)
            best_idx = fidelities.argmax(axis=0)
            self.best_integration_time = best_idx
            #redo calculation with KDEs to get a more accurate estimate
            bins = np.linspace(I_mins[best_idx], I_maxes[best_idx], 100)
            g_KDE = gaussian_kde(ground_I)
            e_KDE = gaussian_kde(ground_Q)
            g_PDF = e_KDE(bins)
            e_PDF = g_KDE(bins)
        else:
            ground_I = np.sum(np.real(weighted_ground))
            ground_Q = np.sum(np.imag(weighted_excited))
            excited_I = np.sum(np.real(weighted_excited))
            excited_Q = np.sum(np.imag(weighted_excited))
            I_min = np.amin(np.minimum(ground_I, excited_I))
            I_max = np.amax(np.maximum(ground_I, excited_I))
            bins = np.linspace(I_min, I_max, 100)
            g_KDE = gaussian_kde(ground_I)
            e_KDE = gaussian_kde(ground_Q)
            g_PDF = e_KDE(bins)
            e_PDF = g_KDE(bins)

        if self.save_kernel:
            logger.debug("Kernel saving not yet implemented!")
            pass

        max_F_I = 1 - 0.5 * (1 - 0.5 * (bins[2] - bins[1]) * np.sum(np.abs(g_PDF - e_PDF)))
        self.pdf_data = {"Max I Fidelity": max_F_I,
                         "I Bins": bins,
                         "Ground I PDF": g_PDF,
                         "Excited I PDF": e_PDF}

        if self.set_threshold:
            indmax = (np.amax(np.abs(np.cumsum(g_PDF / np.sum(g_PDF)))
                        - np.cumsum(e_PDF / np.sum(e_PDF)))).argmax(axis=0)
            self.pdf_data["I Threshold"] = bins[indmax]
            logger.info("Single shot kernel found I threshold at {}.".format(bins[indmax]))

        if self.optimal_integration_time:
            mu_g, sigma_g = norm.fit(int_ground_I[best_idx, :])
            mu_e, sigma_e = norm.fit(int_excited_I[best_idx, :])
        else:
            mu_g, sigma_g = norm.fit(ground_I)
            mu_e, sigma_e = norm.fit(excited_I)
        self.pdf_data["Ground I Gaussian PDF"] = norm.pdf(bins, mu_g, sigma_g)
        self.pdf_data["Excited I Gaussian PDF"] = norm.pdf(bins, mu_e, sigma_e)

        #calculate kernel density estimates for other quadrature
        if self.optimal_integration_time:
            self.pdf_data["Q Bins"] = np.linspace(np.amax(int_ground_Q[best_idx, :], int_excited_Q[best_idx]),
                                                    np.amax(int_ground_Q[best_idx, :], int_excited_Q[best_idx]), 100)
            g_KDE = gaussian_kde(int_ground_Q[best_idx, :])
            e_KDE = gaussian_kde(int_excited_Q[best_idx, :])
            self.pdf_data["Ground Q PDF"] =  g_KDE(self.pdf_data["Q Bins"])
            self.pdf_data["Excited Q PDF"] =  e_KDE(self.pdf_data["Q Bins"])
        else:
            self.pdf_data["Q Bins"] = np.linspace(np.amin([ground_Q, excited_I]), np.amax([ground_Q, excited_Q]), 100)
            g_KDE = gaussian_kde(ground_Q)
            e_KDE = gaussian_kde(excited_Q)
            self.pdf_data["Ground Q PDF"] =  g_KDE(self.pdf_data["Q Bins"])
            self.pdf_data["Excited Q PDF"] =  e_KDE(self.pdf_data["Q Bins"])
        self.pdf_data["Max Q Fidelity"] = 1 - 0.5 * (1 - 0.5 *
            (self.pdf_data["Q Bins"][1] - self.pdf_data["Q Bins"][0])
            * np.sum(np.abs(self.pdf_data["Ground Q PDF"] - self.pdf_data["Excited Q PDF"])) )
        if self.optimal_integration_time:
            mu_g, sigma_g = norm.fit(int_ground_Q[best_idx, :])
            mu_e, sigma_e = norm.fit(int_excited_Q[best_idx, :])
        else:
            mu_g, sigma_g = norm.fit(ground_Q)
            mu_e, sigma_e = norm.fit(excited_Q)
        self.pdf_data["Ground Q Gaussian PDF"] = norm.pdf(bins, mu_g, sigma_g)
        self.pdf_data["Excited Q Gaussian PDF"] = norm.pdf(bins, mu_e, sigma_e)

        self.fidelity_result = self.pdf_data["Max I Fidelity"] + 1j * self.pdf_data["Max Q Fidelity"]
        logger.info("Single shot fidelity filter found: {}".format(self.fidelity_result))

        if self.logistic_regression:
            logger.error("Logistic regression in sigle-shot fidelity filter not yet implemented.")
            pass

    async def on_done(self, data):
        pass
