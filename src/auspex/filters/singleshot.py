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
from scipy.special import betaincinv
from sklearn.linear_model import LogisticRegressionCV
from time import sleep
import os
import sys

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    from queue import Queue
else:
    from multiprocessing import Queue

from .filter import Filter
from auspex.parameter import Parameter, FloatParameter, IntParameter, BoolParameter
from auspex.stream import DataStreamDescriptor, InputConnector, OutputConnector, SweepAxis, DataAxis
from auspex.log import logger
import auspex.config as config
import time

class SingleShotMeasurement(Filter):

    save_kernel = BoolParameter(default=False)
    optimal_integration_time = BoolParameter(default=False)
    set_threshold = BoolParameter(default=False)
    zero_mean = BoolParameter(default=False)
    logistic_regression = BoolParameter(default=False)

    sink = InputConnector()
    source = OutputConnector() # Single shot fidelity

    TOLERANCE = 1e-3

    def __init__(self, save_kernel=False, optimal_integration_time=False,
                    zero_mean=False, set_threshold=False,
                    logistic_regression=False, **kwargs):
        super(SingleShotMeasurement, self).__init__(**kwargs)
        if len(kwargs) > 0:
            self.save_kernel.value = save_kernel
            self.optimal_integration_time.value = optimal_integration_time
            self.zero_mean.value = zero_mean
            self.set_threshold.value = set_threshold
            self.logistic_regression.value = logistic_regression

        self.quince_parameters = [self.save_kernel, self.optimal_integration_time,
            self.zero_mean, self.set_threshold, self.logistic_regression]

        self.pdf_data_queue = Queue() #Output queue
        self.fidelity       = self.source

    def update_descriptors(self):

        logger.debug("Updating Plotter %s descriptors based on input descriptor %s", self.filter_name, self.sink.descriptor)
        self.stream = self.sink.input_streams[0]
        self.descriptor = self.sink.descriptor
        try:
            self.time_pts = self.descriptor.axes[self.descriptor.axis_num("time")].points
            self.record_length = len(self.time_pts)
        except ValueError:
            raise ValueError("Single shot filter sink does not appear to have a time axis!")
        self.num_averages = len(self.sink.descriptor.axes[self.descriptor.axis_num("averages")].points)
        self.num_segments = len(self.sink.descriptor.axes[self.descriptor.axis_num("segment")].points)
        self.ground_data = np.zeros((self.record_length, self.num_averages), dtype=np.complex)
        self.excited_data = np.zeros((self.record_length, self.num_averages), dtype=np.complex)
        self.total_points = self.num_segments*self.record_length*self.num_averages # Total points BEFORE sweep axes

        output_descriptor = DataStreamDescriptor()
        output_descriptor.axes = [_ for _ in self.descriptor.axes if type(_) is SweepAxis]
        output_descriptor._exp_src = self.sink.descriptor._exp_src
        output_descriptor.dtype = np.complex128

        if len(output_descriptor.axes) == 0:
            output_descriptor.add_axis(DataAxis("Fidelity", [1]))

        for os in self.fidelity.output_streams:
            os.set_descriptor(output_descriptor)
            os.end_connector.update_descriptors()


    def final_init(self):
        self.fid_buffer = np.empty(self.record_length*self.num_averages*self.num_segments, dtype=np.complex)
        self.idx = 0

    def process_data(self, data):
        """Fill the ground and excited data bins"""

        self.fid_buffer[self.idx:self.idx+len(data)] = data
        self.idx += len(data)

        if self.idx == self.record_length*self.num_averages*self.num_segments:
            self.idx = 0
            reshaped = self.fid_buffer.reshape(self.record_length, -1, order='F')
            self.ground_data = reshaped[:, ::2]
            self.excited_data = reshaped[:, 1::2]
            self.compute_filter()
            if self.logistic_regression.value:
                self.logistic_fidelity()
            if self.save_kernel.value:
                self._save_kernel()
            for os in self.fidelity.output_streams:
                os.push(self.fidelity_result)
            self.pdf_data_queue.put(self.pdf_data)

    def compute_filter(self):
        """Compute the single shot kernel and obtain single-shot measurement
        fidelity.

        Expects that the data will be in self.ground_data and self.excited_data,
        which are (T, N)-shaped numpy arrays, with T the time axis and N the
        number of shots."""
        #get excited and ground state data
        try:
            ground_mean = np.mean(self.ground_data, axis=1)
            excited_mean = np.mean(self.excited_data, axis=1)
        except AttributeError:
            raise Exception("Single shot filter does not appear to have any data!")
        distance = np.abs(np.mean(ground_mean - excited_mean))
        bias = np.mean(ground_mean + excited_mean) / distance
        logger.debug("Found single-shot measurement distance: {} and bias {}.".format(distance, bias))
        #construct matched filter kernel
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        kernel = np.nan_to_num(np.divide(np.conj(ground_mean - excited_mean), np.var(self.ground_data, ddof=1, axis=1)))
        np.seterr(**old_settings)
        #sets kernel to zero when difference is too small, and prevents
        #kernel from diverging when var->0 at beginning of record_length
        kernel = np.multiply(kernel, np.greater(np.abs(ground_mean - excited_mean), self.TOLERANCE * distance))
        #subtract offset to cancel low-frequency fluctuations when integrating
        #raw data (not demod)
        if self.zero_mean.value:
            kernel = kernel - np.mean(kernel)
        logger.debug("Found single shot filter norm: {}.".format(np.sum(np.abs(kernel))))
        #annoyingly numpy's isreal has the opposite behavior to MATLAB's
        if not np.any(np.imag(kernel) > np.finfo(np.complex128).eps):
            #construct analytic signal from Hilbert transform
            kernel = hilbert(np.real(kernel))
        #normalize between -1 and 1
        kernel = kernel / np.amax(np.hstack([np.abs(np.real(kernel)), np.abs(np.imag(kernel))]))
        #apply matched filter
        weighted_ground = self.ground_data * kernel[:, np.newaxis]
        weighted_excited = self.excited_data * kernel[:, np.newaxis]

        if self.optimal_integration_time.value:
            #take cumulative sum up to each time step
            ground_I = np.real(weighted_ground)
            ground_Q = np.imag(weighted_ground)
            excited_I = np.real(weighted_excited)
            excited_Q = np.imag(weighted_excited)
            int_ground_I = np.cumsum(ground_I, axis=0)
            int_ground_Q = np.cumsum(ground_Q, axis=0)
            int_excited_I = np.cumsum(excited_I, axis=0)
            int_excited_Q = np.cumsum(excited_Q, axis=0)
            I_mins = np.amin(np.minimum(int_ground_I, int_excited_I), axis=1)
            I_maxes = np.amax(np.maximum(int_ground_I, int_excited_I), axis=1)
            num_times = int_ground_I.shape[0]
            fidelities = np.zeros((num_times, ))
            #Loop through each integration point; estimate the CDF and
            #then calculate best measurement fidelity
            for pt in range(num_times):
                bins = np.linspace(I_mins[pt], I_maxes[pt], 100)
                g_PDF = np.histogram(int_ground_I[pt, :], bins)[0]
                e_PDF = np.histogram(int_excited_I[pt,:], bins)[0]
                fidelities[pt] = np.sum(np.abs(g_PDF - e_PDF)) / np.sum(g_PDF + e_PDF)
            best_idx = fidelities.argmax(axis=0)
            self.best_integration_time = best_idx
            logger.info("Found best integration time at {} out of {} decimated points.".format(best_idx, num_times))
            #redo calculation with KDEs to get a more accurate estimate
            bins = np.linspace(I_mins[best_idx], I_maxes[best_idx], 100)
            g_KDE = gaussian_kde(int_ground_I[best_idx, :])
            e_KDE = gaussian_kde(int_excited_I[best_idx, :])
            g_PDF = g_KDE(bins)
            e_PDF = e_KDE(bins)
        else:
            ground_I = np.sum(np.real(weighted_ground), axis=0)
            ground_Q = np.sum(np.imag(weighted_excited), axis=0)
            excited_I = np.sum(np.real(weighted_excited), axis=0)
            excited_Q = np.sum(np.imag(weighted_excited), axis=0)
            I_min = np.amin(np.minimum(ground_I, excited_I))
            I_max = np.amax(np.maximum(ground_I, excited_I))
            bins = np.linspace(I_min, I_max, 100)
            g_KDE = gaussian_kde(ground_I)
            e_KDE = gaussian_kde(excited_I)
            g_PDF = g_KDE(bins)
            e_PDF = e_KDE(bins)

        self.kernel = kernel
        max_F_I = 1 - 0.5 * (1 - 0.5 * (bins[2] - bins[1]) * np.sum(np.abs(g_PDF - e_PDF)))
        self.pdf_data = {"Max I Fidelity": max_F_I,
                         "I Bins": bins,
                         "Ground I PDF": g_PDF,
                         "Excited I PDF": e_PDF}

        if self.set_threshold.value:
            indmax = (np.abs(np.cumsum(g_PDF / np.sum(g_PDF))
                        - np.cumsum(e_PDF / np.sum(e_PDF)))).argmax(axis=0)
            self.pdf_data["I Threshold"] = bins[indmax]
            logger.info("Single shot kernel found I threshold at {}.".format(bins[indmax]))

        if self.optimal_integration_time.value:
            mu_g, sigma_g = norm.fit(int_ground_I[best_idx, :])
            mu_e, sigma_e = norm.fit(int_excited_I[best_idx, :])
        else:
            mu_g, sigma_g = norm.fit(ground_I)
            mu_e, sigma_e = norm.fit(excited_I)
        self.pdf_data["Ground I Gaussian PDF"] = norm.pdf(bins, mu_g, sigma_g)
        self.pdf_data["Excited I Gaussian PDF"] = norm.pdf(bins, mu_e, sigma_e)

        #calculate kernel density estimates for other quadrature
        if self.optimal_integration_time.value:
            Q_min = np.amin([int_ground_Q[best_idx,:], int_excited_Q[best_idx,:]])
            Q_max = np.argmax([int_ground_Q[best_idx,:], int_excited_Q[best_idx,:]])
            qbins = np.linspace(Q_min, Q_max, 100)
            g_KDE = gaussian_kde(int_ground_Q[best_idx, :])
            e_KDE = gaussian_kde(int_excited_Q[best_idx, :])
        else:
            qbins = np.linspace(np.amin([ground_Q, excited_Q]), np.amax([ground_Q, excited_Q]), 100)
            g_KDE = gaussian_kde(ground_Q)
            e_KDE = gaussian_kde(excited_Q)
        self.pdf_data["Q Bins"] = qbins
        g_PDF_Q = g_KDE(qbins)
        e_PDF_Q = e_KDE(qbins)
        self.pdf_data["Ground Q PDF"] =  g_PDF_Q
        self.pdf_data["Excited Q PDF"] =  e_PDF_Q
        self.pdf_data["Max Q Fidelity"] = 1 - 0.5 * (1 - 0.5 * (qbins[2] - qbins[1]) * np.sum(np.abs(g_PDF_Q - e_PDF_Q)))

        if self.optimal_integration_time.value:
            mu_g, sigma_g = norm.fit(int_ground_Q[best_idx, :])
            mu_e, sigma_e = norm.fit(int_excited_Q[best_idx, :])
        else:
            mu_g, sigma_g = norm.fit(ground_Q)
            mu_e, sigma_e = norm.fit(excited_Q)
        self.pdf_data["Ground Q Gaussian PDF"] = norm.pdf(bins, mu_g, sigma_g)
        self.pdf_data["Excited Q Gaussian PDF"] = norm.pdf(bins, mu_e, sigma_e)

        self.fidelity_result = self.pdf_data["Max I Fidelity"] + 1j * self.pdf_data["Max Q Fidelity"]
        logger.info("Single shot fidelity filter found: {}".format(self.fidelity_result))

    def logistic_fidelity(self):
        #group data and assign state labels
        gnd_features = np.hstack([np.real(self.ground_data.T),
                                np.imag(self.ground_data.T)])
        ex_features = np.hstack([np.real(self.excited_data.T),
                                np.imag(self.excited_data.T)])
        #liblinear wants arrays in C order
        features = np.ascontiguousarray(np.vstack([gnd_features, ex_features]))
        state = np.ascontiguousarray(np.hstack([np.zeros(self.ground_data.shape[1]),
                                                np.ones(self.excited_data.shape[1])]))
        #Set up logistic regression with cross-validation using liblinear.
        #Cs sets the inverse of the regularization strength, which will be optimized
        #through cross-validation. Uses the default Stratified K-Folds
        #CV generator, with 3 folds.
        #This is set up to be as consistent with the MATLAB implementation
        #as I can make it. --GJR
        Cs = np.logspace(-1,2,5)
        logreg = LogisticRegressionCV(Cs, cv=3, solver='liblinear')
        logreg.fit(features, state) #fit the model
        predictions = logreg.predict(features) #in-place classification
        score = logreg.score(features,state) #mean accuracy of classification
        N = len(predictions)
        S = np.sum(predictions == state) #how many we got right
        #now calculate confidence intervals
        c = 0.95
        flo = betaincinv(S+1, N-S+1, (1-c)/2., )
        fhi = betaincinv(S+1, N-S+1, (1+c)/2., )
        logger.info(("In-place logistic regression fidelity: " +
                "{:.2f}% ({:.2f}, {:.2f})".format(100*score, 100*flo, 100*fhi)))

    def _save_kernel(self):
        import QGL.config as qconfig
        if not qconfig.KernelDir or not os.path.exists(qconfig.KernelDir):
            logger.warning("No kernel directory provided, please set auspex.config.KernelDir")
            logger.warning("Saving kernel to local directory.")
            dir = "./"
        else:
            dir = qconfig.KernelDir
        try:
            logger.info(self.filter_name)
            filename = self.filter_name + "_kernel.txt"
            header = "Single shot fidelity filter - {}:\nSource: {}".format(time.strftime("%m/%d/%y -- %H:%M"), self.filter_name)
            np.savetxt(os.path.join(dir, filename), self.kernel, header=header, comments="#")
        except (AttributeError, IOError) as ex:
            raise AttributeError("Could not save single shot fidelity kernel!") from ex
