# Copyright 2018 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import asyncio
import os
import numpy as np
import h5py
from adapt.refine import refine_1D

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import WriteToHDF5
from auspex.log import logger
from auspex.analysis.io import load_from_HDF5
from auspex.analysis import fits

config.load_meas_file(config.find_meas_file())

class TestFitMethods(unittest.TestCase):
    # -----
    # Note:  with the methods named "Test<xyz>" the unit test invocation
    # cites zero tests run;  renaming the methods as "test<xyz>" render them
    # runable and revealed call syntax issues...
    #
    #def TestT1Fit(self):
    def testT1Fit(self):  # Added try/except trap with alternate syntax -- TJR
        """Test the fit_t1 experiment """

        # Set parameters and generate synthetic data in natural units
        T1 = 40 * 1e3 # 40 us
        xdata = np.arange(20,120020,1000)
        synth_data = fits.t1_model(xdata, 1, T1, 0)
        synth_data = [np.random.normal(scale=0.1) + i for i in synth_data]

        # fit the T1 data
        result, result_err = fits.fit_t1(xdata,synth_data)

        # -----
        # EEE Original logic routinely raising IndexError to include:
        #     self.assertAlmostEqual(T1, result[1], delta=result_err[1])
        # IndexError: invalid index to scalar variable.
        #
        # Wrapped in a temporary try/except block for experimentation.
        try:
            self.assertAlmostEqual(T1, result[1], delta=result_err[1])
            #
            # EEE This routinely producing IndexError to include the following:
            #     self.assertAlmostEqual(T1, result[1], delta=result_err[1])
            # IndexError: invalid index to scalar variable.
            #
        except Exception as e:
            logger.warning( "Original ~array centric assertAlmostEqual call failed!" \
            "\n\r   << assertAlmostEqual(T1, result[1], delta=result_err[1])"
            "\n\r      << EEE %s Exception %s" \
            "\n\r      $$ type( result):%s, type( result_err):%s" \
            "\n\r         -- repeating assertAlmostEqual call with simple value references..." \
            "\n\r            << assertAlmostEqual(T1, result, delta=result_err)",
            type( e), e, type( result), type( result_err))
            self.assertAlmostEqual(T1, result, delta=result_err)

    def TestRamseyFit(self):
    #def testRamseyFit(self):    # Raises errors with current definitions
        """Test the fit_ramsey experiment """

        # Set parameters and generate synthetic data
        # note the frequency is set relative to ns (the 'natural' units)
        T2 = 40 * 1e3 # 40 us
        xdata = np.arange(20,120020,1000)
        synth_data = fits.ramsey_1f(xdata, 40e-6, 1,  T2, 0, 0)
        synth_data = [np.random.normal(scale=0.05) + i for i in synth_data]

        # fit the T2 data
        result, result_err, popt, perr = fits.fit_ramsey(xdata,synth_data)

        # check the outputs
        self.assertAlmostEqual(T2, result[0], delta=result_err[1])


    #def TestRBFit(self):
    def testRBFit(self): # Runs ok with no changes -- TJR
        """Test the fit_single_qubit_rb experiment """

        # Set parameters and generate synthetic data
        repeats = 32
        lengths = [4,8,16,32,64,128,256]
        seq_lengths = np.concatenate([[length]*repeats for length in lengths])
        r_avg = 2e-3 # avearge error per gate
        synth_data = [fits.rb_model(n, 1, r_avg * 2, 0) for n in seq_lengths]
        synth_data = [np.random.normal(scale=0.05) + i for i in synth_data]

        # fit the RB data
        avg_infidelity, avg_infidelity_err, popt, pcov = \
            fits.fit_single_qubit_rb(synth_data, lengths)

        # check the outputs
        self.assertAlmostEqual(r_avg, avg_infidelity, delta=avg_infidelity_err)


if __name__ == '__main__':
    unittest.main()
