# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import os
import asyncio
import time
import numpy as np
from QGL import *
import QGL.config

# Trick QGL and Auspex into using our local config
# from QGL import config_location
curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = curr_dir.replace('\\', '/')  # use unix-like convention
awg_dir  = os.path.abspath(os.path.join(curr_dir, "AWG" ))
cfg_file = os.path.abspath(os.path.join(curr_dir, "test_config.yml"))

ChannelLibrary(library_file=cfg_file)
import auspex.config
# Dummy mode
import auspex.globals
auspex.globals.auspex_dummy_mode = True

auspex.config.configFile = cfg_file
auspex.config.AWGDir     = awg_dir
QGL.config.AWGDir = awg_dir

# Create the AWG directory if it doesn't exist
if not os.path.exists(awg_dir):
    os.makedirs(awg_dir)

from auspex.exp_factory import QubitExpFactory
import auspex.pulse_calibration as cal


def simulate_rabiAmp(num_steps = 20, over_rotation_factor = 0):
    """
    Simulate the output of a RabiAmp experiment of a given number of amp points.
    amps: array of points between [-1,1]

    returns: ideal data
    """
    amps = np.hstack((np.arange(-1, 0, 2./num_steps),
                        np.arange(2./num_steps, 1+2./num_steps, 2./num_steps)))
    xpoints = amps * (1+over_rotation_factor)
    ypoints = -np.cos(2*np.pi*xpoints/2.)
    # repeated twice for X and Y rotations
    return np.tile(ypoints, 2)

class SingleQubitCalTestCase(unittest.TestCase):
    #qubits = ["q1"]
    #instrs = ['BBNAPS1', 'BBNAPS2', 'X6-1', 'Holz1', 'Holz2']
    #filts  = ['Demod-q1', 'Int-q1', 'avg-q1', 'final-avg-buff']
    q = QubitFactory('q1')
    test_settings = auspex.config.yaml_load(cfg_file)

    def test_rabi_amp(self):
        nbr_round_robins = self.test_settings['instruments']['X6-1']['nbr_round_robins']
        filename = './cal_fake_data.txt'
        ideal_data = np.tile(simulate_rabiAmp(), nbr_round_robins)
        np.savetxt(filename, ideal_data)
        rabi_cal = cal.RabiAmpCalibration('q1', num_steps = len(ideal_data)/(2*nbr_round_robins))
        cal.calibrate([rabi_cal])
        os.remove(filename)
        self.assertAlmostEqual(rabi_cal.pi_amp,1,places=2)
        self.assertAlmostEqual(rabi_cal.pi2_amp,0.5,places=2)

# def simulate_measurement(amp, target, numPulses):

#     idealAmp = 0.34
#     noiseScale = 0.05
#     polarization = 0.99 # residual polarization after each pulse
#
#     # data representing over/under rotation of pi/2 pulse
#     # theta = pi/2 * (amp/idealAmp);
#     theta = target * (amp/idealAmp)
#     ks = [ 2**k for k in range(0,numPulses+1)]

#     xdata = [ polarization**x * np.sin(x*theta) for x in ks];
#     xdata = np.insert(xdata,0,-1.0)
#     zdata = [ polarization**x * np.cos(x*theta) for x in ks];
#     zdata = np.insert(zdata,0,1.0)
#     data = np.array([zdata,xdata]).flatten('F')
#     data = np.tile(data,(2,1)).flatten('F')

#     # add noise
#     #data += noiseScale * np.random.randn(len(data));
#     vardata = noiseScale**2 * np.ones((len(data,)));

#     return data, vardata


# class PhaseEstimateTestCase(unittest.TestCase):

#     def test_simulated_measurement(self):

#         numPulses = 9
#         amp = .55
#         direction = 'X'
#         target = np.pi

#         # Using the same simulated data as matlab
#         data, vardata =  simulate_measurement(amp, target, numPulses)

#         # Verify output matches what was previously seen by matlab
#         phase, sigma = pe.phase_estimation(data, vardata, verbose=True)
#         self.assertAlmostEqual(phase,-1.2012,places=4)
#         self.assertAlmostEqual(sigma,0.0245,places=4)

# class OptimizeAmplitudeTestCase(unittest.TestCase):

#     def test_simulated_measurement(self):

#         numPulses = 9
#         amp = .55
#         direction = 'X'
#         target = np.pi

#         # NOTE: this function is a place holder to simulate an AWG generating
#         # a sequence and a digitizer receiving the sequence.  This function
#         # is passed into the optimize_amplitude routine to be able to update
#         # the amplitude as part of the optimization loop.
#         def update_data(amp):
#             data, vardata =  simulate_measurement(amp, target, numPulses)
#             return data,vardata

#         # Verify output matches what was previously seen by matlab
#         amp_out = oa.optimize_amplitude(amp, direction, target, update_data)

#         # NOTE: expected result is from the same input fed to the matlab
#         # routine
#         self.assertAlmostEqual(amp_out,0.3400,places=4)


if __name__ == '__main__':
    unittest.main()
