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
cfg_file = os.path.abspath(os.path.join(curr_dir, "test_measure.yml"))

ChannelLibrary(library_file=cfg_file)

import auspex.config
auspex.config.auspex_dummy_mode = True
auspex.config.configFile        = cfg_file
auspex.config.AWGDir            = awg_dir
QGL.config.AWGDir               = awg_dir

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

def simulate_ramsey(num_steps = 50, maxt = 50e-6, detuning = 100e3, T2 = 40e-6):
    """
    Simulate the output of a Ramsey experiment of a given number of time steps.
    maxt: longest delay (s)
    detuning: frequency detuning (Hz)
    T2: 1/e decay time (s)

    returns: ideal data
    """

    xpoints = np.linspace(0, maxt, num_steps)
    ypoints = np.cos(2*np.pi*detuning*xpoints)*np.exp(-xpoints/T2)
    return ypoints

def simulate_phase_estimation(amp, target, numPulses):
    """
    Simulate the output of a PhaseEstimation experiment with NumPulses.
    amp: initial pulse amplitude
    target: target pulse amplitude

    returns: ideal data and variance
    """
    idealAmp = 0.34
    noiseScale = 0.05
    polarization = 0.99 # residual polarization after each pulse

    # data representing over/under rotation of pi/2 pulse
    # theta = pi/2 * (amp/idealAmp);
    theta = target * (amp/idealAmp)
    ks = [ 2**k for k in range(0,numPulses+1)]

    xdata = [ polarization**x * np.sin(x*theta) for x in ks];
    xdata = np.insert(xdata,0,-1.0)
    zdata = [ polarization**x * np.cos(x*theta) for x in ks];
    zdata = np.insert(zdata,0,1.0)
    data = np.array([zdata,xdata]).flatten('F')
    data = np.tile(data,(2,1)).flatten('F')

    # add noise
    #data += noiseScale * np.random.randn(len(data));
    vardata = noiseScale**2 * np.ones((len(data,)));

    return data, vardata

class SingleQubitCalTestCase(unittest.TestCase):
    """
    Class for unittests of single-qubit calibrations. Tested so far with a dummy X6 digitizer:
    * RabiAmpCalibration
    * RamseyCalibration
    Ideal data are generated and stored into a temporary file, whose name is set by the X6 property `ideal_data`. Calibrations which span over multiple experiments load different columns of these ideal data. The column (and experiment) number is set by an incremental counter, also a digitizer property `exp_step`. Artificial noise is added by the X6 dummy instrument.
    """

    q = QubitFactory('q1')
    test_settings = auspex.config.load_meas_file(cfg_file)
    nbr_round_robins = test_settings['instruments']['X6-1']['nbr_round_robins']
    filename = './cal_fake_data.npy'

    def test_rabi_amp(self):
        """
        Test RabiAmpCalibration. Ideal data generated by simulate_rabiAmp.
        """

        ideal_data = [np.tile(simulate_rabiAmp(), self.nbr_round_robins)]
        np.save(self.filename, ideal_data)
        rabi_cal = cal.RabiAmpCalibration(self.q.label, num_steps = len(ideal_data[0])/(2*self.nbr_round_robins))
        cal.calibrate([rabi_cal])
        os.remove(self.filename)
        self.assertAlmostEqual(rabi_cal.pi_amp,1,places=2)
        self.assertAlmostEqual(rabi_cal.pi2_amp,0.5,places=2)
        #test update_settings
        new_settings = auspex.config.load_meas_file(cfg_file)
        self.assertAlmostEqual(rabi_cal.pi_amp, new_settings['qubits'][self.q.label]['control']['pulse_params']['piAmp'], places=4)
        self.assertAlmostEqual(rabi_cal.pi2_amp, new_settings['qubits'][self.q.label]['control']['pulse_params']['pi2Amp'], places=4)
        #restore original settings
        auspex.config.dump_meas_file(self.test_settings, cfg_file)

    def sim_ramsey(self, set_source = True):
        """
        Simulate a RamseyCalibration run. Ideal data are generated by simulate_ramsey.
        set_source: True (False) sets the source (qubit) frequency.
        """
        ideal_data = [np.tile(simulate_ramsey(detuning = 90e3), self.nbr_round_robins), np.tile(simulate_ramsey(detuning = 45e3), self.nbr_round_robins)]
        np.save(self.filename, ideal_data)
        ramsey_cal = cal.RamseyCalibration(self.q.label, num_steps = len(ideal_data[0])/(self.nbr_round_robins), added_detuning = 0e3, delays=np.linspace(0.0, 50.0, 50)*1e-6, set_source = set_source)
        cal.calibrate([ramsey_cal])
        os.remove(self.filename)
        return ramsey_cal

    def test_ramsey_set_source(self):
        """
        Test RamseyCalibration with source frequency setting.
        """
        ramsey_cal = self.sim_ramsey()
        self.assertAlmostEqual(ramsey_cal.fit_freq/1e9, (self.test_settings['instruments']['Holz2']['frequency'] + 90e3)/1e9, places=4)
        #test update_settings
        new_settings = auspex.config.load_meas_file(cfg_file)
        self.assertAlmostEqual(ramsey_cal.fit_freq/1e9, new_settings['instruments']['Holz2']['frequency']/1e9, places=4)
        #restore original settings
        auspex.config.dump_meas_file(self.test_settings, cfg_file)
    def test_ramsey_set_qubit(self):
        """
        Test RamseyCalibration with qubit frequency setting.
        """
        ramsey_cal = self.sim_ramsey(False)
        #test update_settings
        new_settings = auspex.config.load_meas_file(cfg_file)
        self.assertAlmostEqual((self.test_settings['qubits'][self.q.label]['control']['frequency']+90e3)/1e6, new_settings['qubits'][self.q.label]['control']['frequency']/1e6, places=2)
        #restore original settings
        auspex.config.dump_meas_file(self.test_settings, cfg_file)
    def test_phase_estimation(self):
        """
        Test generating data for phase estimation
        """
        numPulses = 9
        amp = .55
        direction = 'X'
        target = np.pi

        # Using the same simulated data as matlab
        data, vardata =  simulate_phase_estimation(amp, target, numPulses)

        # Verify output matches what was previously seen by matlab
        phase, sigma = cal.phase_estimation(data, vardata, verbose=False)
        self.assertAlmostEqual(phase,-1.2012,places=4)
        self.assertAlmostEqual(sigma,0.0245,places=4)

    def test_pi_phase_estimation_imag(self):
        """
        Test PiCalibration with phase estimation
        """

        numPulses = 9
        amp = self.test_settings['qubits'][self.q.label]['control']['pulse_params']['piAmp']
        direction = 'X'
        target = np.pi

        # NOTE: this function is a place holder to simulate an AWG generating
        # a sequence and a digitizer receiving the sequence.  This function
        # is passed into the optimize_amplitude routine to be able to update
        # the amplitude as part of the optimization loop.
        def update_data(amp, ct):
                data, vardata =  simulate_phase_estimation(amp, target, numPulses)
                phase, sigma = cal.phase_estimation(data, vardata, verbose=False)
                amp, done_flag = cal.phase_to_amplitude(phase, sigma, amp, target, ct)
                return amp, data, done_flag

        done_flag = 0
        for ct in range(5): #max iterations
            amp, data, done_flag = update_data(amp, ct)
            ideal_data = data if not ct else np.vstack((ideal_data, data))
            if done_flag:
                break
        #save simulated data
        np.save(self.filename, ideal_data)
        # Verify output matches what was previously seen by matlab
        pi_cal = cal.PiCalibration(self.q.label, numPulses, quad='imag')
        cal.calibrate([pi_cal])
        os.remove(self.filename)
        # NOTE: expected result is from the same input fed to the routine
        self.assertAlmostEqual(pi_cal.amplitude, amp, places=3)
        #restore original settings
        auspex.config.dump_meas_file(self.test_settings, cfg_file)

        def test_pi_phase_estimation_amp(self):
            """
            Test PiCalibration with phase estimation
            """

            numPulses = 9
            amp = self.test_settings['qubits'][self.q.label]['control']['pulse_params']['piAmp']
            direction = 'X'
            target = np.pi

            # NOTE: this function is a place holder to simulate an AWG generating
            # a sequence and a digitizer receiving the sequence.  This function
            # is passed into the optimize_amplitude routine to be able to update
            # the amplitude as part of the optimization loop.
            def update_data(amp, ct):
                    data, vardata =  simulate_phase_estimation(amp, target, numPulses)
                    phase, sigma = cal.phase_estimation(data, vardata, verbose=False)
                    amp, done_flag = cal.phase_to_amplitude(phase, sigma, amp, target, ct)
                    return amp, data, done_flag

            done_flag = 0
            for ct in range(5): #max iterations
                amp, data, done_flag = update_data(amp, ct)
                ideal_data = data if not ct else np.vstack((ideal_data, data))
                if done_flag:
                    break
            #save simulated data
            np.save(self.filename, ideal_data)
            # Verify output matches what was previously seen by matlab
            pi_cal = cal.PiCalibration(self.q.label, numPulses, quad='amp')
            cal.calibrate([pi_cal])
            os.remove(self.filename)
            # NOTE: expected result is from the same input fed to the routine
            self.assertAlmostEqual(pi_cal.amplitude, amp, places=3)
            #restore original settings
            auspex.config.dump_meas_file(self.test_settings, cfg_file)

        def test_pi_phase_estimation_real(self):
            """
            Test PiCalibration with phase estimation
            """

            numPulses = 9
            amp = self.test_settings['qubits'][self.q.label]['control']['pulse_params']['piAmp']
            direction = 'X'
            target = np.pi

            # NOTE: this function is a place holder to simulate an AWG generating
            # a sequence and a digitizer receiving the sequence.  This function
            # is passed into the optimize_amplitude routine to be able to update
            # the amplitude as part of the optimization loop.
            def update_data(amp, ct):
                    data, vardata =  simulate_phase_estimation(amp, target, numPulses)
                    phase, sigma = cal.phase_estimation(data, vardata, verbose=False)
                    amp, done_flag = cal.phase_to_amplitude(phase, sigma, amp, target, ct)
                    return amp, data, done_flag

            done_flag = 0
            for ct in range(5): #max iterations
                amp, data, done_flag = update_data(amp, ct)
                ideal_data = data if not ct else np.vstack((ideal_data, data))
                if done_flag:
                    break
            #save simulated data
            np.save(self.filename, ideal_data)
            # Verify output matches what was previously seen by matlab
            pi_cal = cal.PiCalibration(self.q.label, numPulses, quad='real')
            cal.calibrate([pi_cal])
            os.remove(self.filename)
            # NOTE: expected result is from the same input fed to the routine
            self.assertAlmostEqual(pi_cal.amplitude, amp, places=3)
            #restore original settings
            auspex.config.dump_meas_file(self.test_settings, cfg_file)

        def test_pi_phase_estimation_phase(self):
            """
            Test PiCalibration with phase estimation
            """

            numPulses = 9
            amp = self.test_settings['qubits'][self.q.label]['control']['pulse_params']['piAmp']
            direction = 'X'
            target = np.pi

            # NOTE: this function is a place holder to simulate an AWG generating
            # a sequence and a digitizer receiving the sequence.  This function
            # is passed into the optimize_amplitude routine to be able to update
            # the amplitude as part of the optimization loop.
            def update_data(amp, ct):
                    data, vardata =  simulate_phase_estimation(amp, target, numPulses)
                    phase, sigma = cal.phase_estimation(data, vardata, verbose=False)
                    amp, done_flag = cal.phase_to_amplitude(phase, sigma, amp, target, ct)
                    return amp, data, done_flag

            done_flag = 0
            for ct in range(5): #max iterations
                amp, data, done_flag = update_data(amp, ct)
                ideal_data = data if not ct else np.vstack((ideal_data, data))
                if done_flag:
                    break
            #save simulated data
            np.save(self.filename, ideal_data)
            # Verify output matches what was previously seen by matlab
            pi_cal = cal.PiCalibration(self.q.label, numPulses, quad='phase')
            cal.calibrate([pi_cal])
            os.remove(self.filename)
            # NOTE: expected result is from the same input fed to the routine
            self.assertAlmostEqual(pi_cal.amplitude, amp, places=3)
            #restore original settings
            auspex.config.dump_meas_file(self.test_settings, cfg_file)

if __name__ == '__main__':
    unittest.main()
