# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import os
import time
import numpy as np
import tempfile

import QGL.config
import auspex.config
auspex.config.auspex_dummy_mode = True

# Set temporary output directories
awg_dir = tempfile.TemporaryDirectory()
kern_dir = tempfile.TemporaryDirectory()
auspex.config.AWGDir = QGL.config.AWGDir = awg_dir.name
auspex.config.KernelDir = kern_dir.name

from QGL import *
from auspex.qubit import *

cl = ChannelLibrary(db_resource_name=":memory:")
pl = PipelineManager()
pl.create_default_pipeline()

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

def simulate_phase_estimation(amp, target, numPulses, ideal_amp=0.34, add_noise=False):
    """
    Simulate the output of a PhaseEstimation experiment with NumPulses.
    amp: initial pulse amplitude
    target: target angle (pi/2, etc.)

    returns: ideal data and variance
    """
    ideal_amp    = ideal_amp
    noiseScale   = 0.05
    polarization = 0.99 # residual polarization after each pulse

    # data representing over/under rotation of pi/2 pulse
    # theta = pi/2 * (amp/ideal_amp);
    theta   = target * (amp/ideal_amp)
    ks      = [ 2**k for k in range(0,numPulses+1)]

    xdata = [ polarization**x * np.sin(x*theta) for x in ks]
    xdata = np.insert(xdata,0,-1.0)
    zdata = [ polarization**x * np.cos(x*theta) for x in ks]
    zdata = np.insert(zdata,0,1.0)
    data  = np.array([zdata,xdata]).flatten('F')
    data  = np.tile(data,(2,1)).flatten('F')

    if add_noise:
        data += noiseScale * np.random.randn(len(data));

    vardata = noiseScale**2 * np.ones((len(data)))

    return data, vardata

def simulate_drag(deltas = np.linspace(-1,1,21), num_pulses = np.arange(16, 48, 4), drag = 0.6):
    """
    Simulate the output of a DRAG experiment with a set drag value

    returns: ideal data
    """
    ypoints = [t for s in [(n/2)**2*(deltas - drag)**2 for n in num_pulses] for t in s]
    ypoints = np.append(ypoints, np.repeat([max(ypoints),min(ypoints)],2))
    return ypoints

class SingleQubitCalTestCase(unittest.TestCase):
    """
    Class for unittests of single-qubit calibrations. Tested so far with a dummy X6 digitizer:
    * RabiAmpCalibration
    * RamseyCalibration
    Ideal data are generated and stored into a temporary file, whose name is set by the X6 property `ideal_data`. Calibrations which span over multiple experiments load different columns of these ideal data. The column (and experiment) number is set by an incremental counter, also a digitizer property `exp_step`. Artificial noise is added by the X6 dummy instrument.
    """
    
    def _setUp(self):
        cl.clear()
        self.q     = cl.new_qubit("q1")
        self.aps1  = cl.new_APS2("BBNAPS1", address="192.168.5.102")
        self.aps2  = cl.new_APS2("BBNAPS2", address="192.168.5.103")
        self.x6_1  = cl.new_X6("X6_1", address="1", record_length=512)
        self.holz1 = cl.new_source("Holz_1", "HolzworthHS9000", "HS9004A-009-1", power=-30)
        self.holz2 = cl.new_source("Holz_2", "HolzworthHS9000", "HS9004A-009-2", power=-30)
        cl.set_control(self.q, self.aps1, generator=self.holz1)
        cl.set_measure(self.q, self.aps2, self.x6_1.ch("1"), generator=self.holz2)
        cl.set_master(self.aps1, self.aps1.ch("m2"))
        self.num_averages = 50
        pl.create_default_pipeline()

    @unittest.skip("Fix me for updated MP/DB api")
    def test_rabi_amp(self):
        self._setUp()
        """
        Test RabiAmpCalibration. Ideal data generated by simulate_rabiAmp.
        """
        pce = RabiAmpCalibration(self.q, num_steps=20)
        pce.set_fake_data(self.x6_1, simulate_rabiAmp(num_steps=20))
        pce.run_calibration()

        self.assertAlmostEqual(pce.pi_amp,1,places=2)
        self.assertAlmostEqual(pce.pi2_amp,0.5,places=2)
        self.assertAlmostEqual(pce.pi_amp, self.q.pulse_params['piAmp'], places=3)
        self.assertAlmostEqual(pce.pi2_amp, self.q.pulse_params['pi2Amp'], places=3)

    def run_ramsey(self, set_source = True):
        """
        Simulate a RamseyCalibration run. Ideal data are generated by simulate_ramsey.
        set_source: True (False) sets the source (qubit) frequency.
        """
        ideal_data = [simulate_ramsey(num_steps = 50, detuning = 90e3),  
                      simulate_ramsey(num_steps = 50, detuning = 45e3)]
        ramsey_cal = RamseyCalibration(self.q, added_detuning = 0e3, 
                        delays=np.linspace(0.0, 50.0, 50)*1e-6, set_source = set_source)
        ramsey_cal.set_fake_data(self.x6_1, ideal_data)
        ramsey_cal.run_calibration()
        return ramsey_cal

    @unittest.skip("Issues with Linux build.")
    def test_ramsey_set_source(self):
        self._setUp()
        """
        Test RamseyCalibration with source frequency setting.
        """
        ramsey_cal = self.run_ramsey()
        self.assertAlmostEqual(ramsey_cal.fit_freq/1e9, (self.test_settings['instruments']['Holz2']['frequency'] + 90e3)/1e9, places=3)
        #test update_settings
        # new_settings = auspex.config.load_meas_file(cfg_file)
        self.assertAlmostEqual(ramsey_cal.fit_freq/1e9, new_settings['instruments']['Holz2']['frequency']/1e9, places=3)
        #restore original settings
        # auspex.config.dump_meas_file(self.test_settings, cfg_file)

    @unittest.skip("FIX ME for qubit graph fiasco")
    def test_ramsey_set_qubit(self):
        self._setUp()
        """
        Test RamseyCalibration with qubit frequency setting.
        """
        ramsey_cal = self.run_ramsey(False)
        #test update_settings
        # new_settings = auspex.config.load_meas_file(cfg_file)

        print(float(round(ramsey_cal.fit_freq - ramsey_cal.orig_freq)))
        # self.assertTrue( 0.85 < ((self.q.frequency+90e3)/1e6)/(new_settings['qubits'][self.q.label].frequency/1e6) < 1.15)
        #restore original settings
        # auspex.config.dump_meas_file(self.test_settings, cfg_file)

    @unittest.skip("FIX ME for qubit graph fiasco")
    def test_phase_estimation(self):
        self._setUp()
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
        self.assertAlmostEqual(phase,-1.2012,places=3)
        self.assertAlmostEqual(sigma,0.0245,places=3)

    @unittest.skip("There seems to be an issue with this test on linux. Fix me.")
    def test_pi_phase_estimation(self):
        self._setUp()
        """
        Test PiCalibration with phase estimation
        """

        numPulses = 9
        amp = self.q['control']['pulse_params']['piAmp']
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
        for ct in range(15): #max iterations
            amp, data, done_flag = update_data(amp, ct)
            ideal_data = data if not ct else np.vstack((ideal_data, data))
            if done_flag:
                break
        #save simulated data
        np.save(self.filename, ideal_data)
        # Test for one of the quadrature or amp/phase randomly
        quad = np.random.choice(['real', 'imag', 'amp', 'phase'])
        # Verify output matches what was previously seen by matlab
        pi_cal = cal.PiCalibration(self.q.label, self.ef, numPulses, quad=quad)
        cal.calibrate([pi_cal], leave_plots_open = False)
        # NOTE: expected result is from the same input fed to the routine
        self.assertAlmostEqual(pi_cal.amplitude, amp, places=2)
        #restore original settings
        # auspex.config.dump_meas_file(self.test_settings, cfg_file)

    @unittest.skip("FIX ME for qubit graph fiasco")
    def test_drag(self):
        self._setUp()
        """
        Test DRAGCalibration. Ideal data generated by simulate_drag.
        """
        ideal_drag = 0.0 # arbitrary choice for testing
        deltas_0 = np.linspace(-0.3,0.3,21)
        pulses_0 = np.arange(4, 20, 4)
        drag_step_1 = 0.25*(max(deltas_0) - min(deltas_0))
        deltas_1 = np.linspace(ideal_drag - drag_step_1, ideal_drag + drag_step_1, len(deltas_0))
        pulse_step_1 = 2*(max(pulses_0) - min(pulses_0))/len(pulses_0)
        pulses_1 = np.arange(max(pulses_0) - pulse_step_1, max(pulses_0) + pulse_step_1*(len(pulses_0)-1))

        ideal_data = [np.tile(simulate_drag(deltas_0, pulses_0, ideal_drag), self.num_averages), np.tile(simulate_drag(deltas_1, pulses_1, ideal_drag), self.num_averages)]
        np.save(self.filename, ideal_data)
        drag_cal = cal.DRAGCalibration(self.q.label, self.ef, deltas = deltas_0, num_pulses = pulses_0)
        cal.calibrate([drag_cal], leave_plots_open = False)

        self.assertAlmostEqual(drag_cal.drag, ideal_drag, places=2)
        #test update_settings
        # new_settings = auspex.config.load_meas_file(cfg_file)
        self.assertAlmostEqual(drag_cal.drag, new_settings['qubits'][self.q.label]['control']['pulse_params']['drag_scaling'],places=2)
        #restore original settings
        # auspex.config.dump_meas_file(self.test_settings, cfg_file)

if __name__ == '__main__':
    unittest.main()
