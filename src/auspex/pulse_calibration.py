# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from QGL import *
from QGL import config as QGLconfig
import auspex.config as config
from copy import copy
import os

from auspex.exp_factory import QubitExpFactory
from auspex.analysis.io import load_from_HDF5

def calibrate(qubit, calibrations):
    """Takes in a qubit (as a string) and list of calibrations (as instantiated classes).
    e.g. calibrate_pulses("q1", [RabiAmp, PhaseEstimation])"""
    for calibration in calibrations:
        if not isinstance(calibration, PulseCalibration):
            raise TypeError("calibrate_pulses was passed a calibration that is not actually a calibration.")
        calibration.calibrate()

class PulseCalibration(object):
    """Base class for calibration of qubit control pulses."""
    def __init__(self, qubit_name):
        super(PulseCalibration, self).__init__()
        self.qubit_name = qubit
        self.qubit      = QubitFactory(qubit)
        self.filename   = 'None'
    
    def sequence(self):
        """Returns the sequence for the given calibration, must be overridden"""
        return [[Id(self.qubit), MEAS(self.qubit)]]

    def run(self):
        seq_files = compile_to_hardware(self.sequence(), fileName=self.filename)
        metafileName = os.path.join(QGLconfig.AWGDir, self.filename + '-meta.json')
        exp = QubitExpFactory.create(meta_file=metafileName)
        exp.run_sweeps()
        # TODO: there should be no need for saving the calibration data
        wrs = [w for w in exp.writers if w.name == exp.qubit_to_writer[self.qubit_name]]
        filename = wrs[0].filename.value
        groupname = wrs[0].groupname.value

        dataset, descriptor = load_from_HDF5(filename, groupname=groupname)
        # TODO: get the name of the relevant data from the graph
        data, var = dataset['data']['M1'], dataset['data']['M1-var']

        # Return data and variance of the mean
        return data, var/descriptor.metadata["num_averages"]

    def calibrate(self):
        """Runs the actual calibration routine, must be overridden"""
        pass

class RabiAmpCalibration(PulseCalibration):
    def __init__(self, qubit_name, amps=np.linspace(0.0, 1.0, 51)):
        super(RabiAmpCalibration, self).__init__(qubit_name)
        self.amps = amps

    def sequence(self):
        return [[Xtheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps]

class RamseyCalibration(PulseCalibration):
    def __init__(self, qubit_name, delays=np.linspace(0.0, 1.0, 51)):
        super(RamseyCalibration, self).__init__(qubit_name)
        self.delays = delays

    def sequence(self):
        return [[X90(self.qubit), Id(self.qubit, delay), MEAS(self.qubit)] for delay in self.delays]


class PhaseEstimation(PulseCalibration):
    """Estimates pulse rotation angle from a sequence of P^k experiments, where
    k is of the form 2^n. Uses the modified phase estimation algorithm from
    Kimmel et al, quant-ph/1502.02677 (2015). Every experiment i doubled.
    vardata should be the variance of the mean"""

    def __init__(self, qubit_name, num_pulses= 1, amplitude= 0.1, direction = 'X'):
        """Phase estimation calibration. Direction is either 'X' or 'Y',
        num_pulses is log2(n) of the longest sequence n,
        and amplitude is self-exaplanatory."""

        super(PhaseEstimation, self).__init__(qubit_name)
        self.filename        = 'RepeatCal/RepeatCal'
        self.direction       = direction
        self.amplitude       = amplitude
        self.num_pulses      = num_pulses
        self.target          = np.pi/2.0
        self.iteration_limit = 5

    def sequence(self):
        # Exponentially growing repetitions of the target pulse, e.g.
        # (1, 2, 4, 8, 16, 32, 64, 128, ...) x X90
        seqs = [[Xtheta(self.qubit, amp=self.amplitude)]*n for n in 2**np.arange(self.num_pulses+1)]
        # measure each along Z or Y
        seqs = [s + m for s in seqs for m in [ [MEAS(self.qubit)], [X90m(self.qubit), MEAS(self.qubit)] ]]
        # tack on calibrations to the beginning
        seqs = [[Id(self.qubit), MEAS(self.qubit)], [X(self.qubit), MEAS(self.qubit)]] + seqs
        # repeat each
        return [copy(s) for s in seqs for _ in range(2)]

    def calibrate(self):
        """Attempts to optimize the pulse amplitude for a pi/2 or pi pulse about X or Y. """

        ct = 1
        amp = self.amplitude
        while True:
            [phase, sigma] = phase_estimation(*self.run())
            print("Phase: %.4f Sigma: %.4f"%(phase,sigma))
            # correct for some errors related to 2pi uncertainties
            if np.sign(phase) != np.sign(amp):
                phase += np.sign(amp)*2*np.pi
            angle_error = phase - self.target;
            print('Angle error: %.4f'%angle_error);

            amp_target = self.target/phase * amp
            amp_error = amp - amp_target
            print('Amplitude error: %.4f\n'%amp_error)

            amp = amp_target
            ct += 1

            # check for stopping condition
            phase_error = phase - self.target
            if np.abs(phase_error) < 1e-2 or np.abs(phase_error/sigma) < 1 or ct > self.iteration_limit:
                if np.abs(phase_error) < 1e-2:
                    print('Reached target rotation angle accuracy');
                elif abs(phase_error/sigma) < 1:
                    print('Reached phase uncertainty limit');
                else:
                    print('Hit max iteration count');
                break
        print('Amp',amp)
        
        return amp

class Pi2Calibration(PhaseEstimation):
    def __init__(self, qubit_name, num_pulses= 9):
        super(Pi2Calibration, self).__init__(qubit_name)
        self.amplitude = self.qubit.pulseParams['pi2Amp']
        self.target    = np.pi/2.0

class PiCalibration(PulseCalibration):
    def __init__(self, qubit_name, num_pulses= 9):
        super(PiCalibration, self).__init__(qubit_name)
        self.amplitude = self.qubit.pulseParams['piAmp']
        self.target    = np.pi

class DRAGCalibration(PulseCalibration):
    pass

def restrict(phase):
    out = np.mod( phase + np.pi, 2*np.pi, ) - np.pi
    return out

def phase_estimation( data_in, vardata_in, verbose=False):
    """Estimates pulse rotation angle from a sequence of P^k experiments, where
    k is of the form 2^n. Uses the modified phase estimation algorithm from
    Kimmel et al, quant-ph/1502.02677 (2015). Every experiment i doubled.
    vardata should be the variance of the mean"""     

    #average together pairs of data points
    avgdata = (data_in[0::2] + data_in[1::2])/2
    
    # normalize data using the first two pulses to calibrate the "meter"
    data = 1 + 2*(avgdata[2:] - avgdata[0]) / (avgdata[0] - avgdata[1])
    zdata = data[0::2]
    xdata = data[1::2]

    # similar scaling with variances
    vardata = (vardata_in[0::2] + vardata_in[1::2])/2
    vardata = vardata[2:] * 2 / abs(avgdata[0] - avgdata[1])**2
    zvar = vardata[0::2]
    xvar = vardata[1::2]

    phases = np.arctan2(xdata, zdata)
    distances = np.sqrt(xdata**2 + zdata**2)

    curGuess = phases[0]
    phase = curGuess
    sigma = np.pi
    
    if verbose == True:
        print('Current Guess: %f'%(curGuess))
        
    for k in range(1,len(phases)):
        
        if verbose == True:
            print('k: %d'%(k))
            
        # Each step of phase estimation needs to assign the measured phase to
        # the correct half circle. We will conservatively require that the
        # (x,z) tuple is long enough that we can assign it to the correct
        # quadrant of the circle with 2σ confidence
        
        if distances[k] < 2*np.sqrt(xvar[k] + zvar[k]):
            print('Phase estimation terminated at %dth pulse because the (x,z) vector is too short'%(k))
            break
        
        lowerBound = restrict(curGuess - np.pi/2**(k))
        upperBound = restrict(curGuess + np.pi/2**(k))
        possiblesTest = [ restrict((phases[k] + 2*n*np.pi)/2**(k)) for n in range(0,2**(k)+1)]
        
        if verbose == True:
            print('Lower Bound: %f'%lowerBound)
            print('Upper Bound: %f'%upperBound)
        
        possibles=[]
        for p in possiblesTest:
            # NOTE: previous code did not handle upperbound == lowerBound
            if lowerBound >= upperBound:
                satisfiesLB = p > lowerBound or p < 0.
                satisfiesUP = p < upperBound or p > 0.
            else:
                satisfiesLB = p > lowerBound 
                satisfiesUP = p < upperBound 
            
            if satisfiesLB == True and satisfiesUP == True:
                possibles.append(p)
                    
        curGuess = possibles[0]
        if verbose == True:
            print('Current Guess: %f'%(curGuess))
    phase = curGuess
    sigma = np.maximum(np.abs(restrict(curGuess - lowerBound)), np.abs(restrict(curGuess - upperBound)))
    
    return phase, sigma

