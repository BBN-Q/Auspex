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

from auspex.exp_factory import QubitExpFactory
from auspex.analysis.io import load_from_HDF5
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.analysis.fits import *
from auspex.analysis.helpers import normalize_data

from JSONLibraryUtils import LibraryCoders

def calibrate(calibrations):
    """Takes in a qubit (as a string) and list of calibrations (as instantiated classes).
    e.g. calibrate_pulses([RabiAmp("q1"), PhaseEstimation("q1")])"""
    for calibration in calibrations:
        if not isinstance(calibration, PulseCalibration):
            raise TypeError("calibrate_pulses was passed a calibration that is not actually a calibration.")
        calibration.calibrate()
        calibration.exp.plot_server.stop()

class PulseCalibration(object):
    """Base class for calibration of qubit control pulses."""
    def __init__(self, qubit_names):
        super(PulseCalibration, self).__init__()
        self.qubit_names = qubit_names if isinstance(qubit_names, list) else [qubit_names]
        self.qubit     = [QubitFactory(qubit_name) for qubit_name in qubit_names] if isinstance(qubit_names, list) else QubitFactory(qubit_names)
        self.filename   = 'None'
        self.exp        = None
        self.axis_descriptor = None
        self.plot       = self.init_plot()
        self.cw_mode    = False
        with open(config.channelLibFile, 'r') as FID:
            self.chan_settings = json.load(FID)
        with open(config.instrumentLibFile, 'r') as FID:
            self.instr_settings = json.load(FID)

    def sequence(self):
        """Returns the sequence for the given calibration, must be overridden"""
        return [[Id(self.qubit), MEAS(self.qubit)]]

    def set(self, instrs_to_set = []):
        meta_file = compile_to_hardware(self.sequence(), fileName=self.filename, axis_descriptor=self.axis_descriptor)
        self.exp = QubitExpFactory.create(meta_file=meta_file, calibration=True, cw_mode=self.cw_mode)
        if self.plot:
            # Add the manual plotter and the update method to the experiment
            self.exp.add_manual_plotter(self.plot)
        self.exp.connect_instruments()
        #set instruments for calibration
        for instr_to_set in instrs_to_set:
            par = FloatParameter()
            par.assign_method(getattr(self.exp._instruments[instr_to_set['instr']], instr_to_set['method']))
            # Either sweep or set single value
            if 'sweep_values' in instr_to_set.keys():
                par.value = instr_to_set['sweep_values'][0]
                self.exp.add_sweep(par, instr_to_set['sweep_values'])
            else:
                par.value = instr_to_set['value']
                par.push()

    def run(self):
        self.exp.leave_plot_server_open = True
        self.exp.run_sweeps()
        data = {}
        var = {}
        writers = [self.exp.qubit_to_writer[qn] for qn in self.qubit_names]
        data_buffers = [b for b in self.exp.buffers if b.name in writers]

        for buff in self.exp.buffers:
            if self.exp.writer_to_qubit[buff.name] in self.qubit_names:
                dataset, descriptor = buff.get_data(), buff.get_descriptor()
                data[self.exp.writer_to_qubit[buff.name]] = np.real(dataset['Data'])
                if 'Variance' in dataset.dtype.names:
                    var[self.exp.writer_to_qubit[buff.name]] = dataset['Variance']/descriptor.metadata["num_averages"]
                else:
                    var[self.exp.writer_to_qubit[buff.name]] = None

        # Return data and variance of the mean
        if len(data) == 1:
            # if single qubit, get rid of dictionary
            data = list(data.values())[0]
            var = list(var.values())[0]
        return data, var

    def init_plot(self):
        """Return a ManualPlotter object so we can plot calibrations. All
        plot lines, glyphs, etc. must be declared up front!"""
        return None

    def calibrate(self):
        """Runs the actual calibration routine, must be overridden.
        This function is responsible for calling self.update_plot()"""
        pass

    def update_libraries(self, libraries, filenames):
        """Update calibrated json libraries"""
        for library, filename in zip(libraries, filenames):
            with open(filename, 'w') as FID:
                json.dump(library, FID, cls=LibraryCoders.LibraryEncoder, indent=2, sort_keys=True)

class CavitySearch(PulseCalibration):
    def __init__(self, qubit_name, frequencies=np.linspace(4, 5, 100)):
        super(CavitySearch, self).__init__(qubit_name)
        self.frequencies = frequencies
        self.cw_mode = True

    def sequence(self):
        return [[Id(self.qubit), MEAS(self.qubit)]]

    def calibrate(self):
        #find cavity source from config
        cavity_source = self.chan_settings['channelDict'][self.chan_settings['channelDict']['M-'+self.qubit_names[0]]['physChan']]['generator']
        orig_freq = self.instr_settings['instrDict'][cavity_source]['frequency']
        instr_to_set = {'instr': cavity_source, 'method': 'set_frequency', 'sweep_values': self.frequencies}
        self.set([instr_to_set])
        data, _ = self.run()

        # Plot the results
        self.plot["Data"] = (self.frequencies, data)

    def init_plot(self):
        plot = ManualPlotter("Qubit Search", x_label='Frequency (GHz)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data")
        plot.add_fit_trace("Fit")
        return plot

class QubitSearch(PulseCalibration):
    def __init__(self, qubit_name, frequencies=np.linspace(4, 5, 100)):
        super(QubitSearch, self).__init__(qubit_name)
        self.frequencies = frequencies
        self.cw_mode = True

    def sequence(self):
        return [[X(self.qubit), MEAS(self.qubit)]]

    def calibrate(self):
        #find qubit control source from config
        qubit_source = self.chan_settings['channelDict'][self.chan_settings['channelDict'][self.qubit_names[0]]['physChan']]['generator']
        orig_freq = self.instr_settings['instrDict'][qubit_source]['frequency']
        instr_to_set = {'instr': qubit_source, 'method': 'set_frequency', 'sweep_values': self.frequencies}
        self.set([instr_to_set])
        data, _ = self.run()

        # Plot the results
        self.plot["Data"] = (self.frequencies, data)

    def init_plot(self):
        plot = ManualPlotter("Qubit Search", x_label='Frequency (GHz)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data")
        plot.add_fit_trace("Fit")
        return plot

class RabiAmpCalibration(PulseCalibration):
    def __init__(self, qubit_name, amps=np.linspace(0.0, 1.0, 51)):
        super(RabiAmpCalibration, self).__init__(qubit_name)
        self.amps = amps

    def sequence(self):
        return [[Xtheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps]

class RamseyCalibration(PulseCalibration):
    def __init__(self, qubit_name, delays=np.linspace(0.0, 50.0, 51)*1e-6, two_freqs = False, added_detuning = 150e3, set_source = True):
        super(RamseyCalibration, self).__init__(qubit_name)
        self.filename = 'Ramsey/Ramsey'
        self.delays = delays
        self.two_freqs = two_freqs
        self.added_detuning = added_detuning
        self.set_source = set_source
        self.axis_descriptor = [time_descriptor(self.delays)]

    def sequence(self):
        return [[X90(self.qubit), Id(self.qubit, delay), X90(self.qubit), MEAS(self.qubit)] for delay in self.delays]

    def init_plot(self):
        plot = ManualPlotter("Ramsey Fit", x_label='Time (us)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data")
        plot.add_fit_trace("Fit")
        return plot

    def calibrate(self):
        #find qubit control source (from config)
        qubit_source = self.chan_settings['channelDict'][self.chan_settings['channelDict'][self.qubit_names[0]]['physChan']]['generator']
        orig_freq = self.instr_settings['instrDict'][qubit_source]['frequency']
        set_freq = round(orig_freq + self.added_detuning/1e9, 10)
        instr_to_set = {'instr': qubit_source, 'method': 'set_frequency', 'value': set_freq}
        self.set([instr_to_set])
        data, _ = self.run()

        #TODO: fit Ramsey and find new detuning. Finally, set source or qubit channel frequency
        fit_freqs, fit_errs, all_params = fit_ramsey(self.delays, data, two_freqs = self.two_freqs)
        fit_freq_A = np.mean(fit_freqs) #the fit result can be one or two frequencies
        #TODO: set conditions for success
        set_freq = round(orig_freq + self.added_detuning/1e9 + fit_freq_A/2/1e9, 10)
        instr_to_set['value'] = set_freq
        self.set([instr_to_set])

        # Plot the results
        ramsey_f = ramsey_2f if self.two_freqs else ramsey_1f
        finer_delays = np.linspace(np.min(self.delays), np.max(self.delays), 4*len(self.delays))
        self.plot["Data"] = (self.delays, data)
        self.plot["Fit"] = (finer_delays, ramsey_f(finer_delays, *all_params))

        data, _ = self.run()

        fit_freqs, fit_errs, all_params = fit_ramsey(self.delays, data, two_freqs = self.two_freqs)
        fit_freq_B = np.mean(fit_freqs)

        # Plot the results
        ramsey_f = ramsey_2f if self.two_freqs else ramsey_1f
        finer_delays = np.linspace(np.min(self.delays), np.max(self.delays), 4*len(self.delays))
        self.plot["Data"] = (self.delays, data)
        self.plot["Fit"]  = (finer_delays, ramsey_f(finer_delays, *all_params))

        if fit_freq_B < fit_freq_A:
            fit_freq = round(orig_freq + self.added_detuning/1e9 + 0.5*(fit_freq_A + 0.5*fit_freq_A + fit_freq_B)/1e9, 10)
        else:
            fit_freq = round(orig_freq + self.added_detuning/1e9 - 0.5*(fit_freq_A - 0.5*fit_freq_A + fit_freq_B)/1e9, 10)
        if self.set_source:
            self.instr_settings['instrDict'][qubit_source]['frequency'] = fit_freq
            self.update_libraries([self.instr_settings], [config.instrumentLibFile])
        else:
            self.chan_settings['channelDict'][self.qubit_names[0]]['frequency'] += (fit_freq - orig_freq)*1e9
            self.update_libraries([self.chan_settings], [config.channelLibFile])

        print('Frequency', fit_freq)
        return fit_freq

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
        # Determine whether it is a single- or a two-qubit pulse calibration
        if isinstance(self.qubits, list):
            cal_pulse = ZX90_CR(*self.qubits, amp=self.amplitude)
            qubit = self.qubits[1] # qt
        else:
            cal_pulse = [Xtheta(self.qubit, amp=self.amplitude)]
            qubit = self.qubit
        # Exponentially growing repetitions of the target pulse, e.g.
        # (1, 2, 4, 8, 16, 32, 64, 128, ...) x X90
        seqs = [cal_pulse*n for n in 2**np.arange(self.num_pulses+1)]
        # measure each along Z or Y
        seqs = [s + m for s in seqs for m in [ [MEAS(qubit)], [X90m(qubit), MEAS(qubit)] ]]
        # tack on calibrations to the beginning
        seqs = [[Id(qubit), MEAS(qubit)], [X(qubit), MEAS(qubit)]] + seqs
        # repeat each
        return [copy(s) for s in seqs for _ in range(2)]

    def calibrate(self):
        """Attempts to optimize the pulse amplitude for a pi/2 or pi pulse about X or Y. """

        ct = 1
        amp = self.amplitude
        set_amp = 'pi2Amp' if isinstance(self, Pi2Calibration) else 'piAmp' if isinstance(self, PiCalibration) else 'amp'
        #TODO: add writers for variance if not existing
        while True:
            self.set()
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
            #update amplitude
            self.amplitude = amp
        print('Amp',amp)

        set_chan = self.qubit_names[0] if len(self.qubit_names) == 1 else ChannelLibrary.EdgeFactory(*self.qubits).label
        self.chan_settings['channelDict'][set_chan]['pulseParams'][set_amp] = round(amp, 5)
        self.update_libraries([self.chan_settings], [config.channelLibFile])
        return amp

class Pi2Calibration(PhaseEstimation):
    def __init__(self, qubit_name, num_pulses= 9):
        super(Pi2Calibration, self).__init__(qubit_name, num_pulses = num_pulses)
        self.amplitude = self.qubit.pulseParams['pi2Amp']
        self.target    = np.pi/2.0

class PiCalibration(PhaseEstimation):
    def __init__(self, qubit_name, num_pulses= 9):
        super(PiCalibration, self).__init__(qubit_name, num_pulses = num_pulses)
        self.amplitude = self.qubit.pulseParams['piAmp']
        self.target    = np.pi

class CRAmpCalibration_PhEst(PhaseEstimation):
    def __init__(self, qubit_names, num_pulses= 9):
        super(CRAmpCalibration_PhEst, self).__init__(qubit_names, num_pulses = num_pulses)
        CRchan = ChannelLibrary.EdgeFactory(*self.qubits)
        self.amplitude = CRchan.pulseParams['amp']
        self.target    = np.pi/2

class DRAGCalibration(PulseCalibration):
    def __init__(self, qubit_name, deltas = np.linspace(-1,1,11), num_pulses = np.arange(16, 64, 4)):
        super(DRAGCalibration, self).__init__(qubit_name)
        self.filename = 'DRAG/DRAG'
        self.deltas = deltas
        self.num_pulses = num_pulses

    def sequence(self):
        seqs = []
        for n in self.num_pulses:
            seqs += [[X90(q, dragScaling = d), X90m(q, dragScaling = d)]*n + [X90(q, dragScaling = d), MEAS(q)] for d in self.deltas]
        seqs += create_cal_seqs((q,),2)
        return seqs

    def calibrate(self):
        #generate sequence
        self.set()
        #first run
        data, _ = self.run()
        #fit and analyze
        opt_drag, error_drag = fit_drag(self.deltas, self.num_pulses, norm_data)

        #generate sequence with new pulses and drag parameters
        new_drag_step = 0.25*(max(self.deltas) - min(self.deltas))
        self.deltas = np.range(opt_drag - new_drag_step, opt_drag + new_drag_step, len(self.deltas))
        new_pulse_step = 2*(max(self.num_pulses)-min(self.num_pulses))/len(self.num_pulses)
        self.num_pulses = np.arange(max(self.num_pulses) - new_pulse_step, max(self.num_pulses) + new_pulse_step*(len(self.num_pulses)-1), new_pulse_step)
        self.set()

        #second run, finer range
        data, _ = self.run()
        opt_drag, error_drag = fit_drag(data)
        #TODO: success condition

        print("DRAG", opt_drag)

        self.chan_settings['channelDict'][self.qubit_names[0]]['pulseParams']['dragScaling'] = fitted_drag
        self.update_libraries([self.chan_settings], [config.channelLibFile])

        return fitted_drag

'''Two-qubit gate calibrations'''
class CRCalibration(PulseCalibration):
    def __init__(self, qubit_names, lengths=np.linspace(20, 1020, 21)*1e-9, phase = 0, amp = 0.8, rise_fall = 40e-9):
        super(CRCalibration, self).__init__(qubit_names)
        self.lengths = lengths
        self.phases = phase
        self.amps = amp
        self.rise_fall = rise_fall
        self.filename = 'CR/CR'

    def init_plot(self):
        plot = ManualPlotter("CR"+str.lower(self.cal_type.name)+"Fit", x_label=str.lower(self.cal_type.name), y_label='$<Z_{'+self.qubit_names[1]+'}>$')
        plot.add_data_trace("Data 0")
        plot.add_fit_trace("Fit 0")
        plot.add_data_trace("Data 1")
        plot.add_fit_trace("Fit 1")
        return plot

    def calibrate(self):
        #generate sequence
        self.set()
        #run
        data, _ = self.run()
        data_t = data[qt]
        opt_par, all_params_0, all_params_1 = fit_CR(self.lengths, data_t, self.cal_type)

        #update CR channel
        CRchan = ChannelLibrary.EdgeFactory(*self.qubits)
        self.chan_settings['channelDict'][CRchan][str.lower(self.cal_type.name)] = opt_par
        self.update_libraries([self.chan_settings], [config.channelLibFile])

        # Plot the results
        xaxis = self.lengths if self.cal_type==CR_cal_type.LENGTH else self.phases if self.cal_type==CR_cal_type.PHASE else self.amps
        finer_xaxis = np.linspace(np.min(xaxis), np.max(xaxis), 4*len(xaxis))
        self.plot["Data 0"] = (xaxis,       data_t[:len(data_t)/2])
        self.plot["Fit 0"] =  (finer_xaxis, sin_f(finer_lengths, *all_params_0))
        self.plot["Data 1"] = (xaxis,       data_t[len(data_t)/2:])
        self.plot["Fit 1"] =  (finer_xaxis, sin_f(finer_lengths, *all_params_1))

class CRLenCalibration(CRCalibration):
    def __init__(self, qubit_names, lengths=np.linspace(20, 1020, 21)*1e-9, phase = 0, amp = 0.8, rise_fall = 40e-9, cal_type = CR_cal_type.LENGTH):
        super(CRLenCalibration, self).__init__(qubit_names, lengths, phases, amps, rise_fall)
        self.cal_type = cal_type

    def sequence(self):
        qc, qt = self.qubits[:]
        seqs = [[Id(qc)] + echoCR(qc, qt, length=l, phase = self.phases, amp=self.amps, riseFall=self.rise_fall) + [Id(qc), MEAS(qt)*MEAS(qc)]
        for l in self.lengths]+ [[X(qc)] + echoCR(qc, qt, length=l, phase= self.phases, amp=self.amps, riseFall=self.rise_fall) + [X(qc), MEAS(qt)*MEAS(qc)]
        for l in self.lengths] + create_cal_seqs((qt,qc), calRepeats=2, measChans=(qt,qc))

        self.axis_descriptor=[
            time_descriptor(np.concatenate((lengths, lengths))),
            cal_descriptor((qc, qt), 2)
        ]

        return seqs

class CRPhaseCalibration(PulseCalibration):
    def __init__(self, qubit_names, phases = np.linspace(0,2*np.pi,21), amp = 0.8, rise_fall = 40e-9, cal_type = CR_cal_type.PHASE):
        super(CRPhaseCalibration, self).__init__(qubit_names, lengths, phases, amps, rise_fall)
        self.phases = phases
        self.amps = amp
        self.rise_fall = rise_fall
        CRchan = ChannelLibrary.EdgeFactory(*self.qubits)
        length = CRchan.pulseParams['length']

    def sequence(self):
        qc, qt = self.qubits[:]
        seqs = [[Id(qc)] + echoCR(qc, qt, length=length, phase=ph, amp=self.amp, riseFall=self.rise_fall) + [X90(qt)*Id(qc), MEAS(qt)*MEAS(qc)]
        for ph in self.phases]+ [[X(qc)] + echoCR(qc, qt, length=length, phase= ph, amp=self.amp, riseFall=self.rise_fall) + [X90(qt)*X(qc), MEAS(qt)*MEAS(qc)]
        for ph in self.phases] + create_cal_seqs((qt,qc), calRepeats=2, measChans=(qt,qc))

        self.axis_descriptor = [
            {
                'name': 'phase',
                'unit': 'radians',
                'points': list(phases)+list(phases),
                'partition': 1
            },
            cal_descriptor((qc, qt), calRepeats)
        ]

        return seqs

class CRAmpCalibration(PulseCalibration):
    def __init__(self, qubit_names, range = 0.2, amp = 0.8, rise_fall = 40e-9, num_CR = 1, cal_type = CR_cal_type.AMPLITUDE):
        super(CRAmpCalibration, self).__init__(qubit_names, lengths, phases, amps, rise_fall)
        if mod(num_CR, 2) == 0:
            logger.error('The number of ZX90 must be odd')
        self.rise_fall = rise_fall
        amp = CRchan.pulseParams['amp']
        self.amps = np.linspace(0.8*amp, 1.2*amp, 21)
        self.lengths = CRchan.pulseParams['length']
        self.phases = CRchan.pulseParams['phase']

    def sequence(self):
        qc, qt = self.qubits[:]
        CRchan = ChannelLibrary.EdgeFactory(qc, qt)
        seqs = [[Id(qc)] + num_CR*echoCR(qc, qt, length=self.length, phase=self.phase, amp=a, riseFall=self.rise_fall) + [Id(qc), MEAS(qt)*MEAS(qc)]
        for a in self.amps]+ [[X(qc)] + num_CR*echoCR(qc, qt, length=length, phase= self.phase, amp=a, riseFall=self.rise_fall) + [X(qc), MEAS(qt)*MEAS(qc)]
        for a in self.amps] + create_cal_seqs((qt,qc), calRepeats=2, measChans=(qt,qc))

        self.axis_descriptor = [
            {
                'name': 'amplitude',
                'unit': None,
                'points': list(amps)+list(amps),
                'partition': 1
            },
            cal_descriptor((qc, qt), calRepeats)
        ]

        return seqs

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
