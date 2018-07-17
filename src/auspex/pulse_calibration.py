# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

try:
    from QGL import *
    from QGL import config as QGLconfig
    from QGL.BasicSequences.helpers import create_cal_seqs, delay_descriptor, cal_descriptor
except:
    print("Could not find QGL")

import auspex.config as config
from auspex.log import logger
from copy import copy, deepcopy
import os
import pandas as pd

from time import *

from auspex.exp_factory import QubitExpFactory
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.analysis.fits import *
from auspex.analysis.helpers import normalize_data
from matplotlib import cm
import numpy as np
from itertools import product

def calibrate(calibrations, update_settings=True, cal_log=True):
    """Takes in a qubit (as a string) and list of calibrations (as instantiated classes).
    e.g. calibrate_pulses([RabiAmp("q1"), PhaseEstimation("q1")])"""
    for calibration in calibrations:
        if not isinstance(calibration, PulseCalibration):
            raise TypeError("calibrate_pulses was passed a calibration that is not actually a calibration.")
        try:
            cal_result = calibration.calibrate()
            if update_settings:
                calibration.update_settings()
            if cal_log:
                calibration.write_to_log(cal_result)
        except Exception as ex:
            logger.warning('Calibration {} could not complete: got exception: {}.'.format(type(calibration).__name__, ex))
            try:
                calibration.exp.shutdown()
            except:
                pass #Experiment not yet created, so ignore
            raise Exception("Calibration failure") from ex
        finally:
            sleep(0.1) #occasionally ZMQ barfs here
            if hasattr(calibration.exp, 'extra_plot_server'):
                try:
                    calibration.exp.extra_plot_server.stop()
                except:
                    pass

class PulseCalibration(object):
    """Base class for calibration of qubit control pulses."""
    def __init__(self, qubit_names, quad="real"):
        super(PulseCalibration, self).__init__()
        self.qubit_names = qubit_names if isinstance(qubit_names, list) else [qubit_names]
        self.qubit     = [QubitFactory(qubit_name) for qubit_name in qubit_names] if isinstance(qubit_names, list) else QubitFactory(qubit_names)
        self.filename   = 'None'
        self.exp        = None
        self.axis_descriptor = None
        self.cw_mode    = False
        self.saved_settings = config.load_meas_file(config.meas_file)
        self.settings = deepcopy(self.saved_settings) #make a copy for used during calibration
        self.quad = quad
        if quad == "real":
            self.quad_fun = np.real
        elif quad == "imag":
            self.quad_fun = np.imag
        elif quad == "amp":
            self.quad_fun = np.abs
        elif quad == "phase":
            self.quad_fun = np.angle
        else:
            raise ValueError('Quadrature to calibrate must be one of ("real", "imag", "amp", "phase").')
        self.plot       = self.init_plot()


    def sequence(self):
        """Returns the sequence for the given calibration, must be overridden"""
        return [[Id(self.qubit), MEAS(self.qubit)]]

    def set(self, instrs_to_set = [], exp_step = 0, **params):
        try:
            extra_plot_server = self.exp.extra_plot_server
        except Exception as e:
            pass #no experiment yet created, or plot server not yet started
        meta_file = compile_to_hardware(self.sequence(**params), fileName=self.filename, axis_descriptor=self.axis_descriptor)
        if hasattr(self.exp, 'extra_plot_server'):
            extra_plot_server = self.exp.extra_plot_server
        self.exp = QubitExpFactory.create(meta_file=meta_file, calibration=True, save_data=False, cw_mode=self.cw_mode)
        self.exp.leave_plot_server_open = True
        self.exp.first_exp = not bool(exp_step)

        #Update all instruments that need to keep track of experiment numnber. Adapted from https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-python-dictionaries-and-lists
        def find_all_items(obj, key):
            ret = []
            if key in obj:
                out_path = obj
                ret.append(out_path)
            for k, v in obj.items():
                if isinstance(v, dict):
                    ret += find_all_items(v, key)
            return ret

        for k in find_all_items(self.exp.settings, 'exp_step'):
            k['exp_step'] = exp_step

        try:
            self.exp.extra_plot_server = extra_plot_server
        except:
            pass
        if self.plot:
            [self.exp.add_manual_plotter(p) for p in self.plot] if isinstance(self.plot, list) else self.exp.add_manual_plotter(self.plot)
        #sweep instruments for calibration
        for instr_to_set in instrs_to_set:
            par = FloatParameter()
            par.assign_method(getattr(self.exp._instruments[instr_to_set['instr']], instr_to_set['method']))
            if 'sweep_values' in instr_to_set.keys():
                par.value = instr_to_set['sweep_values'][0]
                self.exp.add_sweep(par, instr_to_set['sweep_values'])
            else:
                raise KeyError("Sweep values not defined.")

    def run(self, norm_pts = None):
        self.exp.run_sweeps()
        data = {}
        var = {}
        for buff in self.exp.buffers:
            if self.exp.writer_to_qubit[buff.name][0] in self.qubit_names:
                dataset, descriptor = buff.get_data(), buff.get_descriptor()
                qubit_name = self.exp.writer_to_qubit[buff.name][0]
                if norm_pts:
                    buff_data = normalize_data(dataset, zero_id = norm_pts[qubit_name][0], one_id = norm_pts[qubit_name][1])
                else:
                    buff_data = dataset['Data']
                data[qubit_name] = self.quad_fun(buff_data)
                if 'Variance' in dataset.dtype.names:
                    realvar = np.real(dataset['Variance'])
                    imagvar = np.imag(dataset['Variance'])
                    N = descriptor.metadata["num_averages"]
                    if self.quad in ['real', 'imag']:
                        var[qubit_name] = self.quad_fun(dataset['Variance']) / N
                    elif self.quad == 'amp':
                        var[qubit_name] = (realvar + imagvar) / N
                    elif self.quad == 'phase':
                        # take the approach from Qlab assuming the noise is
                        # Gaussian in both quadratures i.e. 'circular' in the
                        # IQ plane.
                        stddata = np.sqrt(realvar + imagvar)
                        stdtheta = 180/np.pi * 2 * np.arctan(stddata \
                            / abs(data[qubit_name]))
                        var[qubit_name] = (stdtheta**2) / N
                    else:
                        raise Exception('Variance of {} not available. Choose \
                            amp, phase, real or imag'.format(self.quad))
                else:
                    var[qubit_name] = None

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

    def update_settings(self):
        """Update calibrated YAML with calibration parameters"""
        config.dump_meas_file(self.saved_settings, config.meas_file)

    def write_to_log(self, cal_result):
        """Log calibration result"""
        logfile = os.path.join(config.LogDir, ''.join(self.qubit_names) + '_calibration_log.tsv')
        if len(self.qubit_names) == 1:
            log_columns = ["frequency", "pi2Amp", "piAmp", "drag_scaling"]
        elif len(self.qubit_names) == 2:
            log_columns = ['length', 'phase', 'amp']
        else:
            logger.error('Calibrations not supported for >2-qubit gates')
        log_columns+=['date', 'time']
        if os.path.isfile(logfile):
            lf = pd.read_csv(logfile, sep="\t")
        else:
            logger.info("Calibration log file created.")
            lf = pd.DataFrame(columns = log_columns)
        # Read the current (pre-cal) values for the parameters above
        if len(self.qubit_names) == 1:
            ctrl_settings = self.settings['qubits'][self.qubit_names[0]]['control']
            ctrl_freq = self.settings['instruments'][ctrl_settings['generator']]['frequency']
        else:
            ctrl_settings = self.settings['edges'][self.edge_name]
        cal_pars = {}
        for ind, p in enumerate(log_columns[:-2]):
            cal_pars[p] = ctrl_settings[p] if p in ctrl_settings else ctrl_settings['pulse_params'][p]
        if len(self.qubit_names) == 1:
            cal_pars['frequency'] = ctrl_freq + ctrl_settings['frequency'] # actual qubit frequency
        # Update with latest calibration
        cal_pars[cal_result[0]] = cal_result[1]
        new_cal_entry = [[cal_pars[p] for p in log_columns[:-2]] + [strftime("%y%m%d"), strftime("%H%M%S")]]
        lf = lf.append(pd.DataFrame(new_cal_entry, columns = log_columns), ignore_index = True)
        lf.to_csv(logfile, sep="\t", index= False)

class CavitySearch(PulseCalibration):
    def __init__(self, qubit_name, frequencies=np.linspace(4, 5, 100), **kwargs):
        super(CavitySearch, self).__init__(qubit_name, **kwargs)
        self.frequencies = frequencies
        self.cw_mode = True

    def sequence(self):
        return [[Id(self.qubit), MEAS(self.qubit)]]

    def calibrate(self):
        #find cavity source from config
        cavity_source = self.settings['qubits'][self.qubit.label]['measure']['generator']
        orig_freq = self.settings['instruments'][cavity_source]['frequency']
        instr_to_sweep = {'instr': cavity_source, 'method': 'set_frequency', 'sweep_values': self.frequencies}
        self.set([instr_to_sweep])
        data, _ = self.run()
        # Plot the results
        self.plot["Data"] = (self.frequencies, data)

    def init_plot(self):
        plot = ManualPlotter("Qubit Search", x_label='Frequency (GHz)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data", {'color': 'C1'})
        plot.add_fit_trace("Fit", {'color': 'C1'})
        return plot

class QubitSearch(PulseCalibration):
    def __init__(self, qubit_name, frequencies=np.linspace(4, 5, 100), **kwargs):
        super(QubitSearch, self).__init__(qubit_name, **kwargs)
        self.frequencies = frequencies
        self.cw_mode = True

    def sequence(self):
        return [[X(self.qubit), MEAS(self.qubit)]]

    def calibrate(self):
        #find qubit control source from config
        qubit_source = self.settings['qubits'][self.qubit.label]['control']['generator']
        orig_freq = self.settings['instruments'][qubit_source]['frequency']
        instr_to_sweep = {'instr': qubit_source, 'method': 'set_frequency', 'sweep_values': self.frequencies}
        self.set([instr_to_sweep])
        data, _ = self.run()

        # Plot the results
        self.plot["Data"] = (self.frequencies, data)

    def init_plot(self):
        plot = ManualPlotter("Qubit Search", x_label='Frequency (GHz)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data", {'color': 'C1'})
        plot.add_fit_trace("Fit", {'color': 'C1'})
        return plot

class RabiAmpCalibration(PulseCalibration):

    amp2offset = 0.5

    def __init__(self, qubit_name, num_steps = 40, **kwargs):
        super(RabiAmpCalibration, self).__init__(qubit_name, **kwargs)
        self.filename = 'Rabi/Rabi'
        if num_steps % 2 != 0:
            raise ValueError("Number of steps for RabiAmp calibration must be even!")
        #for now, only do one qubit at a time
        self.num_steps = num_steps
        self.amps = np.hstack((np.arange(-1, 0, 2./num_steps),
                            np.arange(2./num_steps, 1+2./num_steps, 2./num_steps)))

    def sequence(self):
        return ([[Xtheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps] +
            [[Ytheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps])

    def calibrate(self):
        self.set()
        data, _ = self.run()
        N = len(data)
        piI, offI, poptI = fit_rabi_amp(self.amps, data[:N//2])
        piQ, offQ, poptQ = fit_rabi_amp(self.amps, data[N//2:])
        #Arbitary extra division by two so that it doesn't push the offset too far.
        self.pi_amp = piI
        self.pi2_amp = piI/2.0
        self.i_offset = offI*self.amp2offset
        self.q_offset = offQ*self.amp2offset
        logger.info("Found X180 amplitude: {}".format(self.pi_amp))
        logger.info("Shifting I offset by: {}".format(self.i_offset))
        logger.info("Shifting Q offset by: {}".format(self.q_offset))
        finer_amps = np.linspace(np.min(self.amps), np.max(self.amps), 4*len(self.amps))
        self.plot["I Data"] = (self.amps, data[:N//2])
        self.plot["Q Data"] = (self.amps, data[N//2:])
        self.plot["I Fit"] = (finer_amps, rabi_amp_model(finer_amps, *poptI))
        self.plot["Q Fit"] = (finer_amps, rabi_amp_model(finer_amps, *poptQ))

        return ('piAmp', self.pi_amp)

    def init_plot(self):
        plot = ManualPlotter("Rabi Amplitude Cal", x_label="I/Q Amplitude", y_label="{} (Arb. Units)".format(self.quad))
        plot.add_data_trace("I Data", {'color': 'C1'})
        plot.add_data_trace("Q Data", {'color': 'C2'})
        plot.add_fit_trace("I Fit", {'color': 'C1'})
        plot.add_fit_trace("Q Fit", {'color': 'C2'})
        return plot

    def update_settings(self):
        #casting to float since YAML was complaining?
        self.saved_settings['qubits'][self.qubit.label]['control']['pulse_params']['piAmp'] = round(float(self.pi_amp), 5)
        self.saved_settings['qubits'][self.qubit.label]['control']['pulse_params']['pi2Amp'] = round(float(self.pi2_amp), 5)
        # a few contortions to get the right awg
        AWG = self.saved_settings['qubits'][self.qubit.label]['control']['AWG'].split(" ")[0]
        amp_factor = self.saved_settings['instruments'][AWG]['tx_channels']['12']['amp_factor']
        self.saved_settings['instruments'][AWG]['tx_channels']['12']['1']['offset'] += round(float(amp_factor*self.amp2offset*self.i_offset), 5)
        self.saved_settings['instruments'][AWG]['tx_channels']['12']['2']['offset'] += round(float(amp_factor*self.amp2offset*self.i_offset), 5)
        super(RabiAmpCalibration, self).update_settings()


class RamseyCalibration(PulseCalibration):
    def __init__(self, qubit_name, delays=np.linspace(0.0, 20.0, 41)*1e-6, two_freqs = False, added_detuning = 150e3, set_source = True, AIC = True, **kwargs):
        super(RamseyCalibration, self).__init__(qubit_name)
        self.filename = 'Ramsey/Ramsey'
        self.delays = delays
        self.two_freqs = two_freqs
        self.added_detuning = added_detuning
        self.set_source = set_source
        self.AIC = AIC #Akaike information criterion for model choice
        self.axis_descriptor = [delay_descriptor(self.delays)]

    def sequence(self):
        return [[X90(self.qubit), Id(self.qubit, delay), X90(self.qubit), MEAS(self.qubit)] for delay in self.delays]

    def init_plot(self):
        plot = ManualPlotter("Ramsey Fit", x_label='Time (us)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data", {'color': 'black'})
        plot.add_fit_trace("Fit", {'color': 'red'})
        return plot

    def calibrate(self):
        #find qubit control source (from config)
        qubit_source = self.settings['qubits'][self.qubit.label]['control']['generator']
        orig_freq = self.settings['instruments'][qubit_source]['frequency']
        #plot settings
        finer_delays = np.linspace(np.min(self.delays), np.max(self.delays), 4*len(self.delays))
        if self.set_source:
            self.settings['instruments'][qubit_source]['frequency'] = round(orig_freq + self.added_detuning, 10)
        else:
            self.settings['qubits'][self.qubit.label]['control']['frequency'] += float(self.added_detuning)
            config.dump_meas_file(self.settings, config.meas_file) # kludge to update qubit frequency
        self.set()
        data, _ = self.run()
        try:
            fit_freqs, fit_errs, all_params, all_errs = fit_ramsey(self.delays, data, two_freqs = self.two_freqs, AIC = self.AIC)
        except:
            self.update_settings()
        # Plot the results
        self.plot["Data"] = (self.delays, data)
        ramsey_f = ramsey_2f if len(fit_freqs) == 2 else ramsey_1f #1-freq fit if the 2-freq has failed
        self.plot["Fit"] = (finer_delays, ramsey_f(finer_delays, *all_params))

        #TODO: set conditions for success
        fit_freq_A = np.mean(fit_freqs) #the fit result can be one or two frequencies
        if self.set_source:
            self.settings['instruments'][qubit_source]['frequency'] = round(orig_freq + self.added_detuning + fit_freq_A/2, 10)
        else:
            self.settings['qubits'][self.qubit.label]['control']['frequency'] += float(fit_freq_A/2)
            config.dump_meas_file(self.settings, config.meas_file)
        self.set(exp_step = 1)
        data, _ = self.run()

        fit_freqs, fit_errs, all_params, all_errs = fit_ramsey(self.delays, data, two_freqs = self.two_freqs, AIC = self.AIC)
        try:
            fit_freqs, fit_errs, all_params, all_errs = fit_ramsey(self.delays, data, two_freqs = self.two_freqs, AIC = self.AIC)
        except:
            self.update_settings() # restore settings
        # Plot the results
        self.init_plot()
        self.plot["Data"] = (self.delays, data)
        ramsey_f = ramsey_2f if len(fit_freqs) == 2 else ramsey_1f
        self.plot["Fit"]  = (finer_delays, ramsey_f(finer_delays, *all_params))

        fit_freq_B = np.mean(fit_freqs)
        if fit_freq_B < fit_freq_A:
            self.fit_freq = round(orig_freq + self.added_detuning + 0.5*(fit_freq_A + 0.5*fit_freq_A + fit_freq_B), 10)
        else:
            self.fit_freq = round(orig_freq + self.added_detuning - 0.5*(fit_freq_A - 0.5*fit_freq_A + fit_freq_B), 10)
        if self.set_source:
            self.saved_settings['instruments'][qubit_source]['frequency'] = float(round(self.fit_freq))
        else:
            self.saved_settings['qubits'][self.qubit.label]['control']['frequency'] += float(round(self.fit_freq - orig_freq))
            # update edges where this is the target qubit
            for predecessor in ChannelLibraries.channelLib.connectivityG.predecessors(self.qubit):
                edge = ChannelLibraries.channelLib.connectivityG[predecessor][self.qubit]['channel']
                edge_source = self.saved_settings['edges'][edge.label]['generator']
                self.saved_settings['edges'][edge.label]['frequency'] = self.saved_settings['qubits'][self.qubit_names[0]]['control']['frequency'] + (self.saved_settings['instruments'][qubit_source]['frequency'] - self.saved_settings['instruments'][edge_source]['frequency'])
        qubit_set_freq = self.saved_settings['instruments'][qubit_source]['frequency'] + self.saved_settings['qubits'][self.qubit.label]['control']['frequency']
        logger.info("Qubit set frequency = {} GHz".format(round(float(qubit_set_freq/1e9),5)))
        return ('frequency', qubit_set_freq)

class PhaseEstimation(PulseCalibration):
    """Estimates pulse rotation angle from a sequence of P^k experiments, where
    k is of the form 2^n. Uses the modified phase estimation algorithm from
    Kimmel et al, quant-ph/1502.02677 (2015). Every experiment i doubled.
    vardata should be the variance of the mean"""

    def __init__(self, qubit_name, num_pulses= 1, amplitude= 0.1, direction = 'X', **kwargs):
        """Phase estimation calibration. Direction is either 'X' or 'Y',
        num_pulses is log2(n) of the longest sequence n,
        and amplitude is self-explanatory."""

        super(PhaseEstimation, self).__init__(qubit_name)
        self.filename        = 'RepeatCal/RepeatCal'
        self.direction       = direction
        self.amplitude       = amplitude
        self.num_pulses      = num_pulses
        self.target          = np.pi/2.0
        self.iteration_limit = 5

    def sequence(self):
        # Determine whether it is a single- or a two-qubit pulse calibration
        if isinstance(self.qubit, list):
            qubit = self.qubit[1]
            self.chan = self.saved_settings['edges'][self.edge_name]['pulse_params']
        else:
            self.chan = self.saved_settings['qubits'][self.qubit.label]['control']['pulse_params']
            qubit = self.qubit
        # define cal_pulse with updated amplitude
        cal_pulse = [ZX90_CR(*self.qubit, amp=self.amplitude)] if isinstance(self.qubit, list) else [Xtheta(self.qubit, amp=self.amplitude)]
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
        done_flag = 0
        while not done_flag:
            self.set(exp_step = ct-1)
            if isinstance(self, CRAmpCalibration_PhEst):
                #trick PulseCalibration to ignore the control qubit
                temp_qubit = copy(self.qubit)
                temp_qubit_names = copy(self.qubit_names)
                self.qubit = self.qubit[1]
                self.qubit_names.pop(0)

            [phase, sigma] = phase_estimation(*self.run())

            if isinstance(self, CRAmpCalibration_PhEst):
                self.qubit = copy(temp_qubit)
                self.qubit_names = copy(temp_qubit_names)

            logger.info("Phase: %.4f Sigma: %.4f"%(phase,sigma))
            #update amplitude
            self.amplitude, done_flag = phase_to_amplitude(phase, sigma, self.amplitude, self.target, ct, self.iteration_limit)
            ct+=1

        logger.info("Found amplitude for {} calibration of: {}".format(type(self).__name__, self.amplitude))
        #set_chan = self.qubit_names[0] if len(self.qubit) == 1 else ChannelLibraries.EdgeFactory(*self.qubits).label
        return (set_amp, self.amplitude)

    def update_settings(self):
        set_amp = 'pi2Amp' if isinstance(self, Pi2Calibration) else 'piAmp' if isinstance(self, PiCalibration) else 'amp'
        self.chan[set_amp] = round(float(self.amplitude), 5)
        super(PhaseEstimation, self).update_settings()

class Pi2Calibration(PhaseEstimation):
    def __init__(self, qubit_name, num_pulses= 9, **kwargs):
        super(Pi2Calibration, self).__init__(qubit_name, num_pulses = num_pulses, **kwargs)
        self.amplitude = self.qubit.pulse_params['pi2Amp']
        self.target    = np.pi/2.0

class PiCalibration(PhaseEstimation):
    def __init__(self, qubit_name, num_pulses= 9, **kwargs):
        super(PiCalibration, self).__init__(qubit_name, num_pulses = num_pulses, **kwargs)
        self.amplitude = self.qubit.pulse_params['piAmp']
        self.target    = np.pi

class CRAmpCalibration_PhEst(PhaseEstimation):
    def __init__(self, qubit_names, num_pulses= 9):
        super(CRAmpCalibration_PhEst, self).__init__(qubit_names, num_pulses = num_pulses)
        self.CRchan = ChannelLibraries.EdgeFactory(*self.qubit)
        self.amplitude = self.CRchan.pulse_params['amp']
        self.target    = np.pi/2
        self.edge_name = self.CRchan.label

class DRAGCalibration(PulseCalibration):
    def __init__(self, qubit_name, deltas = np.linspace(-1,1,21), num_pulses = np.arange(8, 48, 4)):
        self.filename = 'DRAG/DRAG'
        self.deltas = deltas
        self.num_pulses = num_pulses
        super(DRAGCalibration, self).__init__(qubit_name)

    def sequence(self):
        seqs = []
        for n in self.num_pulses:
            seqs += [[X90(self.qubit, drag_scaling = d), X90m(self.qubit, drag_scaling = d)]*n + [X90(self.qubit, drag_scaling = d), MEAS(self.qubit)] for d in self.deltas]
        seqs += create_cal_seqs((self.qubit,),2)
        return seqs

    def init_plot(self):
        plot = ManualPlotter("DRAG Cal", x_label=['DRAG parameter', 'Number of pulses'], y_label=['Amplitude (Arb. Units)', 'Fit DRAG parameter'], numplots = 2)
        cmap = cm.viridis(np.linspace(0, 1, len(self.num_pulses)))
        for n in range(len(self.num_pulses)):
            plot.add_data_trace('Data_{}'.format(n), {'color': list(cmap[n])})
            plot.add_fit_trace('Fit_{}'.format(n), {'color': list(cmap[n])})
        plot.add_data_trace('Data_opt', subplot_num = 1) #TODO: error bars
        return plot

    def calibrate(self):
        # run twice for different DRAG parameter ranges
        steps = 2
        for k in range(steps):
        #generate sequence
            self.set(exp_step = k)
            #first run
            data, _ = self.run()
            finer_deltas = np.linspace(np.min(self.deltas), np.max(self.deltas), 4*len(self.deltas))
            #normalize data with cals
            data = quick_norm_data(data)
            opt_drag, error_drag, popt_mat = fit_drag(data, self.deltas, self.num_pulses)

            #plot
            norm_data = data.reshape((len(self.num_pulses), len(self.deltas)))
            for n in range(len(self.num_pulses)):
                self.plot['Data_{}'.format(n)] = (self.deltas, norm_data[n, :])
                finer_deltas = np.linspace(np.min(self.deltas), np.max(self.deltas), 4*len(self.deltas))
                self.plot['Fit_{}'.format(n)] = (finer_deltas, quadf(finer_deltas, *popt_mat[:, n]))
            self.plot["Data_opt"] = (self.num_pulses, opt_drag) #TODO: add error bars

            if k < steps-1:
                #generate sequence with new pulses and drag parameters
                new_drag_step = 0.25*(max(self.deltas) - min(self.deltas))
                self.deltas = np.linspace(opt_drag[-1] - new_drag_step, opt_drag[-1] + new_drag_step, len(self.deltas))
                new_pulse_step = int(np.floor(2*(max(self.num_pulses)-min(self.num_pulses))/len(self.num_pulses)))
                self.num_pulses = np.arange(max(self.num_pulses) - new_pulse_step, max(self.num_pulses) + new_pulse_step*(len(self.num_pulses)-1), new_pulse_step)

        self.saved_settings['qubits'][self.qubit.label]['control']['pulse_params']['drag_scaling'] = round(float(opt_drag[-1]), 5)
        self.drag = opt_drag[-1]
        return ('drag_scaling', opt_drag[-1])

class MeasCalibration(PulseCalibration):
    def __init__(self, qubit_name):
        super(MeasCalibration, self).__init__(qubit_name)
        self.meas_name = "M-" + qubit_name

class CLEARCalibration(MeasCalibration):
    '''
    Calibration of cavity reset pulse
    aux_qubit: auxiliary qubit used for CLEAR pulse
    kappa: cavity linewidth (angular frequency: 1/s)
    chi: half of the dispershive shift (angular frequency: 1/s)
    tau: duration of each of the 2 depletion steps (s)
    alpha: scaling factor
    T1factor: decay due to T1 between end of msm't and start of Ramsey
    T2: measured T2*
    nsteps: calibration steps/sweep
    cal_steps: choose ranges for calibration steps. 1: +-100%; 0: skip step
    '''
    def __init__(self, qubit_name, aux_qubit, kappa = 2e6, chi = 1e6, t_empty = 200e-9, ramsey_delays=np.linspace(0.0, 50.0, 51)*1e-6, ramsey_freq = 100e3, meas_delay = 0, tau = 200e-9, \
    alpha = 1, T1factor = 1, T2 = 30e-6, nsteps = 11, eps1 = None, eps2 = None, cal_steps = (1,1,1)):
        super(CLEARCalibration, self).__init__(qubit_name)
        self.filename = 'CLEAR/CLEAR'
        self.aux_qubit = aux_qubit
        self.kappa = kappa
        self.chi = chi
        self.ramsey_delays = ramsey_delays
        self.ramsey_freq = ramsey_freq
        self.meas_delay = meas_delay
        self.tau = tau
        self.alpha = alpha
        self.T1factor = T1factor
        self.T2 = T2
        self.nsteps = nsteps
        if not eps1:
            # theoretical values as default
            self.eps1 = (1 - 2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))/(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
            self.eps2 = 1/(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
        self.cal_steps = cal_steps

    def sequence(self, **params):
        qM = QubitFactory(self.aux_qubit) #TODO: replace with MEAS(q) devoid of digitizer trigger
        prep = X(self.qubit) if params['state'] else Id(self.qubit)
        seqs = [[prep, MEAS(qM, amp1 = params['eps1'], amp2 =  params['eps2'], step_length = self.tau), X90(self.qubit), Id(self.qubit,d), U90(self.qubit,phase = self.ramsey_freq*d),
        Id(self.qubit, self.meas_delay), MEAS(self.qubit)] for d in self.ramsey_delays]
        seqs += create_cal_seqs((self.qubit,), 2, delay = self.meas_delay)
        return seqs

    def init_plot(self): #keep in a single plot?
        plot_raw = ManualPlotter("CLEAR Ramsey", x_label='Time (us)', y_label='<Z>')
        plot_res = ManualPlotter("CLEAR Cal", x_label= ['eps1, eps2', 'eps1', 'eps2'], y_label=['Residual photons n0', '', ''], numplots=3)

        plot_raw.add_data_trace('Data')
        plot_raw.add_fit_trace('Fit')
        for sweep_num, state in product([0,1,2], [0,1]):
            plot_res.add_data_trace('sweep {}, state {}'.format(sweep_num, state), {'color': 'C{}'.format(state+1)}, sweep_num) #TODO: error bar
            plot_res.add_fit_trace('Fit sweep {}, state {}'.format(sweep_num, state), {'color' : 'C{}'.format(state+1)}, sweep_num) #TODO
        return [plot_raw, plot_res]


    def calibrate(self):
        cal_step = 0
        for ct in range(3):
            if not self.cal_steps[ct]:
                continue
            #generate sequence
            xpoints = np.linspace(1-self.cal_steps[ct], 1+self.cal_steps[ct], self.nsteps)
            n0vec = np.zeros(self.nsteps)
            err0vec = np.zeros(self.nsteps)
            n1vec = np.zeros(self.nsteps)
            err1vec = np.zeros(self.nsteps)
            for k in range(self.nsteps):
                eps1 = self.eps1 if k==1 else xpoints[k]*self.eps1
                eps2 = self.eps2 if k==2 else xpoints[k]*self.eps2
                #run for qubit in 0/1
                for state in [0,1]:
                    self.set(eps1 = eps1, eps2 = eps2, state = state, exp_step = cal_step)
                    #analyze
                    data, _ = self.run()
                    norm_data = quick_norm_data(data)
                    eval('n{}vec'.format(state))[k], eval('err{}vec'.format(state))[k], fit_curve = fit_photon_number(self.ramsey_delays, norm_data, [self.kappa, self.ramsey_freq, 2*self.chi, self.T2, self.T1factor, 0])
                    #plot
                    self.plot[0]['Data'] = (self.ramsey_delays, norm_data)
                    self.plot[0]['Fit'] = fit_curve
                    self.plot[1]['sweep {}, state 0'.format(ct)] = (xpoints, n0vec)
                    self.plot[1]['sweep {}, state 1'.format(ct)] = (xpoints, n1vec)
                    cal_step+=1

            #fit for minimum photon number
            popt_0,_ = fit_quad(xpoints, n0vec)
            popt_1,_ = fit_quad(xpoints, n1vec)
            finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
            opt_scaling = np.mean(popt_0[0], popt_1[0])
            logger.info("Optimal scaling factor for step {} = {}".format(ct+1, opt_scaling))

            if ct<2:
                self.eps1*=opt_scaling
            if ct!=1:
                self.eps2*=opt_scaling
            self.plot[1]['Fit sweep {}, state 0'.format(ct)] = (finer_xpoints, quadf(finer_xpoints, popt_0))
            self.plot[1]['Fit sweep {}, state 1'.format(ct)] = (finer_xpoints, quadf(finer_xpoints, popt_1))

        def update_settings(self):
            #update library (default amp1, amp2 for MEAS)
            self.saved_settings['qubits'][self.qubit.label]['measure']['pulse_params']['amp1'] = round(float(self.eps1), 5)
            self.saved_settings['qubits'][self.qubit.label]['measure']['pulse_params']['amp2'] = round(float(self.eps2), 5)
            self.saved_settings['qubits'][self.qubit.label]['measure']['pulse_params']['step_length'] = round(float(self.tau), 5)
            super(CLEARCalibration, self).update_settings()

'''Two-qubit gate calibrations'''
class CRCalibration(PulseCalibration):
    def __init__(self, qubit_names, lengths=np.linspace(20, 1020, 21)*1e-9, phase = 0, amp = 0.8, rise_fall = 40e-9):
        super(CRCalibration, self).__init__(qubit_names)
        self.lengths = lengths
        self.phases = phase
        self.amps = amp
        self.rise_fall = rise_fall
        self.filename = 'CR/CR'
        self.edge_name = ChannelLibraries.EdgeFactory(*self.qubit).label

    def init_plot(self):
        plot = ManualPlotter("CR"+str.lower(self.cal_type.name)+"Fit", x_label=str.lower(self.cal_type.name), y_label='$<Z_{'+self.qubit_names[1]+'}>$', y_lim=(-1.02,1.02))
        plot.add_data_trace("Data 0", {'color': 'C1'})
        plot.add_fit_trace("Fit 0", {'color': 'C1'})
        plot.add_data_trace("Data 1", {'color': 'C2'})
        plot.add_fit_trace("Fit 1", {'color': 'C2'})
        return plot

    def calibrate(self):
        # generate sequence
        self.set()
        # run and load normalized data
        data, _ = self.run(norm_pts = {self.qubit_names[0]: (0, 1), self.qubit_names[1]: (0, 2)})
        # select target qubit
        data_t = data[self.qubit_names[1]]
        # fit
        self.opt_par, all_params_0, all_params_1 = fit_CR([self.lengths, self.phases, self.amps], data_t, self.cal_type)
        # plot the result
        xaxis = self.lengths if self.cal_type==CR_cal_type.LENGTH else self.phases if self.cal_type==CR_cal_type.PHASE else self.amps
        finer_xaxis = np.linspace(np.min(xaxis), np.max(xaxis), 4*len(xaxis))
        self.plot["Data 0"] = (xaxis,       data_t[:len(data_t)//2])
        self.plot["Fit 0"] =  (finer_xaxis, np.polyval(all_params_0, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, *all_params_0))
        self.plot["Data 1"] = (xaxis,       data_t[len(data_t)//2:])
        self.plot["Fit 1"] =  (finer_xaxis, np.polyval(all_params_1, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, *all_params_1))
        return (str.lower(self.cal_type.name), self.opt_par)

    def update_settings(self):
        self.saved_settings['edges'][self.edge_name]['pulse_params'][str.lower(self.cal_type.name)] = float(self.opt_par)
        super(CRCalibration, self).update_settings()

class CRLenCalibration(CRCalibration):
    def __init__(self, qubit_names, lengths=np.linspace(20, 1020, 21)*1e-9, phase = 0, amp = 0.8, rise_fall = 40e-9, cal_type = CR_cal_type.LENGTH):
        self.cal_type = cal_type
        super(CRLenCalibration, self).__init__(qubit_names, lengths, phase, amp, rise_fall)

    def sequence(self):
        qc, qt = self.qubit
        seqs = [[Id(qc)] + echoCR(qc, qt, length=l, phase = self.phases, amp=self.amps, riseFall=self.rise_fall).seq + [Id(qc), MEAS(qt)*MEAS(qc)]
        for l in self.lengths]+ [[X(qc)] + echoCR(qc, qt, length=l, phase= self.phases, amp=self.amps, riseFall=self.rise_fall).seq + [X(qc), MEAS(qt)*MEAS(qc)]
        for l in self.lengths] + create_cal_seqs((qt,qc), 2, measChans=(qt,qc))

        self.axis_descriptor=[
            delay_descriptor(np.concatenate((self.lengths, self.lengths))),
            cal_descriptor(tuple(self.qubit), 2)
        ]

        return seqs

class CRPhaseCalibration(CRCalibration):
    def __init__(self, qubit_names, phases = np.linspace(0,2*np.pi,21), amp = 0.8, rise_fall = 40e-9, cal_type = CR_cal_type.PHASE):
        self.cal_type = cal_type
        super(CRPhaseCalibration, self).__init__(qubit_names, 0, phases, amp, rise_fall)
        CRchan = ChannelLibraries.EdgeFactory(*self.qubit)
        self.lengths = CRchan.pulse_params['length']


    def sequence(self):
        qc, qt = self.qubit
        seqs = [[Id(qc)] + echoCR(qc, qt, length=self.lengths, phase=ph, amp=self.amps, riseFall=self.rise_fall).seq + [X90(qt)*Id(qc), MEAS(qt)*MEAS(qc)]
        for ph in self.phases]+ [[X(qc)] + echoCR(qc, qt, length=self.lengths, phase= ph, amp=self.amps, riseFall=self.rise_fall).seq + [X90(qt)*X(qc), MEAS(qt)*MEAS(qc)]
        for ph in self.phases] + create_cal_seqs((qt,qc), 2, measChans=(qt,qc))

        self.axis_descriptor = [
            {
                'name': 'phase',
                'unit': 'radians',
                'points': list(self.phases)+list(self.phases),
                'partition': 1
            },
            cal_descriptor(tuple(self.qubit), 2)
        ]

        return seqs

class CRAmpCalibration(CRCalibration):
    def __init__(self, qubit_names, amp_range = 0.4, amp = 0.8, rise_fall = 40e-9, num_CR = 1, cal_type = CR_cal_type.AMP):
        self.num_CR = num_CR
        if num_CR % 2 == 0:
            logger.error('The number of ZX90 must be odd')
        self.cal_type = cal_type
        amps = np.linspace((1-amp_range/2)*amp, (1+amp_range/2)*amp, 21)
        super(CRAmpCalibration, self).__init__(qubit_names, 0, 0, amps, rise_fall)
        CRchan = ChannelLibraries.EdgeFactory(*self.qubit)
        self.lengths = CRchan.pulse_params['length']
        self.phases = CRchan.pulse_params['phase']

    def sequence(self):
        qc, qt = self.qubit
        seqs = [[Id(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths, phase=self.phases, amp=a, riseFall=self.rise_fall).seq + [Id(qc), MEAS(qt)*MEAS(qc)]
        for a in self.amps]+ [[X(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths, phase= self.phases, amp=a, riseFall=self.rise_fall).seq + [X(qc), MEAS(qt)*MEAS(qc)]
        for a in self.amps] + create_cal_seqs((qt,qc), 2, measChans=(qt,qc))

        self.axis_descriptor = [
            {
                'name': 'amplitude',
                'unit': None,
                'points': list(self.amps)+list(self.amps),
                'partition': 1
            },
            cal_descriptor(tuple(self.qubit), 2)
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
            logger.info('Phase estimation terminated at %dth pulse because the (x,z) vector is too short'%(k))
            break

        lowerBound = restrict(curGuess - np.pi/2**(k))
        upperBound = restrict(curGuess + np.pi/2**(k))
        possiblesTest = [ restrict((phases[k] + 2*n*np.pi)/2**(k)) for n in range(0,2**(k)+1)]

        if verbose == True:
            logger.info('Lower Bound: %f'%lowerBound)
            logger.info('Upper Bound: %f'%upperBound)

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
            logger.info('Current Guess: %f'%(curGuess))

        phase = curGuess
        sigma = np.maximum(np.abs(restrict(curGuess - lowerBound)), np.abs(restrict(curGuess - upperBound)))

    return phase, sigma

def phase_to_amplitude(phase, sigma, amp, target, ct, iteration_limit=5):
    # correct for some errors related to 2pi uncertainties
    if np.sign(phase) != np.sign(amp):
        phase += np.sign(amp)*2*np.pi
    angle_error = phase - target;
    logger.info('Angle error: %.4f'%angle_error);

    amp_target = target/phase * amp
    amp_error = amp - amp_target
    logger.info('Set amplitude: %.4f\n'%amp)
    logger.info('Amplitude error: %.4f\n'%amp_error)

    amp = amp_target
    done_flag = 0

    # check for stopping condition
    phase_error = phase - target
    if np.abs(phase_error) < 1e-2 or np.abs(phase_error/sigma) < 1 or ct > iteration_limit:
        if np.abs(phase_error) < 1e-2:
            logger.info('Reached target rotation angle accuracy');
        elif abs(phase_error/sigma) < 1:
            logger.info('Reached phase uncertainty limit');
        else:
            logger.info('Hit max iteration count');
        done_flag = 1
    return amp, done_flag

def quick_norm_data(data): #TODO: generalize as in Qlab.jl
    """Rescale data assuming 2 calibrations / single qubit state at the end of the sequence"""
    data = 2*(data-np.mean(data[-4:-2]))/(np.mean(data[-4:-2])-np.mean(data[-2:])) + 1
    data = data[:-4]
    return data
