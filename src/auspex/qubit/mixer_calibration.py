# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import auspex.config as config
from copy import copy
import os
import json

import numpy as np
from scipy.optimize import curve_fit
import time
from time import sleep

from .qubit_exp import instrument_map
from auspex.log import logger
from auspex.filters import DataBuffer
from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.qubit.pulse_calibration import Calibration
from auspex.filters.plot import ManualPlotter

def find_null_offset(xpts, powers, default=0.0):
    """Finds the offset corresponding to the minimum power using a fit to the measured data"""
    def model(x, a, b, c):
        return a*(x - b)**2 + c
    powers = np.power(10, powers/10.)
    min_idx = np.argmin(powers)
    try:
        fit = curve_fit(model, xpts, powers, p0=[1, xpts[min_idx], powers[min_idx]])
    except RuntimeError:
        logger.warning("Mixer null offset fit failed.")
        return default, np.zeros(len(powers))
    best_offset = np.real(fit[0][1])
    best_offset = np.minimum(best_offset, xpts[-1])
    best_offset = np.maximum(best_offset, xpts[0])
    xpts_fine = np.linspace(xpts[0],xpts[-1],101)
    fit_pts = np.array([np.real(model(x, *fit[0])) for x in xpts_fine])
    if min(fit_pts)<0: fit_pts-=min(fit_pts)-1e-10 #prevent log of a negative number
    return best_offset, xpts_fine, 10*np.log10(fit_pts)


class MixerCalibration(Calibration):

    MIN_OFFSET = -0.4
    MAX_OFFSET = 0.4
    MIN_AMPLITUDE = 0.2
    MAX_AMPLITUDE = 1.2
    MIN_PHASE = -0.3
    MAX_PHASE = 0.3

    def __init__(self, channel, spectrum_analyzer, mixer="control", first_cal="phase",
                offset_range = (-0.2,0.2), amp_range = (0.6,1.4), phase_range = (-np.pi/6,np.pi/6), nsteps = 51, plot=True):
        self.channel = channel
        self.spectrum_analyzer = spectrum_analyzer
        self.mixer = mixer
        self.do_plotting = plot
        self.first_cal = first_cal
        self.offset_range = offset_range
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.nsteps = nsteps
        super(MixerCalibration, self).__init__()

    def init_plots(self):
        self.plt1 = ManualPlotter(name="Mixer offset calibration", x_label='{} {} offset (V)'.format(self.channel, self.mixer), y_label='Power (dBm)')
        self.plt1.add_data_trace("I-offset", {'color': 'C1'})
        self.plt1.add_data_trace("Q-offset", {'color': 'C2'})
        self.plt1.add_fit_trace("Fit I-offset", {'color': 'C1'}) #TODO: fix axis labels
        self.plt1.add_fit_trace("Fit Q-offset", {'color': 'C2'})

        self.plt2 = ManualPlotter(name="Mixer amp calibration", x_label='{} {} amplitude (V)'.format(self.channel, self.mixer), y_label='Power (dBm)')
        self.plt2.add_data_trace("amplitude_factor", {'color': 'C4'})
        self.plt2.add_fit_trace("Fit amplitude_factor", {'color': 'C4'})

        self.plt3 = ManualPlotter(name="Mixer phase calibration", x_label='{} {} phase (rad)'.format(self.channel, self.mixer), y_label='Power (dBm)')
        self.plt3.add_data_trace("phase_skew", {'color': 'C3'})
        self.plt3.add_fit_trace("Fit phase_skew", {'color': 'C3'})

        self.plotters = [self.plt1, self.plt2, self.plt3]
        return self.plotters

    def _calibrate(self):

        offset_pts = np.linspace(self.offset_range[0], self.offset_range[1], self.nsteps)
        amp_pts    = np.linspace(self.amp_range[0], self.amp_range[1], self.nsteps)
        phase_pts  = np.linspace(self.phase_range[0], self.phase_range[1], self.nsteps)
        first_cal  = self.first_cal

        config_dict = {"I_offset": 0.0, "Q_offset": 0.0, "phase_skew": 0.0,
                       "amplitude_factor": 1.0, "sideband_modulation": False}

        I1_amps = self.run_sweeps("I_offset", offset_pts, config_dict)
        try:
            I1_offset, xpts, ypts = find_null_offset(offset_pts[1:], I1_amps[1:])
        except:
            raise ValueError("Could not find null offset")
        self.plt1["I-offset"] = (offset_pts, I1_amps)
        self.plt1["Fit I-offset"] = (xpts, ypts)
        logger.info("Found first pass I offset of {}.".format(I1_offset))
        config_dict['I_offset'] = I1_offset

        Q1_amps = self.run_sweeps("Q_offset", offset_pts, config_dict)
        try:
            Q1_offset, xpts, ypts = find_null_offset(offset_pts[1:], Q1_amps[1:])
        except:
            raise ValueError("Could not find null offset")
        self.plt1["Q-offset"] = (offset_pts, Q1_amps)
        self.plt1["Fit Q-offset"] = (xpts, ypts)
        logger.info("Found first pass Q offset of {}.".format(Q1_offset))
        config_dict['Q_offset'] = Q1_offset

        I2_amps = self.run_sweeps("I_offset", offset_pts, config_dict)
        try:
            I2_offset, xpts, ypts = find_null_offset(offset_pts[1:], I2_amps[1:])
        except:
            raise ValueError("Could not find null offset")
        self.plt1["I-offset"] = (offset_pts, I2_amps)
        self.plt1["Fit I-offset"] = (xpts, ypts)
        logger.info("Found second pass I offset of {}.".format(I2_offset))
        config_dict['I_offset'] = I2_offset

        #this is a bit hacky but OK...
        cals = {"phase": "phase_skew", "amplitude": "amplitude_factor"}
        cal_pts = {"phase": phase_pts, "amplitude": amp_pts}
        correct_plotter = {"phase": self.plt3, "amplitude": self.plt2}
        cal_defaults = {"phase": 0.0, "amplitude": 1.0}
        if first_cal not in cals.keys():
            raise ValueError("First calibration should be one of ('phase, amplitude'). Instead got {}".format(first_cal))
        second_cal = list(set(cals.keys()).difference({first_cal,}))[0]

        config_dict['sideband_modulation'] = True

        amps1 = self.run_sweeps(cals[first_cal], cal_pts[first_cal], config_dict)
        try:
            offset1, xpts, ypts = find_null_offset(cal_pts[first_cal][1:], amps1[1:], default=cal_defaults[first_cal])
        except:
            raise ValueError("Could not find null offset")
        correct_plotter[first_cal][cals[first_cal]] = (cal_pts[first_cal], amps1)
        correct_plotter[first_cal]["Fit "+cals[first_cal]] = (xpts, ypts)
        logger.info("Found {} of {}.".format(str.replace(cals[first_cal], '_', ' '), offset1))
        config_dict[cals[first_cal]] = offset1

        amps2 = self.run_sweeps(cals[second_cal], cal_pts[second_cal], config_dict)
        try:
            offset2, xpts, ypts = find_null_offset(cal_pts[second_cal][1:], amps2[1:], default=cal_defaults[second_cal])
        except:
            raise ValueError("Could not find null offset")
        correct_plotter[second_cal][cals[second_cal]] = (cal_pts[second_cal], amps2)
        correct_plotter[second_cal]["Fit "+cals[second_cal]] = (xpts, ypts)
        logger.info("Found {} of {}.".format(str.replace(cals[first_cal], '_', ' '), offset2))
        config_dict[cals[second_cal]] = offset2

        # if write_to_file:
        #     mce.write_to_file()
        logger.info(("Mixer calibration: I offset = {}, Q offset = {}, "
                    "Amplitude Imbalance = {}, Phase Skew = {}").format(config_dict["I_offset"],
                                                                        config_dict["Q_offset"],
                                                                        config_dict["amplitude_factor"],
                                                                        config_dict["phase_skew"]))

        assert config_dict["I_offset"] > self.MIN_OFFSET and config_dict["I_offset"] < self.MAX_OFFSET, "I_offset looks suspicious."
        assert config_dict["Q_offset"] > self.MIN_OFFSET and config_dict["Q_offset"] < self.MAX_OFFSET, "Q_offset looks suspicious."
        assert config_dict["amplitude_factor"] > self.MIN_AMPLITUDE and config_dict["amplitude_factor"] < self.MAX_AMPLITUDE, "amplitude_factor looks suspicious."
        assert config_dict["phase_skew"] > self.MIN_PHASE and config_dict["phase_skew"] < self.MAX_PHASE, "phase_skew looks suspicious."

        self.succeeded = True
        self.config_dict = config_dict

    def update_settings(self):
        self.exp._phys_chan.amp_factor = self.config_dict["amplitude_factor"]
        self.exp._phys_chan.phase_skew = self.config_dict["phase_skew"]
        self.exp._phys_chan.I_channel_offset = self.config_dict["I_offset"]
        self.exp._phys_chan.Q_channel_offset = self.config_dict["Q_offset"]

    def run_sweeps(self, sweep_parameter, pts, config_dict):
        self.exp = MixerCalibrationExperiment(self.channel, self.spectrum_analyzer, config_dict, mixer=self.mixer)
        self.exp.add_sweep(getattr(self.exp, sweep_parameter), pts)
        self.exp.run_sweeps()
        return self.exp.buff.get_data()['amplitude']

class MixerCalibrationExperiment(Experiment):

    SSB_FREQ = 10e6

    amplitude = OutputConnector(unit='dBc')

    I_offset = FloatParameter(default=0.0, unit="V")
    Q_offset = FloatParameter(default=0.0, unit="V")
    amplitude_factor = FloatParameter(default=1.0)
    phase_skew = FloatParameter(default=0.0, unit="rad")

    sideband_modulation = False

    def __init__(self, channel, spectrum_analyzer, config_dict, mixer="control"):
        """Initialize MixerCalibrationExperiment Experiment.
            Args:
                channel: channel identifier (qubit or edge)
                spectrum_analyzer: which spectrum analyzer should be used.
                mixer: One of 'control', 'measure' to select which mixer to cal.
        """
        super(MixerCalibrationExperiment, self).__init__()

        self.channel = channel
        self.config_dict = config_dict
        self.sideband_modulation = config_dict["sideband_modulation"]
        self._sa = spectrum_analyzer
        assert self._sa.LO_source is not None, "No microwave source associated with spectrum analyzer"
        self._LO = self._sa.LO_source
        if mixer.lower() == "measure":
            self._awg = channel.measure_chan.phys_chan.transmitter
            self._phys_chan = channel.measure_chan.phys_chan
            self._source = channel.measure_chan.phys_chan.generator
        elif mixer.lower() == "control":
            self._awg = channel.phys_chan.transmitter
            self._phys_chan = channel.phys_chan
            self._source = channel.phys_chan.generator
        else:
            raise ValueError("Unknown mixer {}: must be either 'measure' or 'control'.".format(mixer))

        self.instrument_proxies = [self._sa, self._LO, self._awg, self._source]
        self.instruments = []
        for instrument in self.instrument_proxies:
            instr = instrument_map[instrument.model](instrument.address, instrument.label) # Instantiate
            # For easy lookup
            instr.proxy_obj = instrument
            instrument.instr = instr
            # Add to the experiment's instrument list
            self._instruments[instrument.label] = instr
            self.instruments.append(instr)

        self.sa = self._instruments[self._sa.label]
        self.LO = self._instruments[self._LO.label]
        self.source = self._instruments[self._source.label]
        self.awg = self._instruments[self._awg.label]

        self.buff = DataBuffer()
        edges = [(self.amplitude, self.buff.sink)]
        self.set_graph(edges)

    def connect_instruments(self):
        """Extend connect_instruments to reset I,Q offsets and amplitude and phase
        imbalance."""
        super(MixerCalibrationExperiment, self).connect_instruments()
        self.awg.set_offset(0, 0.0)
        self.awg.set_offset(1, 0.0)
        self.awg.set_mixer_amplitude_imbalance(0.0)
        self.awg.set_mixer_phase_skew(0.0)

    def init_instruments(self):
        for k,v in self.config_dict.items():
            if k != "sideband_modulation":
                getattr(self, k).value = v

        self.I_offset.assign_method(lambda x: self.awg.set_offset(0, x))
        self.Q_offset.assign_method(lambda x: self.awg.set_offset(1, x))
        self.amplitude_factor.assign_method(self.awg.set_mixer_amplitude_imbalance)
        self.phase_skew.assign_method(self.awg.set_mixer_phase_skew)

        self.I_offset.add_post_push_hook(lambda: time.sleep(0.1))
        self.Q_offset.add_post_push_hook(lambda: time.sleep(0.1))
        self.amplitude_factor.add_post_push_hook(lambda: time.sleep(0.1))
        self.phase_skew.add_post_push_hook(lambda: time.sleep(0.1))

        for name, instr in self._instruments.items():
            # Configure with dictionary from the instrument proxy
            instr.configure_with_proxy(instr.proxy_obj)

        #make sure the microwave generators are set up properly
        self.source.output = True
        LO_freq = self.source.frequency - self.sa.IF_FREQ
        if self.sideband_modulation:
            LO_freq -= self.SSB_FREQ
        self.LO.frequency = LO_freq
        self.LO.output = True
        self._setup_awg_ssb()
        time.sleep(0.1)

    def reset_calibration(self):
        try:
            self.awg.set_mixer_amplitude_imbalance(1.0)
            self.awg.set_mixer_phase_skew(0.0)
            self.awg.set_offset(0, 0.0)
            self.awg.set_offset(1, 0.0)
        except Exception as ex:
            raise Exception("Could not reset APS2 mixer calibration. Is the AWG connected?") from ex

    def _setup_awg_ssb(self):
        #set up ingle sideband modulation IQ playback on the AWG
        self.awg.stop()
        self.awg.load_waveform(1, 0.5*np.ones(1200, dtype=np.float))
        self.awg.load_waveform(2, np.zeros(1200, dtype=np.float))
        self.awg.waveform_frequency = -self.SSB_FREQ
        self.awg.run_mode = "CW_WAVEFORM"
        #start playback
        self.awg.run()
        logger.debug("Playing SSB CW IQ modulation on {} at frequency: {} MHz".format(self.awg, self.SSB_FREQ/1e6))

    def shutdown_instruments(self):
        #reset the APS2, just in case.
        self.LO.output = False
        self.source.output = False
        self.awg.stop()

    def init_streams(self):
        pass

    def run(self):
        self.amplitude.push(self.sa.peak_amplitude())
