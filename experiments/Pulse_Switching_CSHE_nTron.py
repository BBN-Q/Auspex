# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import KeysightM8190A, Scenario, Sequence
from auspex.instruments import Picosecond10070A
from auspex.instruments import SR865
from auspex.instruments import Keithley2400
from auspex.instruments import AMI430
from auspex.instruments import Attenuator

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Print, WriteToHDF5, Averager
from auspex.log import logger
from PyDAQmx import *

import itertools
import numpy as np

import time, sys
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

import auspex.analysis.switching as sw
from adapt import refine

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

def arb_pulse(amplitude, duration, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude
    return wf

def pulse_voltage_lookup(nTron_calib="calibration/nTron_20160815.csv",
                        midpoint_calib="calibration/midpoint_20160718.csv"):
    df_midpoint = pd.read_csv(midpoint_calib, sep=",")
    df_nTron = pd.read_csv(nTron_calib, sep=",")
    midpoint_lookup = interp1d(df_midpoint["Sample Voltage"],df_midpoint["Midpoint Voltage"])
    nTron_control_lookup = interp1d(df_nTron["Amp Out"],df_nTron["Control Voltage"])
    sample_volts = []
    control_volts = []
    for volt in df_midpoint['Sample Voltage']:
        mid_volt = midpoint_lookup(volt)
        if (mid_volt > min(df_nTron['Amp Out'])) and (mid_volt < max(df_nTron['Amp Out'])):
            sample_volts.append(volt)
            control_volts.append(nTron_control_lookup(mid_volt))
    return interp1d(sample_volts, control_volts)

def ntron_pulse(amplitude=1.0, rise_time=80e-12, hold_time=170e-12, fall_time=1.0e-9, sample_rate=12e9):
    delay    = 2.0e-9 # Wait a few TCs for the rising edge
    duration = delay + hold_time + 6.0*fall_time # Wait 6 TCs for the slow decay
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        duration = 319/sample_rate
        # times = np.arange(0, duration, 1/sample_rate)
        times = np.linspace(0, duration, 320)
    else:
        pulse_points = 64*np.ceil(pulse_points/64.0)
        duration = (pulse_points-1)/sample_rate
        # times = np.arange(0, duration, 1/sample_rate)
        times = np.linspace(0, duration, pulse_points)

    rise_mask = np.less(times, delay)
    hold_mask = np.less(times, delay + hold_time)*np.greater_equal(times, delay)
    fall_mask = np.greater_equal(times, delay + hold_time)

    wf  = rise_mask*np.exp((times-delay)/rise_time)
    wf += hold_mask
    wf += fall_mask*np.exp(-(times-delay-hold_time)/fall_time)

    return amplitude*wf

class nTronSwitchingExperiment(Experiment):

    # Sample information
    sample         = "CSHE"
    comment        = "Phase Diagram with nTron pulse"
    # Parameters
    field          = FloatParameter(default=0.0, unit="T")
    pulse_voltage  = FloatParameter(default=0.2, unit="V")
    pulse_duration = FloatParameter(default=1e-9, unit="s")
    pulse_durations = [0]
    pulse_voltages  = [0]

    # Constants (set with attribute access if you want to change these!)
    iteration       = 5
    attempts        = 1 << 11
    settle_delay    = 200e-6
    measure_current = 3.0e-6
    samps_per_trig  = 5

    polarity        = 1

    min_daq_voltage = 0.0
    max_daq_voltage = 0.4

    reset_amplitude = 0.2
    reset_duration  = 5.0e-9

    # Things coming back
    voltage     = OutputConnector()

    # Instrument resources
    mag   = AMI430("192.168.5.109")
    # lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")
    # pspl  = Picosecond10070A("GPIB0::24::INSTR")
    # atten = Attenuator("calibration/RFSA2113SB_HPD_20160706.csv", lock.set_ao2, lock.set_ao3)
    arb   = KeysightM8190A("192.168.5.108")
    keith = Keithley2400("GPIB0::25::INSTR")

    def init_instruments(self):

        # ===================
        #    Setup the Keithley
        # ===================

        self.keith.triad()
        self.keith.conf_meas_res(res_range=1e5)
        self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
        self.keith.current = self.measure_current
        self.mag.ramp()

        # ===================
        #    Setup the AWG
        # ===================

        self.arb.set_output(True, channel=1)
        self.arb.set_output(False, channel=2)
        self.arb.sample_freq = 12.0e9
        self.arb.waveform_output_mode = "WSPEED"
        self.arb.set_output_route("DC", channel=1)
        self.arb.voltage_amplitude = 1.0
        self.arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
        self.arb.set_marker_level_high(1.5, channel=1, marker_type="sync")
        self.arb.continuous_mode = False
        self.arb.gate_mode = False

        self.setup_arb()

        # ===================
        #   Setup the NIDAQ
        # ===================

        self.analog_input = Task()
        self.read = int32()
        self.buf_points = 2*self.samps_per_trig*self.attempts
        self.analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff,
            self.min_daq_voltage, self.max_daq_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , self.samps_per_trig)
        self.analog_input.CfgInputBuffer(self.buf_points)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        self.analog_input.SetStartTrigRetriggerable(1)
        self.analog_input.StartTask()

        # Assign methods
        self.field.assign_method(self.mag.set_field)
    #     self.pulse_voltage.assign_method(self.set_arb_voltage)
    #     self.pulse_duration.assign_method(self.set_arb_duration)
    #
    # def set_arb_voltage(self, voltage):
    #     self.setup_arb(voltage=voltage, duration=self.duration.value)
    #

    def setup_arb(self):
        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        reset_wf    = arb_pulse(-self.polarity*self.reset_amplitude, self.reset_duration)
        wf_data     = KeysightM8190A.create_binary_wf_data(reset_wf)
        rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, rst_segment_id)

        no_reset_wf = arb_pulse(0.0, 3.0/12e9)
        wf_data     = KeysightM8190A.create_binary_wf_data(no_reset_wf)
        no_rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, no_rst_segment_id)

        # nTron waveforms
        nTron_control_voltage = pulse_voltage_lookup()
        nTron_segment_ids = []
        for dur,vpeak in zip(self.pulse_durations, self.pulse_voltages):
            volt = self.polarity*nTron_control_voltage(vpeak)
            logger.debug("Set nTron pulse: {}V -> AWG {}V, {}s".format(vpeak,volt,dur))
            ntron_wf    = ntron_pulse(amplitude=volt, fall_time=dur)
            wf_data     = KeysightM8190A.create_binary_wf_data(ntron_wf)
            ntron_segment_id  = self.arb.define_waveform(len(wf_data))
            self.arb.upload_waveform(wf_data, ntron_segment_id)
            nTron_segment_ids.append(ntron_segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
        nidaq_trig_segment_id = self.arb.define_waveform(len(nidaq_trig_wf))
        self.arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

        settle_pts = int(640*np.ceil(self.settle_delay * 12e9 / 640))

        self.start_idxs = [0]
        self.start_id = 0
        for si in nTron_segment_ids:
            scenario = Scenario()
            seq = Sequence(sequence_loop_ct=int(self.attempts))
            seq.add_waveform(rst_segment_id)
            seq.add_idle(settle_pts, 0.0)
            seq.add_waveform(nidaq_trig_segment_id)
            seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
            # seq.add_waveform(pspl_trig_segment_id)
            seq.add_waveform(si)
            seq.add_idle(settle_pts, 0.0)
            seq.add_waveform(nidaq_trig_segment_id)
            seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
            scenario.sequences.append(seq)
            self.arb.upload_scenario(scenario, start_idx=self.start_idxs[-1])
            self.start_idxs.append(self.start_idxs[-1] + len(scenario.scpi_strings()))

        # The last entry is eroneous
        self.start_idxs = self.start_idxs[:-1]
        self.arb.sequence_mode = "SCENARIO"
        self.arb.scenario_advance_mode = "SINGLE"
        self.arb.scenario_start_index = 0
        # self.arb.run()

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("sample", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("state", range(2)))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))
        self.voltage.set_descriptor(descrip)

    def run(self):
        """This is run for each step in a sweep."""
        self.arb.stop()
        self.arb.scenario_start_index = self.start_idxs[self.start_id]
        logger.debug("Now run step #{}.".format(self.start_id+1))
        self.arb.run()
        self.arb.advance()
        self.arb.trigger()
        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        self.voltage.push(buf)
        # Seemingly we need to give the filters some time to catch up here...
        time.sleep(0.02)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))
        self.start_id += 1
        if self.start_id == len(self.start_idxs):
            self.start_id = 0
            logger.warning("Sweep completed. Return to beginning.")

    def shutdown_instruments(self):
        self.keith.current = 0.0e-5
        # self.mag.zero()
        self.arb.stop()
        # self.pspl.output = False
        try:
            self.analog_input.StopTask()
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass

if __name__ == '__main__':
    exp = nTronSwitchingExperiment()
    exp.sample = "CSHE5-C1R3"
    exp.comment = "nTrong Phase Diagram -  P to AP - Interations = 2"
    exp.polarity = 1 # -1: AP to P; 1: P to AP
    exp.iteration = 2
    exp.reset_amplitude = 0.2
    exp.reset_duration = 5e-9
    coarse_ts = np.linspace(1,2,3)*1e-9
    coarse_vs = np.linspace(0.4,0.6,3)
    points    = [coarse_ts, coarse_vs]
    points    = np.array(list(itertools.product(*points)))
    exp.pulse_durations = points[:,0]
    exp.pulse_voltages = points[:,1]
    exp.field.value = -0.0074
    exp.measure_current = 0e-6
    exp.init_instruments()

    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die5-C1R3\CSHE5-C1R3_nTron_P2AP_2016-07-15.h5")
    edges = [(exp.voltage, wr.data)]
    exp.set_graph(edges)


    main_sweep = exp.add_unstructured_sweep([exp.pulse_duration, exp.pulse_voltage], points)
    figs = []
    t1 = time.time()
    for i in range(exp.iteration):
        exp.reset()
        exp.run_sweeps()
        points, mean = sw.load_switching_data(wr.filename)
        figs.append(sw.phase_diagram_mesh(points, mean, title="Iteration={}".format(i)))
        new_points = refine.refine_scalar_field(points, mean, all_points=False,
                                    criterion="integral", threshold = "one_sigma")
        if new_points is None:
            print("No more points can be added.")
            break
        #
        print("Added {} new points.".format(len(new_points)))
        print("Elapsed time: {}".format((time.time()-t1)/60))
        main_sweep.update_values(new_points)
        exp.pulse_durations = new_points[:,0]
        exp.pulse_voltages = new_points[:,1]
        exp.setup_arb()
        time.sleep(3)


    t2 = time.time()
    print("Elapsed time: {} min".format((t2-t1)/60))
    # Shut down
    exp.shutdown_instruments()

    points, mean = sw.load_switching_data(wr.filename)
    sw.phase_diagram_mesh(points, mean)
    # For evaluation of adaptive method, plot the mesh
    # mesh, scale_factors = sw.scaled_Delaunay(points)
    # fig_mesh = sw.phase_diagram_mesh(points, mean, shading='gouraud')
    # plt.triplot(mesh.points[:,0]/scale_factors[0],
    #             mesh.points[:,1]/scale_factors[1], mesh.simplices.copy());

    # plt.show()
