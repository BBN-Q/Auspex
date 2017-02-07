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

from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import OutputConnector
from auspex.filters.io import WriteToHDF5
from auspex.filters.plot import Plotter
from auspex.log import logger

from PyDAQmx import *

import itertools
import numpy as np
import pandas as pd
import asyncio
import time, sys
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import beta

import analysis.switching as sw
from analysis.h5shell import h5shell

from auspex.log import logger
# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

def arb_voltage_lookup(arb_calib="calibration/AWG_20160718.csv",
                        midpoint_calib="calibration/midpoint_20160718.csv"):
    df_midpoint = pd.read_csv(midpoint_calib, sep=",")
    df_arb = pd.read_csv(arb_calib, sep=",")
    midpoint_lookup = interp1d(df_midpoint["Sample Voltage"],df_midpoint["Midpoint Voltage"])
    arb_control_lookup = interp1d(df_arb["Midpoint Voltage"],df_arb["Control Voltage"])
    sample_volts = []
    control_volts = []
    for volt in df_midpoint['Sample Voltage']:
        mid_volt = midpoint_lookup(volt)
        if (mid_volt > min(df_arb['Midpoint Voltage'])) and (mid_volt < max(df_arb['Midpoint Voltage'])):
            sample_volts.append(volt)
            control_volts.append(arb_control_lookup(mid_volt))
    return interp1d(sample_volts, control_volts)

class SWRExperiment(Experiment):
    """ Experiment class for Switching probability measurment
    Determine switching probability for V << V0
    with varying V (and durations?)
    """

    field          = FloatParameter(default=0.0, unit="T")
    pulse_duration = FloatParameter(default=1.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=0.1, unit="V")
    daq_buffer     = OutputConnector()

    attempts        = 1 << 10
    settle_delay    = 50e-6
    measure_current = 3.0e-6
    samps_per_trig  = 5

    polarity        = 1

    min_daq_voltage = 0.0
    max_daq_voltage = 0.4

    reset_amplitude = 0.1
    reset_duration  = 5.0e-9

    mag   = AMI430("192.168.5.109")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")
    pspl  = Picosecond10070A("GPIB0::24::INSTR")
    arb   = KeysightM8190A("192.168.5.108")
    keith = Keithley2400("GPIB0::25::INSTR")

    def init_streams(self):
        self.daq_buffer.add_axis(DataAxis("samples", range(self.samps_per_trig)))
        self.daq_buffer.add_axis(DataAxis("state", range(2)))
        self.daq_buffer.add_axis(DataAxis("attempts", range(self.attempts)))

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
        self.setup_arb(self.pulse_voltage.value)

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
        self.pulse_voltage.assign_method(self.setup_arb)

    def setup_arb(self,volt):
        def arb_pulse(amplitude, duration, sample_rate=12e9):
            arb_voltage = arb_voltage_lookup()
            pulse_points = int(duration*sample_rate)
            if pulse_points < 320:
                wf = np.zeros(320)
            else:
                wf = np.zeros(64*np.ceil(pulse_points/64.0))
            wf[:pulse_points] = np.sign(amplitude)*arb_voltage(abs(amplitude))
            return wf

        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        # Reset waveform
        reset_wf    = arb_pulse(-self.polarity*self.reset_amplitude, self.reset_duration)
        wf_data     = KeysightM8190A.create_binary_wf_data(reset_wf)
        rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, rst_segment_id)

        # Switching waveform
        switch_wf    = arb_pulse(self.polarity*volt, self.pulse_duration.value)
        wf_data     = KeysightM8190A.create_binary_wf_data(switch_wf)
        sw_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, sw_segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
        nidaq_trig_segment_id = self.arb.define_waveform(len(nidaq_trig_wf))
        self.arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

        settle_pts = int(640*np.ceil(self.settle_delay * 12e9 / 640))

        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=int(self.attempts))
        #First try with reset flipping pulse
        seq.add_waveform(rst_segment_id)
        seq.add_idle(settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
        seq.add_waveform(sw_segment_id)
        seq.add_idle(settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
        scenario.sequences.append(seq)
        self.arb.upload_scenario(scenario, start_idx=0)
        self.arb.sequence_mode = "SCENARIO"
        self.arb.scenario_advance_mode = "REPEAT"
        self.arb.scenario_start_index = 0
        self.arb.run()

    async def run(self):
        # Keep track of the previous values
        logger.debug("Waiting for filters.")
        await asyncio.sleep(1.0)
        self.arb.advance()
        self.arb.trigger()
        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        await self.daq_buffer.push(buf)
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.002)
        logger.debug("Stream has filled {} of {} points".format(self.daq_buffer.points_taken, self.daq_buffer.num_points() ))

    def shutdown_instruments(self):
        self.keith.current = 0.0e-5
        # self.mag.zero()
        self.arb.stop()
        try:
            self.analog_input.StopTask()
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass

def data_at_volt(fname, volt):
    """ Extract datasets in file fname that have pulse_voltage == volt """
    with h5shell(fname, 'r') as f:
        dsets = [f[k] for k in f.ls('-d')]
        dsets = [dset for dset in dsets if abs(float(dset.attrs['pulse_voltage'])-volt)/volt<0.01]
        data_mean = [np.mean(dset.value, axis=-1) for dset in dsets]
    return np.concatenate(data_mean,axis=0)

# def stop_measure(data, **kwargs):
#     """ Determine whether we should stop the measurement at a given pulse voltage """
#     results = sw.switching_BER(data, **kwargs)
#     limit = results[1]
#     ci95 = results[3]
#     return limit > max(ci95)

def stop_measure(data, **kwargs):
    """ Determine whether we should stop the measurement at a given pulse voltage """
    counts, start_stt = sw.count_matrices(data, multiple=False,**kwargs)
    count_mat = counts[0]
    switched_stt = 1 - start_stt
    # mean_not = beta.mean(1+count_mat[start_stt,start_stt],1+count_mat[start_stt,switched_stt])
    limit = beta.mean(1+count_mat[start_stt,switched_stt]+count_mat[start_stt,start_stt], 1)
    ci95 = beta.interval(0.95, 1+count_mat[start_stt,start_stt],1+count_mat[start_stt,switched_stt])
    return limit > max(ci95)

def load_SWR_data(fname):
    with h5shell(fname, 'r') as f:
        dsets = [f[k] for k in f.ls('-d')]
        volts = [float(dset.attrs['pulse_voltage']) for dset in dsets]

    unique_volts = []
    for v in sorted(volts):
        if len(unique_volts) == 0:
            unique_volts.append(v)
        else:
            check = [abs((uv-v)/v)<0.01 for uv in unique_volts]
            if not np.any(check):
                unique_volts.append(v)
    data = [data_at_volt(fname,volt) for volt in unique_volts]
    return np.array(unique_volts), np.array(data)

def plot_SWR(volts, results):
    mean = []; limit = []; ci68 = []; ci95 = []
    for datum in results:
        mean.append(datum[0])
        limit.append(datum[1])
        ci68.append(datum[2])
        ci95.append(datum[3])
    mean = np.array(mean)
    limit = np.array(limit)
    fig = plt.figure()
    plt.semilogy(volts, mean, '-o')
    plt.semilogy(volts, 1-limit, linestyle="--")
    plt.fill_between(volts, [ci[0] for ci in ci68], [ci[1] for ci in ci68],  alpha=0.2, edgecolor="none")
    plt.fill_between(volts, [ci[0] for ci in ci95], [ci[1] for ci in ci95],  alpha=0.2, edgecolor="none")
    plt.ylabel("Switching Probability", size=14)
    plt.xlabel("Pulse Voltage (V)", size=14)
    return fig

if __name__ == '__main__':
    exp = SWRExperiment()
    exp.sample = "CSHE5 - C2R3"
    exp.comment = "Switching Probability - AP to P - 5ns"
    exp.polarity = 1 # 1: AP to P; -1: P to AP
    exp.field.value = 0.0081
    exp.attempts = 1 << 11
    exp.pulse_duration.value = 5e-9 # Fixed
    exp.reset_amplitude = 0.7
    exp.reset_duration = 5e-9
    exp.init_instruments()

    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die5-C2R3\\test\CSHE5-C2R3-AP2P_2016-07-27_SWR_5ns.h5")
    edges = [(exp.daq_buffer, wr.data)]
    exp.set_graph(edges)

    V0 = 0.5
    voltages_list = V0*np.linspace(1.0,0.2,5)

    t1 = [] # Keep track of time
    t2 = []
    max_points = 1<<13
    finish = False
    for volt in voltages_list:
        if finish:
            print("Reached maximum points. Exit.")
            break
        print("=========================")
        print("Now at: {}.".format(volt))
        t1.append(time.time())
        exp.pulse_voltage.value = volt
        forward = False
        count = 0
        while not forward:
            count = count + 1
            print("Measurement count = %d" %count)
            exp.init_streams()
            exp.reset()
            exp.run_loop()
            time.sleep(1) # Wait for filters
            data = data_at_volt(wr.filename, volt)
            finish = data.size >= max_points*2
            forward = stop_measure(data, start_state=1, threshold=0.36) or finish
        t2.append(time.time())
        print("Done one series. Elapsed time: {} min".format((t2[-1]-t1[-1])/60))
        time.sleep(2)

    exp.shutdown_instruments()
    # Plot data
    volts, data = load_SWR_data(wr.filename)
    results = [sw.switching_BER(datum, start_state=1, threshold=0.36) for datum in data]
    fig = plot_SWR(volts, results)
