# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.keysight import *
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430
from pycontrol.instruments.rfmd import Attenuator

from pycontrol.experiment import FloatParameter, IntParameter, Experiment
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print
from pycontrol.filters.io import WriteToHDF5

from PyDAQmx import *

import itertools
import numpy as np
import asyncio
import time, sys
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

import analysis.switching as sw
from adapt import refine

from pycontrol.log import logger

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

def nTron_voltage_lookup(nTron_calib="calibration/nTron_20160718.csv",
                        midpoint_calib="calibration/midpoint_20160718.csv"):
    df_midpoint = pd.read_csv(midpoint_calib, sep=",")
    df_nTron = pd.read_csv(nTron_calib, sep=",")
    midpoint_lookup = interp1d(df_midpoint["Sample Voltage"],df_midpoint["Midpoint Voltage"])
    nTron_control_lookup = interp1d(df_nTron["Midpoint Voltage"],df_nTron["Control Voltage"])
    sample_volts = []
    control_volts = []
    for volt in df_midpoint['Sample Voltage']:
        mid_volt = midpoint_lookup(volt)
        if (mid_volt > min(df_nTron['Midpoint Voltage'])) and (mid_volt < max(df_nTron['Midpoint Voltage'])):
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

class nTronBERExperiment(Experiment):

    # Sample information
    sample         = "CSHE"
    comment        = "Bit Error Rate with nTron pulses"
    # Parameters
    field          = FloatParameter(default=0.0, unit="T")
    nTron_voltage  = FloatParameter(default=0.2, unit="V")
    nTron_duration = FloatParameter(default=1e-9, unit="s")
    attempts       = IntParameter(default=1 << 10)

    # Constants (set with attribute access if you want to change these!)
    settle_delay    = 200e-6
    measure_current = 3.0e-6
    samps_per_trig  = 5

    polarity        = 1

    min_daq_voltage = 0.0
    max_daq_voltage = 0.4

    reset_amplitude = 0.2
    reset_duration  = 5.0e-9

    # Things coming back
    daq_buffer     = OutputConnector()

    # Instrument resources
    mag   = AMI430("192.168.5.109")
    # lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")
    # pspl  = Picosecond10070A("GPIB0::24::INSTR")
    # atten = Attenuator("calibration/RFSA2113SB_HPD_20160706.csv", lock.set_ao2, lock.set_ao3)
    arb   = M8190A("192.168.5.108")
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

        self.nTron_control_voltage = nTron_voltage_lookup()
        self.setup_arb(self.nTron_voltage.value)

        # Assign methods
        self.field.assign_method(self.mag.set_field)
        self.nTron_voltage.assign_method(self.setup_arb)

    def setup_arb(self, vpeak):
        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        reset_wf    = arb_pulse(-self.polarity*self.reset_amplitude, self.reset_duration)
        wf_data     = M8190A.create_binary_wf_data(reset_wf)
        rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, rst_segment_id)

        no_reset_wf = arb_pulse(0.0, 3.0/12e9)
        wf_data     = M8190A.create_binary_wf_data(no_reset_wf)
        no_rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, no_rst_segment_id)

        # nTron waveforms
        volt = self.polarity*self.nTron_control_voltage(vpeak)
        logger.debug("Set nTron pulse: {}V -> AWG {}V, {}s".format(vpeak,volt,self.nTron_duration.value))
        ntron_wf    = ntron_pulse(amplitude=volt, fall_time=self.nTron_duration.value)
        wf_data     = M8190A.create_binary_wf_data(ntron_wf)
        ntron_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, ntron_segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
        nidaq_trig_segment_id = self.arb.define_waveform(len(nidaq_trig_wf))
        self.arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

        settle_pts = int(640*np.ceil(self.settle_delay * 12e9 / 640))

        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=int(self.attempts.value))
        seq.add_waveform(rst_segment_id)
        seq.add_idle(settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
        # seq.add_waveform(pspl_trig_segment_id)
        seq.add_waveform(ntron_segment_id)
        seq.add_idle(settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
        scenario.sequences.append(seq)
        self.arb.upload_scenario(scenario, start_idx=0)

        self.arb.sequence_mode = "SCENARIO"
        self.arb.scenario_advance_mode = "REPEAT"
        self.arb.scenario_start_index = 0
        self.arb.run()

        # ===================
        #   Setup the NIDAQ
        # ===================

        self.analog_input = Task()
        self.read = int32()
        self.buf_points = 2*self.samps_per_trig*self.attempts.value
        self.analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff,
                                            self.min_daq_voltage, self.max_daq_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , self.samps_per_trig)
        self.analog_input.CfgInputBuffer(self.buf_points)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        self.analog_input.SetStartTrigRetriggerable(1)
        self.analog_input.StartTask()


    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("state", range(2)))
        descrip.add_axis(DataAxis("attempts", range(self.attempts.value)))
        self.daq_buffer.set_descriptor(descrip)

    async def run(self):
        """This is run for each step in a sweep."""
        self.arb.advance()
        self.arb.trigger()
        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        await self.daq_buffer.push(buf)
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
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

if __name__ == '__main__':
    exp = nTronBERExperiment()
    exp.sample = "CSHE5 - C1R3"
    exp.comment = "nTron Bit Error Rate - P to AP - 10ns"
    exp.polarity = 1 # -1: AP to P; 1: P to AP
    exp.field.value = -0.0074
    exp.nTron_duration.value = 10e-9 # Fixed
    exp.init_instruments()

    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die5-C1R3\CSHE5-C1R3-P2AP_2016-07-20_BER_10ns.h5")
    edges = [(exp.daq_buffer, wr.data)]
    exp.set_graph(edges)

    attempts_list = [1 << int(x) for x in np.linspace(12,16,5)]
    # attempts_list = [int(6e6), int(6e6)]
    voltages_list = np.linspace(0.6,1.0,5)
    t1 = [] # Keep track of time
    t2 = []
    for att, vol in zip(attempts_list, voltages_list):
        print("=========================")
        print("Now at ({},{}).".format(att,vol))
        t1.append(time.time())
        exp.attempts.value = att
        exp.nTron_voltage.value = vol
        exp.init_streams()
        exp.reset()
        exp.run_loop()
        t2.append(time.time())
        print("Elapsed time: {}".format((t2[-1]-t1[-1])/60))
        time.sleep(3)

    # Plot data
    data_mean = sw.load_BER_data(wr.filename)
    fig = sw.plot_BER(voltages_list, data_mean, start_state=0)
