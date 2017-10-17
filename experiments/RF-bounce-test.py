# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import KeysightM8190A, Scenario, Sequence
import datetime, time
import numpy as np
import matplotlib.pyplot as plt

def switching_pulse(amplitude, duration, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)
    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude
    wf = np.append(np.zeros(1<<12), wf)
    wf = np.append(wf, np.zeros(1<<12))
    return wf

def measure_pulse(amplitude, duration, frequency, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)
    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude*np.sin(2.0*np.pi*frequency*np.arange(pulse_points)/sample_rate)
    wf = np.append(np.zeros(1<<13), wf)
    return wf

# Instrument resources
arb   = KeysightM8190A("192.168.5.108")
arb.connect()

arb.set_output(True, channel=2)
arb.set_output(True, channel=1)
arb.sample_freq = 10.0e9
arb.set_waveform_output_mode("WSPEED", channel=1)
arb.set_waveform_output_mode("WSPEED", channel=2)
arb.set_output_route("DC", channel=1)
arb.set_output_route("DC", channel=2)
arb.set_output_complement(False, channel=1)
arb.set_output_complement(False, channel=2)
arb.voltage_amplitude = 1.0
arb.continuous_mode = False
arb.gate_mode = False
arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
arb.set_marker_level_high(1.5, channel=1, marker_type="sync")
arb.abort()
arb.delete_all_waveforms()
arb.reset_sequence_table()

# psuedoRandom waveforms
seg_ids_ch1 = []
seg_ids_ch2 = []

durations  = 100e-9*np.ones(1)
amplitudes = 0.01*np.ones(1) #0.5*np.random.random(1 << 8)

for dur,amp in zip(durations, amplitudes):
    wf      = measure_pulse(amplitude=0.04, duration=150e-9, frequency=100e6)
    wf_data = KeysightM8190A.create_binary_wf_data(wf, sync_mkr=1)
    seg_id  = arb.define_waveform(len(wf_data), channel=1)
    arb.upload_waveform(wf_data, seg_id, channel=1)
    seg_ids_ch1.append(seg_id)

    wf      = switching_pulse(amplitude=0.8, duration=40e-9)
    wf_data = KeysightM8190A.create_binary_wf_data(wf)
    seg_id  = arb.define_waveform(len(wf_data), channel=2)
    arb.upload_waveform(wf_data, seg_id, channel=2)
    seg_ids_ch2.append(seg_id)

settle_pts = int(640*np.ceil(2e-6 * 12e9 / 640))

trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
trig_segment_id = arb.define_waveform(len(trig_wf), channel=2)
arb.upload_waveform(trig_wf, trig_segment_id, channel=2)

dummy_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=0)
dummy_trig_segment_id = arb.define_waveform(len(dummy_trig_wf), channel=1)
arb.upload_waveform(dummy_trig_wf, dummy_trig_segment_id, channel=1)

scenario = Scenario()
seq = Sequence(sequence_loop_ct=64)
for si in seg_ids_ch1:#np.random.choice(seg_ids_ch1, 1024):
    # seq.add_waveform(dummy_trig_segment_id)
    # seq.add_idle(1<<16, 0.0)
    seq.add_waveform(si)
    seq.add_idle(settle_pts, 0.0)
scenario.sequences.append(seq)
arb.upload_scenario(scenario, start_idx=0, channel=1)

scenario = Scenario()
seq = Sequence(sequence_loop_ct=64)
for si in seg_ids_ch2:#np.random.choice(seg_ids_ch2, 1024):
    # seq.add_waveform(trig_segment_id)
    # seq.add_idle(1<<16, 0.0)
    seq.add_waveform(si)
    seq.add_idle(settle_pts, 0.0)
scenario.sequences.append(seq)
arb.upload_scenario(scenario, start_idx=0, channel=2)

arb.set_sequence_mode("SCENARIO", channel=1)
arb.set_scenario_advance_mode("SINGLE", channel=1)
arb.set_scenario_start_index(0, channel=1)
arb.set_sequence_mode("SCENARIO", channel=2)
arb.set_scenario_advance_mode("SINGLE", channel=2)
arb.set_scenario_start_index(0, channel=2)
arb.initiate(channel=1)
arb.initiate(channel=2)

arb.set_scenario_start_index(0, channel=1)
arb.set_scenario_start_index(0, channel=2)
arb.advance()
arb.trigger()
arb.stop()
arb.disconnect()
