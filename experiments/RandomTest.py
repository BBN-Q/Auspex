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

def switching_pulse(amplitude, duration, offset_amp=0.0, offset_dur=0.0, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)
    pulse_points_offset = int(duration*sample_rate)

    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude
    return wf


# Instrument resources
arb   = KeysightM8190A("192.168.5.108")
arb.connect()

arb.set_output(True, channel=2)
arb.set_output(True, channel=1)
arb.sample_freq = 12.0e9
arb.set_waveform_output_mode("WSPEED", channel=1)
arb.set_waveform_output_mode("WSPEED", channel=2)
arb.set_output_route("DC", channel=1)
arb.set_output_route("DC", channel=2)
arb.set_output_complement(False, channel=1)
arb.set_output_complement(False, channel=2)
arb.voltage_amplitude = 1.0
arb.continuous_mode = False
arb.gate_mode = False

arb.abort()
arb.delete_all_waveforms()
arb.reset_sequence_table()

# psuedoRandom waveforms
seg_ids_ch1 = []
seg_ids_ch2 = []

durations  = 5e-9*np.ones(1 << 8)
amplitudes = np.ones(1<<8) #0.5*np.random.random(1 << 8)

for dur,amp in zip(durations, amplitudes):
    wf      = switching_pulse(amplitude=amp, duration=240e-12)
    wf_data = KeysightM8190A.create_binary_wf_data(wf)

    seg_id  = arb.define_waveform(len(wf_data), channel=1)
    arb.upload_waveform(wf_data, seg_id, channel=1)
    seg_ids_ch1.append(seg_id)

    wf      = switching_pulse(amplitude=amp, duration=2*dur)
    wf_data = KeysightM8190A.create_binary_wf_data(wf)
    seg_id  = arb.define_waveform(len(wf_data), channel=2)
    arb.upload_waveform(wf_data, seg_id, channel=2)
    seg_ids_ch2.append(seg_id)

start_idxs_ch1 = [0]
start_id_ch1 = 0
for jj in range(10):
    scenario = Scenario()
    seq = Sequence(sequence_loop_ct=1)
    for si in seg_ids_ch1:#np.random.choice(seg_ids_ch1, 1024):
        seq.add_waveform(si)
    seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
    scenario.sequences.append(seq)
    arb.upload_scenario(scenario, start_idx=start_idxs_ch1[-1], channel=1)
    start_idxs_ch1.append(start_idxs_ch1[-1] + len(scenario.scpi_strings()))

start_idxs_ch2 = [0]
start_id_ch2 = 0
for jj in range(10):
    scenario = Scenario()
    seq = Sequence(sequence_loop_ct=1)
    for si in seg_ids_ch2:#np.random.choice(seg_ids_ch2, 1024):
        seq.add_waveform(si)
    seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
    scenario.sequences.append(seq)
    arb.upload_scenario(scenario, start_idx=start_idxs_ch2[-1], channel=2)
    start_idxs_ch2.append(start_idxs_ch2[-1] + len(scenario.scpi_strings()))

# The last entry is eroneous
start_idxs_ch1 = start_idxs_ch1[:-1]
start_idxs_ch2 = start_idxs_ch2[:-1]
arb.set_sequence_mode("SCENARIO", channel=1)
arb.set_scenario_advance_mode("SINGLE", channel=1)
arb.set_scenario_start_index(0, channel=1)
arb.set_sequence_mode("SCENARIO", channel=2)
arb.set_scenario_advance_mode("SINGLE", channel=2)
arb.set_scenario_start_index(0, channel=2)
arb.initiate(channel=1)
arb.initiate(channel=2)

for jj in range(10):
    arb.scenario_start_index = start_idxs_ch1[start_id_ch1]
    print(f"Now running step #{jj+1}.")
    # arb.run()
    arb.advance()
    arb.trigger()

    time.sleep(0.02)
    start_id_ch1 += 1
    if start_id_ch1 == len(start_idxs_ch1):
        start_id_ch1 = 0

    start_id_ch2 += 1
    if start_id_ch2 == len(start_idxs_ch2):
        start_id_ch2 = 0

arb.stop()
arb.disconnect()
