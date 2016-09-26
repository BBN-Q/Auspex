# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments.keysight import *
import numpy as np

def waveform(time, delay=1.5e-9, rise_time=150e-12, fall_time=2.0e-9):
    if time<=delay:
        return np.exp(-(time-delay)**2/(2*rise_time**2))
    if time>delay:
        return np.exp(-(time-delay)/fall_time)

if __name__ == '__main__':
    arb = M8190A("192.168.5.108")
    print(arb.interface.query("*IDN?"))

    arb.set_output(True, channel=1)
    arb.set_output(False, channel=2)
    arb.sample_freq = 12.0e9
    arb.waveform_output_mode = "WSPEED"

    #
    #
    # sync_mkr = np.zeros(len(volts), dtype=np.int16)
    # # samp_mkr = np.zeros(len(volts), dtype=np.int16)
    # # samp_mkr[0:128] = 1
    # sync_mkr[320:] = 1

    times = np.arange(0, 42.6e-9, 1/12e9)
    fall_times = np.arange(1.0e-9, 10.1e-9, 1.0e-9)

    arb.abort()
    arb.delete_all_waveforms()
    arb.reset_sequence_table()

    segment_ids = []

    # for ft in fall_times:
    #     volts   = [waveform(t, rise_time=0.00e-9, fall_time=ft) for t in times]
    for amp in np.arange(0.1, 0.9, 0.1):
        volts = np.concatenate((np.linspace(0, amp, 12032), np.zeros(2048)))
        sync_mkr = np.zeros(len(volts), dtype=np.int16)
        sync_mkr[:320] = 1
        wf_data = arb.create_binary_wf_data(np.array(volts), sync_mkr=sync_mkr)
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    for amp in np.arange(-0.1, -0.9, -0.1):
        volts = np.concatenate((np.linspace(0, amp, 12032), np.zeros(2048)))
        sync_mkr = np.zeros(len(volts), dtype=np.int16)
        sync_mkr[:320] = 1
        wf_data = arb.create_binary_wf_data(np.array(volts), sync_mkr=sync_mkr)
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    scenario = Scenario()
    start_idx = 0
    for si,si2 in zip(segment_ids[:8], segment_ids[8:]):
        seq = Sequence(sequence_loop_ct=3)
        seq.add_waveform(si)
        seq.add_idle(6400, 0.0)
        seq.add_waveform(si2)
        seq.add_idle(12800, 0.0)
        scenario.sequences.append(seq)
        seq = Sequence(sequence_loop_ct=3)
        seq.add_waveform(si2)
        seq.add_idle(6400, 0.0)
        seq.add_waveform(si)
        seq.add_idle(12800, 0.0)
        scenario.sequences.append(seq)

    arb.upload_scenario(scenario, start_idx=start_idx)
    start_idx += len(scenario.scpi_strings())

    scenario = Scenario()
    for si,si2 in zip(segment_ids[:8], segment_ids[8:]):
        seq = Sequence(sequence_loop_ct=3)
        seq.add_waveform(si2)
        seq.add_idle(6400, 0.0)
        seq.add_waveform(si)
        seq.add_idle(12800, 0.0)
        scenario.sequences.append(seq)
        seq = Sequence(sequence_loop_ct=3)
        seq.add_waveform(si)
        seq.add_idle(6400, 0.0)
        seq.add_waveform(si2)
        seq.add_idle(12800, 0.0)
        scenario.sequences.append(seq)

    arb.upload_scenario(scenario, start_idx=start_idx)


    #
    # idx = 0
    # arb.interface.write('STAB1:DATA {:d}, {:d}, 10, 1, {:d}, 0, {:d}'.format(idx, 0x11000000, 1, 0xffffffff) )
    # idx += 1
    # arb.interface.write('STAB1:DATA {:d}, {:d}, 1, 0, 0, 6400, 0'.format(idx, 0xc0000000) )
    # idx += 1
    # for si in segment_ids[1:-1]:
    #     arb.interface.write('STAB1:DATA {:d}, {:d}, 10, 1, {:d}, 0, {:d}'.format(idx, 0x11000000, si, 0xffffffff) )
    #     idx += 1
    #     arb.interface.write('STAB1:DATA {:d}, {:d}, 1, 0, 0, 6400, 0'.format(idx, 0xc0000000) )
    #     idx += 1
    # arb.interface.write('STAB1:DATA {:d}, {:d}, 10, 1, {:d}, 0, {:d}'.format(idx, 0x11000000, len(segment_ids), 0xffffffff) )
    # idx += 1
    # arb.interface.write('STAB1:DATA {:d}, {:d}, 1, 0, 0, 6400, 0'.format(idx, 0xe0000000) )

    # arb.select_waveform(segment_id)
    # arb.initiate()

    # segment_id = 2
    # wf = arb.create_binary_wf_data(np.array(volts), sync_mkr=sync_mkr, samp_mkr=samp_mkr)
    # arb.use_waveform(wf)
