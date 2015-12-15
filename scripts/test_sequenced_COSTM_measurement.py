from pycontrol.instruments.keysight import *
from pycontrol.instruments.stanford import SR830
import numpy as np
import time

def waveform(time, delay=1.5e-9, rise_time=150e-12, fall_time=2.0e-9):
    if time<=delay:
        return np.exp(-(time-delay)**2/(2*rise_time**2))
    if time>delay:
        return np.exp(-(time-delay)/fall_time)

if __name__ == '__main__':
    arb = M8190A("Test Arb", "192.168.5.108")
    print(arb.interface.query("*IDN?"))
    # arb.interface.write("*RST")
    # time.sleep(2)
    # arb.interface.query("*OPC?")

    arb.set_output(True, channel=1)
    arb.set_output(False, channel=2)
    arb.sample_freq = 12.0e9
    arb.waveform_output_mode = "WSPEED"

    arb.abort()
    arb.delete_all_waveforms()
    arb.interface.write(":STAB:RES")

    time.sleep(1) #otherwise first define_waveform can fail after a reset

    arb.set_output_route("DC", channel=1)
    arb.voltage_amplitude = 0.15

    arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
    arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

    arb.continuous_mode = False
    arb.gate_mode = False

    times = np.arange(0, 42.6e-9, 1/12e9)
    fall_times = np.arange(1.0e-9, 10.1e-9, 1.0e-9)

    segment_ids = []

    # for ft in fall_times:
    #     volts   = [waveform(t, rise_time=0.00e-9, fall_time=ft) for t in times]
    for amp in np.arange(0.1, 0.9, 0.1):
        volts = np.concatenate((np.linspace(0, amp, 12032), np.zeros(2048)))
        sync_mkr = np.zeros(len(volts), dtype=np.int16)
        wf_data = M8190A.create_binary_wf_data(np.array(volts))
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    for amp in np.arange(-0.1, -0.9, -0.1):
        volts = np.concatenate((np.linspace(0, amp, 12032), np.zeros(2048)))
        sync_mkr = np.zeros(len(volts), dtype=np.int16)
        wf_data = M8190A.create_binary_wf_data(np.array(volts))
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    trig_segment_id = arb.define_waveform(len(trig_wf))
    arb.upload_waveform(trig_wf, trig_segment_id)

    scenario = Scenario()
    start_idx = 0
    for si,si2 in zip(segment_ids[:8], segment_ids[8:]):
        seq = Sequence(sequence_loop_ct=128)
        seq.add_waveform(si)
        seq.add_idle(16384, 0.0)
        seq.add_waveform(trig_segment_id)
        seq.add_idle(1 << 25, 0.0)
        scenario.sequences.append(seq)
        seq = Sequence(sequence_loop_ct=128)
        seq.add_waveform(si2)
        seq.add_idle(16384, 0.0)
        seq.add_waveform(trig_segment_id)
        seq.add_idle(1 << 25, 0.0)
        scenario.sequences.append(seq)

    arb.upload_scenario(scenario, start_idx=start_idx)
    start_idx += len(scenario.scpi_strings())

    scenario = Scenario()
    for si,si2 in zip(segment_ids[:8], segment_ids[8:]):
        seq = Sequence(sequence_loop_ct=128)
        seq.add_waveform(si2)
        seq.add_idle(16384, 0.0)
        seq.add_waveform(trig_segment_id)
        seq.add_idle(1 << 25, 0.0)
        scenario.sequences.append(seq)
        seq = Sequence(sequence_loop_ct=128)
        seq.add_waveform(si)
        seq.add_idle(16384, 0.0)
        seq.add_waveform(trig_segment_id)
        seq.add_idle(1 << 25, 0.0)
        scenario.sequences.append(seq)

    arb.upload_scenario(scenario, start_idx=start_idx)

    arb.sequence_mode = "SCENARIO"
    arb.scenario_advance_mode = "SINGLE"

    lock = SR830("Lockin Amplifier", "GPIB0::9::INSTR")
    lock.sample_rate = "Trigger"
    lock.buffer_mode = "SHOT"
    lock.sample_rate = "Trigger"
    lock.buffer_trigger_mode = False
    lock.buffer_reset()
    lock.buffer_start()

    time.sleep(0.1)

    arb.run()
    arb.trigger()

    # lock.buffer_pause()
