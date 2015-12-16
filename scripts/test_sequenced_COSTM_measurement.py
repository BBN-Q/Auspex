from pycontrol.instruments.keysight import *
from pycontrol.instruments.stanford import SR830, SR865
from pycontrol.instruments.kepco import BOP2020M
from pycontrol.instruments.magnet import Electromagnet
from pycontrol.instruments.hall_probe import HallProbe
import numpy as np
import time

def waveform(time, delay=1.5e-9, rise_time=150e-12, fall_time=2.0e-9):
    if time<=delay:
        return np.exp(-(time-delay)**2/(2*rise_time**2))
    if time>delay:
        return np.exp(-(time-delay)/fall_time)

def pulse(amplitude, duration, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude
    return wf

if __name__ == '__main__':
    arb  = M8190A("Test Arb", "192.168.5.108")
    # lock = SR830("Lockin Amplifier", "GPIB0::9::INSTR")
    lock = SR865("Lockin Amplifier", "USB0::0xB506::0x2000::002638::INSTR")
    bop  = BOP2020M("Kepco Power Supply", "GPIB0::1::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)

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
    arb.voltage_amplitude = 1.0

    arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
    arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

    arb.continuous_mode = False
    arb.gate_mode = False

    # times = np.arange(0, 42.6e-9, 1/12e9)
    # fall_times = np.arange(1.0e-9, 10.1e-9, 1.0e-9)

    segment_ids = []

    for amp in np.arange(0.90, 1.00, 0.05):
        waveform   = pulse(amp, 0.5e-9)
        wf_data    = M8190A.create_binary_wf_data(waveform)
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    # Add the blank waveform
    waveform_blank   = pulse(amp, 0.5e-9)
    wf_data_blank    = M8190A.create_binary_wf_data(waveform_blank)
    blank_segment_id = arb.define_waveform(len(wf_data_blank))
    arb.upload_waveform(wf_data_blank, blank_segment_id)

    # Trigger waveform
    trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    trig_segment_id = arb.define_waveform(len(trig_wf))
    arb.upload_waveform(trig_wf, trig_segment_id)

    start_idxs = [0]

    rate = 1.25e6/(2**8)

    for si in segment_ids:
        scenario = Scenario()

        # Add the trigger pulse
        seq_trig = Sequence()
        seq_trig.add_waveform(trig_segment_id)
        scenario.sequences.append(seq_trig)

        # Add the dummy pulses
        seq = Sequence(sequence_loop_ct=128)
        seq.add_waveform(blank_segment_id)
        # seq.add_idle(1 << 24, 0.0) # Lockin TC
        # seq.add_waveform(trig_segment_id)
        # seq.add_idle(1 << 25, 0.0) # Lockin sample rate
        seq.add_idle(int(12e9/rate - len(wf_data_blank)))
        scenario.sequences.append(seq)

        # Add the real pulses
        seq = Sequence(sequence_loop_ct=128)
        seq.add_waveform(si)
        # seq.add_idle(1 << 24, 0.0) # Lockin TC
        # seq.add_waveform(trig_segment_id)
        # seq.add_idle(1 << 25, 0.0) # Lockin sample rate
        seq.add_idle(int(12e9/rate - len(wf_data)))
        scenario.sequences.append(seq)

        arb.upload_scenario(scenario, start_idx=start_idxs[-1])
        start_idxs.append(start_idxs[-1] + len(scenario.scpi_strings()))

    # The last entry is eroneous
    start_idxs = start_idxs[:-1]

    arb.sequence_mode = "SCENARIO"
    arb.scenario_advance_mode = "SINGLE"
    #
    # lock.sample_rate = "Trigger"
    # lock.buffer_mode = "SHOT"
    # lock.sample_rate = "Trigger"
    # lock.buffer_trigger_mode = False

    # 865

    lock.capture_rate   = 1.25e6/(2**8)
    lock.capture_length = 1024
    lock.capture_quants = "RT"
    # lock.capture_start(hw_trigger=True)

    buffers = np.empty((len(segment_ids), 1024))
    mag.field = -364
    time.sleep(4)

    for i, idx in enumerate(start_idxs):
        # lock.buffer_reset()
        # lock.buffer_start()
        lock.capture_start(hw_trigger=True)
        time.sleep(0.1)
        arb.stop()
        arb.scenario_start_index = idx
        arb.run()
        arb.trigger()
        # Pause for complete acquisition
        # while lock.buffer_points < 512:
            # time.sleep(0.1)
        while not lock.capture_done:
            time.sleep(0.1)
        lock.capture_stop()
        buf = lock.get_capture(1)[::2]
        # buf = lock.get_buffer(1)
        buffers[i] = buf

    mag.field = 0
    bop.current = 0

    # lock.capture_stop()
    # lock.buffer_pause()
