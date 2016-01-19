from pycontrol.instruments.keysight import *
from pycontrol.instruments.stanford import SR830, SR865
from pycontrol.instruments.kepco import BOP2020M
from pycontrol.instruments.magnet import Electromagnet
from pycontrol.instruments.hall_probe import HallProbe

from PyDAQmx import *

import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.stats import beta

def waveform(time, delay=1.5e-9, rise_time=150e-12, fall_time=2.0e-9):
    if time<=delay:
        return np.exp(-(time-delay)**2/(2*rise_time**2))
    if time>delay:
        return np.exp(-(time-delay)/fall_time)

def arb_pulse(amplitude, duration, sample_rate=12e9):
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

    #rough guess at reset
    reset_wf    = arb_pulse(0.75, 3.0/12e9)
    wf_data     = M8190A.create_binary_wf_data(reset_wf)
    rst_segment_id  = arb.define_waveform(len(wf_data))
    arb.upload_waveform(wf_data, rst_segment_id)

    segment_ids = []
    amps = np.arange(0.4, 0.80, 0.01)
    # amp = 0.55
    # durations = np.arange(1.0/12.0e9, 1.0e-9, 1.0/12.0e9)
    # for duration in durations:
    duration = 3.0/12e9
    for amp in amps:
        waveform   = arb_pulse(amp, duration)
        wf_data    = M8190A.create_binary_wf_data(waveform)
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    # NI-DAQ trigger waveform
    trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    trig_segment_id = arb.define_waveform(len(trig_wf))
    arb.upload_waveform(trig_wf, trig_segment_id)

    start_idxs = [0]

    rate = 1.25e6/(2**8)

    reps = 1 << 12
    lockin_settle_delay = 100e-6
    lockin_settle_pts = int(640*np.ceil(lockin_settle_delay * 12e9 / 640))

    for si in segment_ids:
        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=int(reps/2))
        #First with reset pulse
        seq.add_waveform(rst_segment_id)
        seq.add_idle(lockin_settle_pts, 0.0)
        seq.add_waveform(trig_segment_id) # Trigger the NIDAQ measurement
        seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
        seq.add_waveform(si) # Apply variable pulse to the sample
        seq.add_idle(lockin_settle_pts, 0.0) # Wait for the measurement to settle
        seq.add_waveform(trig_segment_id) # Trigger the NIDAQ measurement

        #second without
        seq.add_idle(lockin_settle_pts, 0.0)
        seq.add_waveform(trig_segment_id) # Trigger the NIDAQ measurement
        seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
        seq.add_waveform(si) # Apply variable pulse to the sample
        seq.add_idle(lockin_settle_pts, 0.0) # Wait for the measurement to settle
        seq.add_waveform(trig_segment_id) # Trigger the NIDAQ measurement
        seq.add_idle(1 << 14, 0.0) # bonus contiguous memory issue
        scenario.sequences.append(seq)

        arb.upload_scenario(scenario, start_idx=start_idxs[-1])
        start_idxs.append(start_idxs[-1] + len(scenario.scpi_strings()))

    # The last entry is eroneous
    start_idxs = start_idxs[:-1]

    arb.sequence_mode = "SCENARIO"
    arb.scenario_advance_mode = "SINGLE"

    mag.field = -364
    time.sleep(2)

    analog_input = Task()
    read = int32()

    # DAQmx Configure Code
    analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_RSE, -10.0,10.0, DAQmx_Val_Volts, None)
    analog_input.CfgSampClkTiming("/Dev1/PFI0", 20000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, reps)

    # DAQmx Start Code
    analog_input.StartTask()

    buffers = np.empty((len(segment_ids), 2*reps))

    for ct, idx in enumerate(tqdm(start_idxs)):
        arb.stop()
        arb.scenario_start_index = idx
        arb.run()
        arb.trigger()
        analog_input.ReadAnalogF64(2*reps, -1, DAQmx_Val_GroupByChannel, buffers[ct], 2*reps, byref(read), None)

    analog_input.StopTask()
    arb.stop()

    mag.field = 0
    bop.current = 0
