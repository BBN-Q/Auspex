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

def ntron_pulse(amplitude=1.0, rise_time=80e-12, hold_time=320e-12, fall_time=1.0e-9, sample_rate=12e9):
    delay    = 2.0e-9 # Wait a few TCs for the rising edge
    duration = delay + hold_time + 6.0*fall_time # Wait 6 TCs for the slow decay
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        duration = 320/sample_rate
        times = np.arange(0, duration, 1/sample_rate)
    else:
        pulse_points = 64*np.ceil(pulse_points/64.0)
        duration = pulse_points/sample_rate
        times = np.arange(0, duration, 1/sample_rate)

    rise_mask = np.less(times, delay)
    hold_mask = np.less(times, delay + hold_time)*np.greater_equal(times, delay)
    fall_mask = np.greater_equal(times, delay + hold_time)

    wf  = rise_mask*np.exp((times-delay)/rise_time)
    wf += hold_mask
    wf += fall_mask*np.exp(-(times-delay-hold_time)/fall_time)

    return amplitude*wf

def arb_pulse(amplitude, duration, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude
    return wf

if __name__ == '__main__':
    arb  = M8190A("192.168.5.108")
    lock = SR865("USB0::0xB506::0x2000::002638::INSTR")
    bop  = BOP2020M("GPIB0::1::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)

    print(arb.interface.query("*IDN?"))

    arb.set_output(True, channel=1)
    arb.set_output(False, channel=2)
    arb.sample_freq = 12.0e9
    arb.waveform_output_mode = "WSPEED"

    arb.abort()
    arb.delete_all_waveforms()
    arb.reset_sequence_table()

    time.sleep(1) #otherwise first define_waveform can fail after a reset

    arb.set_output_route("DC", channel=1)
    arb.voltage_amplitude = 1.0

    arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
    arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

    arb.continuous_mode = False
    arb.gate_mode = False

    segment_ids = []
    amplitudes = np.arange(0.43, 0.95, 0.02)
    fall_times = np.arange(0.50e-9, 1.50e-9, 0.05e-9)
    for fall_time in fall_times:
        for amplitude in amplitudes:
            waveform   = ntron_pulse(amplitude=amplitude, fall_time=fall_time)
            wf_data    = M8190A.create_binary_wf_data(waveform)
            segment_id = arb.define_waveform(len(wf_data))
            segment_ids.append(segment_id)
            arb.upload_waveform(wf_data, segment_id)

    # Reset waveform
    reset_wf    = arb_pulse(0.513, 4.0/12e9)
    wf_data     = M8190A.create_binary_wf_data(reset_wf)
    rst_segment_id  = arb.define_waveform(len(wf_data))
    arb.upload_waveform(wf_data, rst_segment_id)

    # NIDAQ Trigger waveform
    nidaq_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    nidaq_trig_segment_id = arb.define_waveform(len(nidaq_trig_wf))
    arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

    start_idxs = [0]

    rate = 1.25e6/(2**8)

    reps = 1 << 8
    lockin_settle_delay = 52e-6
    lockin_settle_pts = int(640*np.ceil(lockin_settle_delay * 12e9 / 640))

    for si in segment_ids:
        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=reps)

        seq.add_waveform(rst_segment_id) # Reset first
        seq.add_idle(lockin_settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay

        seq.add_waveform(si) # Apply nTron pulse to the sample
        seq.add_idle(lockin_settle_pts, 0.0) # Wait for the measurement to settle
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delaytrig_segment_id) # Trigger the NIDAQ measurement

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
    analog_input.CfgSampClkTiming("/Dev1/PFI0", 20000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 2*reps)

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

    #Get an idea of SNR
    #Cluster all the data into two based with starting point based on edges
    all_vals = buffers.flatten()
    all_vals.resize((all_vals.size,1))
    init_guess = np.array([np.min(all_vals), np.max(all_vals)])
    init_guess.resize((2,1))
    clusterer = KMeans(init=init_guess, n_clusters=2)
    state = clusterer.fit_predict(all_vals)

    #Approximate SNR from centre distance and variance
    std0 = np.std(all_vals[state == 0])
    std1 = np.std(all_vals[state == 1])
    mean_std = 0.5*(std0 + std1)
    centre0 = clusterer.cluster_centers_[0,0]
    centre1 = clusterer.cluster_centers_[1,0]
    centre_dist = centre1 - centre0
    print("Centre distance = {:.3f} with widths = {:.4f} / {:.4f} gives SNR ratio {:.3}".format(centre_dist, std0, std1, centre_dist/mean_std))

    #Have a look at the distributions
    plt.figure()
    sns.distplot(all_vals[state == 0])
    sns.distplot(all_vals[state == 1])

    #calculate some switching matrices for each amplitude
    # 0->0 0->1
    # 1->0 1->1
    counts = []
    for buf in buffers:
        state = clusterer.predict(buf.reshape((2*reps,1)))
        init_state = state[::2]
        final_state = state[1::2]
        switched = np.logical_xor(init_state, final_state)

        count_mat = np.zeros((2,2), dtype=np.int)

        count_mat[0,0] = np.sum(np.logical_and(init_state == 0, np.logical_not(switched) ))
        count_mat[0,1] = np.sum(np.logical_and(init_state == 0, switched ))
        count_mat[1,0] = np.sum(np.logical_and(init_state == 1, switched ))
        count_mat[1,1] = np.sum(np.logical_and(init_state == 1, np.logical_not(switched) ))

        counts.append(count_mat)

    import h5py
    FID = h5py.File("data/nTron-AmpFall-PhaseDiagram-10V-52us-HighRes.h5", "w")
    FID.create_dataset("/buffer", data=buffers, compression="lzf")
    FID.create_dataset("/fall_times", data=fall_times, compression="lzf")
    FID.create_dataset("/amplitudes", data=amplitudes, compression="lzf")
    FID.close()

    mean_PtoAP = np.array([beta.mean(1+c[0,1], 1+c[0,0]) for c in counts])
    mean_APtoP = np.array([beta.mean(1+c[1,0], 1+c[1,1]) for c in counts])
    plt.figure()
    plt.title("P to AP")
    plt.xlabel("Pulse Falltime (ns)")
    plt.ylabel("Pulse Amplitude (Arb. Units)")
    means_diagram_PtoAP = mean_PtoAP.reshape(len(amplitudes), len(fall_times), order='F')
    plt.pcolormesh(fall_times*1e9, amplitudes, means_diagram_PtoAP, cmap="RdGy")
    plt.colorbar()
    plt.figure()
    plt.title("AP to P")
    plt.xlabel("Pulse Duration (ns)")
    plt.ylabel("Pulse Amplitude (V)")
    means_diagram_APtoP = mean_APtoP.reshape(len(amplitudes), len(fall_times), order='F')
    plt.pcolormesh(fall_times*1e9, amplitudes, means_diagram_APtoP, cmap="RdGy")
    plt.colorbar()
    plt.show()
