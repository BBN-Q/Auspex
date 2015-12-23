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

    segment_ids = []
    amps = np.arange(0.6, 1.00, 0.01)
    for amp in amps:
        waveform   = pulse(amp, 0.6e-9)
        wf_data    = M8190A.create_binary_wf_data(waveform)
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    # Trigger waveform
    trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    trig_segment_id = arb.define_waveform(len(trig_wf))
    arb.upload_waveform(trig_wf, trig_segment_id)

    start_idxs = [0]

    rate = 1.25e6/(2**8)

    reps = 1 << 13
    lockin_settle_delay = 30e-6
    lockin_settle_pts = int(640*np.ceil(lockin_settle_delay * 12e9 / 640))

    for si in segment_ids:
        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=reps)
        seq.add_waveform(si) # Apply pulse to the sample
        seq.add_idle(lockin_settle_pts, 0.0) # Wait for the measurement to settle
        seq.add_waveform(trig_segment_id) # Trigger the NIDAQ measurement
        seq.add_idle(1 << 17, 0.0) # Lockin sample rate delay 1 << 17 = 11us
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

    buffers = np.empty((len(segment_ids), reps))

    for ct, idx in enumerate(tqdm(start_idxs)):
        arb.stop()
        arb.scenario_start_index = idx
        arb.run()
        arb.trigger()
        analog_input.ReadAnalogF64(reps, -1, DAQmx_Val_GroupByChannel, buffers[ct], reps, byref(read), None)

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
        state = clusterer.predict(buf.reshape((reps,1)))
        init_state = state[:-1]
        final_state = state[1:]
        switched = np.logical_xor(init_state, final_state)

        count_mat = np.zeros((2,2), dtype=np.int)

        count_mat[0,0] = np.sum(np.logical_and(init_state == 0, np.logical_not(switched) ))
        count_mat[0,1] = np.sum(np.logical_and(init_state == 0, switched ))
        count_mat[1,0] = np.sum(np.logical_and(init_state == 1, switched ))
        count_mat[1,1] = np.sum(np.logical_and(init_state == 1, np.logical_not(switched) ))

        counts.append(count_mat)

    plt.figure()
    mean_PtoAP = [beta.mean(1+c[0,1], 1+c[0,0]) for c in counts]
    mean_APtoP = [beta.mean(1+c[1,0], 1+c[1,1]) for c in counts]
    ci68_PtoAP = [beta.interval(0.68, 1+c[0,1], 1+c[0,0]) for c in counts]
    ci68_APtoP = [beta.interval(0.68, 1+c[1,0], 1+c[1,1]) for c in counts]
    ci95_PtoAP = [beta.interval(0.95, 1+c[0,1], 1+c[0,0]) for c in counts]
    ci95_APtoP = [beta.interval(0.95, 1+c[1,0], 1+c[1,1]) for c in counts]
    current_palette = sns.color_palette()
    plt.plot(amps, mean_PtoAP)
    plt.fill_between(amps, [ci[0] for ci in ci68_PtoAP], [ci[1] for ci in ci68_PtoAP], color=current_palette[0], alpha=0.2, edgecolor="none")
    plt.fill_between(amps, [ci[0] for ci in ci95_PtoAP], [ci[1] for ci in ci95_PtoAP], color=current_palette[0], alpha=0.2, edgecolor="none")
    plt.plot(amps, mean_APtoP)
    plt.fill_between(amps, [ci[0] for ci in ci68_APtoP], [ci[1] for ci in ci68_APtoP], color=current_palette[1], alpha=0.2, edgecolor="none")
    plt.fill_between(amps, [ci[0] for ci in ci95_APtoP], [ci[1] for ci in ci95_APtoP], color=current_palette[1], alpha=0.2, edgecolor="none")

    plt.legend(("P->AP", "AP->P"))

    plt.show()
