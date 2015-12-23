from pycontrol.instruments.keysight import *
from pycontrol.instruments.stanford import SR830, SR865
from pycontrol.instruments.kepco import BOP2020M
from pycontrol.instruments.magnet import Electromagnet
from pycontrol.instruments.hall_probe import HallProbe
from pycontrol.instruments.picosecond import Picosecond10070A

from PyDAQmx import *

import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.stats import beta
from scipy.interpolate import interp1d
import pandas as pd

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
    lock = SR865("Lockin Amplifier", "USB0::0xB506::0x2000::002638::INSTR")
    bop  = BOP2020M("Kepco Power Supply", "GPIB0::1::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)
    pspl = Picosecond10070A("Pulse Generator", "GPIB0::24::INSTR")


    arb.set_output(True, channel=1)
    arb.set_output(False, channel=2)
    arb.sample_freq = 12.0e9
    arb.waveform_output_mode = "WSPEED"

    arb.abort()
    arb.delete_all_waveforms()
    arb.interface.write(":STAB:RES")

    arb.set_output_route("DC", channel=1)
    arb.voltage_amplitude = 1.0

    arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
    arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

    arb.continuous_mode = False
    arb.gate_mode = False

    segment_ids = []

    reset_wf    = arb_pulse(0.7, 0.6e-9)
    wf_data     = M8190A.create_binary_wf_data(reset_wf)
    rst_segment_id  = arb.define_waveform(len(wf_data))
    segment_ids.append(rst_segment_id)
    arb.upload_waveform(wf_data, rst_segment_id)

    # Picosecond trigger waveform
    pspl_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), samp_mkr=1)
    pspl_trig_segment_id = arb.define_waveform(len(pspl_trig_wf))
    arb.upload_waveform(pspl_trig_wf, pspl_trig_segment_id)

    # Picosecond trigger waveform
    nidaq_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    nidaq_trig_segment_id = arb.define_waveform(len(nidaq_trig_wf))
    arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

    reps = 1<<17
    lockin_settle_delay = 40e-6
    lockin_settle_pts = int(640*np.ceil(lockin_settle_delay * 12e9 / 640))

    scenario = Scenario()
    seq = Sequence(sequence_loop_ct=reps)
    seq.add_waveform(rst_segment_id)
    seq.add_idle(lockin_settle_pts, 0.0)
    seq.add_waveform(nidaq_trig_segment_id)
    seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
    seq.add_waveform(pspl_trig_segment_id)
    seq.add_idle(lockin_settle_pts, 0.0)
    seq.add_waveform(nidaq_trig_segment_id)
    seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay

    scenario.sequences.append(seq)
    arb.upload_scenario(scenario, start_idx=0)

    arb.sequence_mode = "SCENARIO"
    arb.scenario_advance_mode = "REPEAT"

    mag.field = -364
    time.sleep(2)

    #Setup picosecond
    pspl.duration  = 0.5e-9
    pspl.amplitude = 7.5*np.power(10, -4/20)
    pspl.trigger_source = "EXT"
    pspl.output = True

    #Variable attenuator
    df = pd.read_csv("calibration/RFSA2113SB.tsv", sep="\t")
    attenuator_interp = interp1d(df["Attenuation"], df["Control Voltage"])
    attenuator_lookup = lambda x : float(attenuator_interp(x))

    analog_input = Task()
    read = int32()

    # DAQmx Configure Code
    analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_RSE, -2.0,2.0, DAQmx_Val_Volts, None)
    analog_input.CfgSampClkTiming("/Dev1/PFI0", 20000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 2*reps)

    # DAQmx Start Code
    analog_input.StartTask()

    arb.scenario_start_index = 0
    arb.run()
    # attens = np.arange(-7.5,-6,0.01)
    attens    = [-6.01]
    durations = 1e-9*np.arange(0.38, 0.65, 0.01)
    # durations = [0.6e-9]
    buffers = np.empty((len(attens)*len(durations), 2*reps))
    idx = 0
    # lock.ao3 = attenuator_lookup(atten)
    lock.ao3 = attenuator_lookup(attens[0])
    time.sleep(0.01)
    # for atten in tqdm(attens):
    # pspl.duration = durations[0]

    for dur in tqdm(durations):
        pspl.duration = dur
        time.sleep(0.1)
        arb.advance()
        arb.trigger()
        analog_input.ReadAnalogF64(2*reps, -1, DAQmx_Val_GroupByChannel, buffers[idx], 2*reps, byref(read), None)

        idx += 1

    analog_input.StopTask()
    arb.stop()

    mag.field = 0
    bop.current = 0
    pspl.output = False

    #Get an idea of SNR
    #Cluster all the data into two based with starting point based on edges
    all_vals = buffers.flatten()
    all_vals.resize((all_vals.size,1))
    init_guess = np.array([np.min(all_vals), np.max(all_vals)])
    init_guess.resize((2,1))
    clusterer = KMeans(init=init_guess, n_clusters=2, max_iter=1000, n_jobs=-1, tol=1e-5)
    # clusterer = DBSCAN()
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

    mean_PtoAP = np.array([beta.mean(1+c[0,1], 1+c[0,0]) for c in counts])
    mean_APtoP = np.array([beta.mean(1+c[1,0], 1+c[1,1]) for c in counts])
    ci68_PtoAP = np.array([beta.interval(0.68, 1+c[0,1], 1+c[0,0]) for c in counts])
    ci68_APtoP = np.array([beta.interval(0.68, 1+c[1,0], 1+c[1,1]) for c in counts])
    ci95_PtoAP = np.array([beta.interval(0.95, 1+c[0,1], 1+c[0,0]) for c in counts])
    ci95_APtoP = np.array([beta.interval(0.95, 1+c[1,0], 1+c[1,1]) for c in counts])

    plt.figure()
    # volts = 7.5*np.power(10, (-5+attens)/20)
    current_palette = sns.color_palette()
    plt.plot(durations, mean_PtoAP)
    plt.fill_between(durations, [ci[0] for ci in ci68_PtoAP], [ci[1] for ci in ci68_PtoAP], color=current_palette[0], alpha=0.2, edgecolor="none")
    plt.fill_between(durations, [ci[0] for ci in ci95_PtoAP], [ci[1] for ci in ci95_PtoAP], color=current_palette[0], alpha=0.2, edgecolor="none")
    plt.plot(durations, mean_APtoP)
    plt.fill_between(durations, [ci[0] for ci in ci68_APtoP], [ci[1] for ci in ci68_APtoP], color=current_palette[1], alpha=0.2, edgecolor="none")
    plt.fill_between(durations, [ci[0] for ci in ci95_APtoP], [ci[1] for ci in ci95_APtoP], color=current_palette[1], alpha=0.2, edgecolor="none")
    # plt.xlabel("Pulse Amp (V)")
    plt.xlabel("Pulse Duration (ns)")
    plt.ylabel("Estimated Switching Probability")
    # plt.title("P to AP")
    # means_diagram_PtoAP = mean_PtoAP.reshape(len(attens), len(durations), order='F')
    # plt.pcolormesh(durations*1e9, volts, means_diagram_PtoAP, cmap="RdGy")
    # plt.colorbar()
    # plt.figure()
    # plt.title("AP to P")
    # means_diagram_APtoP = mean_APtoP.reshape(len(attens), len(durations), order='F')
    # plt.pcolormesh(durations*1e9, volts, means_diagram_APtoP, cmap="RdGy")
    # plt.colorbar()

    plt.show()
