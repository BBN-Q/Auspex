# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments.keysight import *
from auspex.instruments.stanford import SR865
from auspex.instruments.keithley import Keithley2400
from auspex.instruments.ami import AMI430

from PyDAQmx import *

import numpy as np
import time
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os.path
import h5py
# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0

# PARAMETERS: Confirm these before running
RST_DURATION = 5.0e-9 # Reset duration, second
RST_AMPS = np.arange(-0.7, 0.71, 0.05) # Reset amplitudes
MEASURE_CURRENT = 3.0e-6
SET_FIELD = 0.0133 # Tesla
REPS = 1 << 10 # Number of repeats per sequence
REPS_OVER = 5 # Number of repeats of scenario
SAMPS_PER_TRIG = 5 # Samples per trigger

# File to save
FOLDER = "data\\CSHE-Switching\\CSHE-Die2-C4R2"
FILENAME = "CSHE-2-C4R2_Search_Reset" # No extension
DATASET = "CSHE-2-C4R2/2016-06-30/Search_Reset"

def arb_pulse(amplitude, duration, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*int(np.ceil(pulse_points/64.0)))
    wf[:pulse_points] = amplitude
    return wf

def cluster_loop(buffers, num_clusters=2):
    """ Split 'buffers' into 'num_clusters' clusters """
    #Get an idea of SNR
    #Cluster all the data into three based with starting point based on edges
    all_vals = buffers.flatten()
    all_vals.resize((all_vals.size,1))
    init_guess = np.linspace(np.min(all_vals), np.max(all_vals), num_clusters)
    init_guess[[1,-1]] = init_guess[[-1,1]]
    init_guess.resize((num_clusters,1))
    clusterer = KMeans(init=init_guess, n_clusters=num_clusters)
    state = clusterer.fit_predict(all_vals)

    #Approximate SNR from centre distance and variance
    std0 = np.std(all_vals[state == 0])
    std1 = np.std(all_vals[state == 1])
    mean_std = 0.5*(std0 + std1)
    centre0 = clusterer.cluster_centers_[0,0]
    centre1 = clusterer.cluster_centers_[1,0]
    return (state, centre0, centre1, std0, std1)

def mk_dataset(f, dsetname, data):
    """ Make new dataset in the HDF5 file handle f """
    dset_list = []
    f.visit(lambda x: dset_list.append(x))
    dname = dsetname
    while dname in dset_list:
        # print("Found an existing dataset. Increase name by 1.")
        dname = dname[:-1] + chr(ord(dname[-1])+1)
    print("Make new dataset: %s" %dname)
    return f.create_dataset(dname, data=data)

if __name__ == '__main__':
    arb   = KeysightM8190A("192.168.5.108")
    mag   = AMI430("192.168.5.109")
    keith = Keithley2400("GPIB0::25::INSTR")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

    APtoP = False
    polarity = 1 if APtoP else -1

    duration  = RST_DURATION

    keith.triad()
    keith.conf_meas_res(res_range=1e6)
    keith.conf_src_curr(comp_voltage=0.6, curr_range=1.0e-5)
    keith.current = MEASURE_CURRENT
    mag.ramp()

    arb.set_output(True, channel=1)
    arb.set_output(False, channel=2)
    arb.sample_freq = 12.0e9
    arb.waveform_output_mode = "WSPEED"

    arb.abort()
    arb.delete_all_waveforms()
    arb.reset_sequence_table()

    arb.set_output_route("DC", channel=1)
    arb.voltage_amplitude = 1.0

    arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
    arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

    arb.continuous_mode = True
    arb.gate_mode = False

    amps = np.append(RST_AMPS, np.flipud(RST_AMPS))
    segment_ids = []
    for amp in amps:
        waveform   = arb_pulse(amp, duration)
        wf_data    = KeysightM8190A.create_binary_wf_data(waveform)
        segment_id = arb.define_waveform(len(wf_data))
        segment_ids.append(segment_id)
        arb.upload_waveform(wf_data, segment_id)

    # NIDAQ trigger waveform
    nidaq_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    nidaq_trig_segment_id = arb.define_waveform(len(nidaq_trig_wf))
    arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

    reps = REPS
    settle_delay = 50e-6
    settle_pts = int(640*np.ceil(settle_delay * 12e9 / 640))

    start_idxs = [0]

    scenario = Scenario()

    for si in segment_ids:
        seq = Sequence(sequence_loop_ct=int(reps))
        seq.add_waveform(si) # Apply switching pulse to the sample
        seq.add_idle(settle_pts, 0.0) # Wait for the measurement to settle
        seq.add_waveform(nidaq_trig_segment_id) # Trigger the NIDAQ measurement
        seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
        scenario.sequences.append(seq)

    arb.upload_scenario(scenario, start_idx=start_idxs[-1])
    start_idxs.append(start_idxs[-1] + len(scenario.scpi_strings()))

    # The last entry is eroneous
    start_idxs = start_idxs[:-1]

    arb.sequence_mode = "SCENARIO"
    arb.scenario_advance_mode = "SINGLE"

    # Ramp to the switching field
    mag.set_field(SET_FIELD)

    # Variable attenuator
    df = pd.read_csv("calibration/RFSA2113SB.tsv", sep="\t")
    attenuator_interp = interp1d(df["Attenuation"], df["Control Voltage"])
    attenuator_lookup = lambda x : float(attenuator_interp(x))

    analog_input = Task()
    read = int32()

    # DAQmx Configure Code
    samps_per_trig = SAMPS_PER_TRIG
    analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff, 0.0, 0.5, DAQmx_Val_Volts, None)
    analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, samps_per_trig)
    analog_input.CfgInputBuffer(samps_per_trig * reps)
    analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
    analog_input.SetStartTrigRetriggerable(1)

    # DAQmx Start Code
    analog_input.StartTask()

    # Add an extra dimension
    reps_over = REPS_OVER
    # Establish buffers
    buffers = np.empty((reps_over,len(amps),reps*samps_per_trig))
    arb.stop()
    arb.scenario_start_index = 0
    arb.run()
    arb.advance()
    arb.trigger()
    analog_input.ReadAnalogF64(buffers.size, -1, DAQmx_Val_GroupByChannel,
                                  buffers, buffers.size, byref(read), None)

    # Shutting down
    try:
        analog_input.StopTask()
    except Exception as e:
        print("Warning failed to stop task.")
        pass
    arb.stop()
    keith.current = 0.0
    # mag.zero()

    # Save the data
    fname = os.path.join(FOLDER, FILENAME+'.h5')
    with h5py.File(fname,'a') as f:
        data1 = amps
        data2 = buffers
        dset1 = mk_dataset(f, DATASET+'_AMPS_A', data1)
        dset2 = mk_dataset(f, DATASET+'_VOLTS_A', data2)

    # Plot the result
    fig = plt.figure(0)
    NUM = len(amps)
    for i in range(NUM):
        buff = buffers[:,i,:].flatten()
        plt.plot(amps[i]*np.ones(buff.size), 1e-3*buff/max(MEASURE_CURRENT,1e-7),
                    '.', color='blue')
    mean_state = np.mean(buffers, axis=(0,2))
    plt.plot(amps, 1e-3*mean_state/max(MEASURE_CURRENT,1e-7), '-', color='red')
    plt.xlabel("AWG amplitude (V)", size=14);
    plt.ylabel("Resistance (kOhm)", size=14);
    plt.title("AWG Reset Amplitude Search")
    plt.show()
    # Analyze data by clustering
    """
    NUM = len(amps)
    V1 = np.zeros(NUM)
    V2 = np.zeros(NUM)
    dV1 = np.zeros(NUM)
    dV2 = np.zeros(NUM)
    for i in range(NUM):
        clus = cluster_loop(buffers[:,i,:])
        V1[i] = clus[1]
        V2[i] = clus[2]
        dV1[i] = clus[3]
        dV2[i] = clus[4]
    # Plot
    plt.figure(1)
    plt.errorbar(amps, 1e-3*V1/max(MEASURE_CURRENT,1e-7),
                yerr=1e-3*dV1/max(MEASURE_CURRENT,1e-7), fmt='--o')
    plt.errorbar(amps, 1e-3*V2/max(MEASURE_CURRENT,1e-7),
                yerr=1e-3*dV2/max(MEASURE_CURRENT,1e-7), fmt='--o')
    plt.xlabel("AWG amplitude (V)", size=14);
    plt.ylabel("Resistance (kOhm)", size=14);
    """
    print("Finished.")
