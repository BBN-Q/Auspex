from pycontrol.instruments.keysight import *
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430

from PyDAQmx import *

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.cluster import KMeans
import os.path
import h5py

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

# PARAMETERS: Confirm these before running
SET_FIELD = -0.017 # Tesla
MEASURE_CURRENT = 3e-6 # Ampere, should not be zero!
BASE_ATTENUATION = 4
DURATIONS = 1e-9*np.array([3.0, 6.0]) # List of durations
ATTENUATIONS = np.arange(-20.0,-6.0,1) # Between -28 and -6

REPS = 1 << 10 # Number of attemps
SAMPLES_PER_TRIGGER = 5 # Samples per trigger

# File to save
FOLDER = "data\\CSHE-Switching\\CSHE-Die2-C5R7"
FILENAME = "CSHE-2-C5R7_Search_Switch" # No extension
DATASET = "CSHE-2-C5R7/2016-06-27/Search_Switch"

def arb_pulse(amplitude, duration, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
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
    arb   = M8190A("192.168.5.108")
    pspl  = Picosecond10070A("GPIB0::24::INSTR")
    mag   = AMI430("192.168.5.109")
    keith = Keithley2400("GPIB0::25::INSTR")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

    APtoP = False
    polarity = 1 if APtoP else -1

    # configure the Keithley
    keith.triad()
    keith.conf_meas_res(res_range=1e6)
    keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
    keith.current = MEASURE_CURRENT
    mag.ramp()

    # configure the AWG
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

    arb.continuous_mode = False
    arb.gate_mode = False

    # define waveform
    no_reset_wf = arb_pulse(0.0, 3.0/12e9)
    wf_data     = M8190A.create_binary_wf_data(no_reset_wf)
    no_rst_segment_id  = arb.define_waveform(len(wf_data))
    arb.upload_waveform(wf_data, no_rst_segment_id)

    # Picosecond trigger waveform
    pspl_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), samp_mkr=1)
    pspl_trig_segment_id = arb.define_waveform(len(pspl_trig_wf))
    arb.upload_waveform(pspl_trig_wf, pspl_trig_segment_id)

    # NIDAQ trigger waveform
    nidaq_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
    nidaq_trig_segment_id = arb.define_waveform(len(nidaq_trig_wf))
    arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

    settle_delay = 200e-6
    settle_pts = int(640*np.ceil(settle_delay * 12e9 / 640))

    reps = REPS
    # define Scenario
    scenario = Scenario()
    seq = Sequence(sequence_loop_ct=reps)
    seq.add_waveform(pspl_trig_segment_id)
    seq.add_idle(settle_pts, 0.0)
    seq.add_waveform(nidaq_trig_segment_id)
    seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
    scenario.sequences.append(seq)
    arb.upload_scenario(scenario, start_idx=0)

    arb.sequence_mode = "SCENARIO"
    arb.scenario_advance_mode = "SINGLE"

    # Setup picosecond
    pspl.duration  = 5e-9
    pspl_attenuation = BASE_ATTENUATION
    pspl.amplitude = polarity*7.5*np.power(10, -pspl_attenuation/20.0)
    pspl.trigger_source = "EXT"
    pspl.output = True
    pspl.trigger_level = 0.1

    # Ramp to the switching field
    mag.set_field(SET_FIELD)

    # Variable attenuator
    df = pd.read_csv("calibration/RFSA2113SB.tsv", sep="\t")
    attenuator_interp = interp1d(df["Attenuation"], df["Control Voltage"])
    attenuator_lookup = lambda x : float(attenuator_interp(x))

    analog_input = Task()
    read = int32()

    # DAQmx Configure Code
    samps_per_trig = SAMPLES_PER_TRIGGER
    analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_RSE, 0, 1.0, DAQmx_Val_Volts, None)
    analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , samps_per_trig)
    analog_input.CfgInputBuffer(samps_per_trig*reps)
    analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
    analog_input.SetStartTrigRetriggerable(1)

    # DAQmx Start Code
    analog_input.StartTask()

    arb.scenario_start_index = 0
    arb.run()

    durations = DURATIONS
    attens = ATTENUATIONS

    def execute(pol=polarity, direction=1):
        """ Carry out the measurement

        polarity: polarity of PSPL amplitude \
        direction: 1 - magnitude small --> large \
                  -1 - magnitude large --> small
        """
        id_dur = 0
        pspl.amplitude = pol*7.5*np.power(10, -pspl_attenuation/20)
        attenss = attens
        if direction==-1: # amplitude large to small
            attenss = np.flipud(attens)

        volts = 7.5*np.power(10, (-pspl_attenuation+attenss-10)/20)
        buffers = np.zeros((len(durations), len(attens), samps_per_trig*reps))
        for dur in tqdm(durations, leave=True):
            id_atten = 0
            pspl.duration = dur
            time.sleep(0.1) # Allow the PSPL to settle
            for atten in tqdm(attenss, nested=True, leave=False):
                lock.ao3 = attenuator_lookup(atten)
                time.sleep(0.02) # Make sure attenuation is set
                # trigger out
                arb.advance()
                arb.trigger()
                analog_input.ReadAnalogF64(samps_per_trig*reps, -1, DAQmx_Val_GroupByChannel,
                                           buffers[id_dur, id_atten], samps_per_trig*reps, byref(read), None)
                id_atten += 1
            id_dur += 1
        return pol*volts, buffers

    # Execute
    volts1, buffers1 = execute(-1,-1)
    volts2, buffers2 = execute(1, 1)
    volts3, buffers3 = execute(1,-1)
    volts4, buffers4 = execute(-1,1)

    # Shutting down
    try:
        analog_input.StopTask()
    except Exception as e:
        print("Warning failed to stop task.")
        pass

    arb.stop()
    keith.current = 0.0
    # mag.zero()
    pspl.output = False

    # Do some polishment
    volts_tot = np.concatenate((volts1, volts2, volts3, volts4), axis=0)
    buffers_tot = np.concatenate((buffers1, buffers2, buffers3, buffers4), axis=1)
    buffers_mean = np.mean(buffers_tot, axis=2) # Average over samps_per_trig

    # Save the data
    fname = os.path.join(FOLDER, FILENAME+'.h5')
    with h5py.File(fname,'a') as f:
        data1 = volts_tot
        data2 = buffers_tot
        dset1 = mk_dataset(f, DATASET+'_AMPS_A', data1)
        dset2 = mk_dataset(f, DATASET+'_VOLTS_A', data2)

    # Plot
    NUM = len(volts_tot)
    V1 = np.zeros(NUM)
    V2 = np.zeros(NUM)
    dV1 = np.zeros(NUM)
    dV2 = np.zeros(NUM)
    for i, dur in enumerate(durations):
        # Plot the original data
        plt.figure(i)
        for j in range(NUM):
            buff = buffers_tot[i,j,:].flatten()
            plt.plot(volts_tot[j]*np.ones(buff.size), 1e-3*buff/max(MEASURE_CURRENT,1e-7),
                        '.', color='blue')
        plt.plot(volts_tot, 1e-3*buffers_mean[i]/max(MEASURE_CURRENT, 1e-7), '-', color='red')
        plt.xlabel("Output V", size=14)
        plt.ylabel("Resitance (kOhm)", size=14)
        plt.title("Duration = {0} ns".format(dur*1e+9))
    plt.show()
    """
    # Analyze data by clustering
    for j in range(NUM):
        clus = cluster_loop(buffers_tot[i,j,:])
        V1[j] = clus[1]
        V2[j] = clus[2]
        dV1[j] = clus[3]
        dV2[j] = clus[4]
    # Plot
    plt.subplot(1,3,2)
    plt.errorbar(volts_tot, 1e-3*V1/max(MEASURE_CURRENT,1e-7),
                yerr=1e-3*dV1/max(MEASURE_CURRENT,1e-7), fmt='--o')
    plt.errorbar(volts_tot, 1e-3*V2/max(MEASURE_CURRENT,1e-7),
                yerr=1e-3*dV2/max(MEASURE_CURRENT,1e-7), fmt='--o')
    plt.xlabel("Output V", size=14);
    plt.ylabel("Resistance (kOhm)", size=14);
    plt.title("Duration = {0} ns".format(dur*1e+9))

    """
    print("Finished.")
