from pycontrol.instruments.keysight import *
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430

from PyDAQmx import *

import numpy as np
import time
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

def arb_pulse(amplitude, duration, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)

    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude
    return wf

if __name__ == '__main__':
    arb   = M8190A("Test Arb", "192.168.5.108")
    pspl  = Picosecond10070A("Pulse Generator", "GPIB0::24::INSTR")
    mag   = AMI430("This is a magnet", "192.168.5.109")
    keith = Keithley2400("This is a keithley", "GPIB0::25::INSTR")
    lock  = SR865("Lockin Amplifier", "USB0::0xB506::0x2000::002638::INSTR")

    APtoP = False
    polarity = 1 if APtoP else -1

    keith.triad()
    keith.conf_meas_res(res_range=1e6)
    keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
    keith.current = 5e-6
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

    arb.continuous_mode = False
    arb.gate_mode = False

    reset_wf    = arb_pulse(-polarity*0.32, 5.0e-9)
    wf_data     = M8190A.create_binary_wf_data(reset_wf)
    rst_segment_id  = arb.define_waveform(len(wf_data))
    arb.upload_waveform(wf_data, rst_segment_id)

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

    reps = 1 << 10
    settle_delay = 40e-6
    settle_pts = int(640*np.ceil(settle_delay * 12e9 / 640))

    scenario = Scenario()
    seq = Sequence(sequence_loop_ct=int(reps))
    #First try with reset flipping pulse
    seq.add_waveform(rst_segment_id)
    seq.add_idle(settle_pts, 0.0)
    seq.add_waveform(nidaq_trig_segment_id)
    seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
    seq.add_waveform(pspl_trig_segment_id)
    seq.add_idle(settle_pts, 0.0)
    seq.add_waveform(nidaq_trig_segment_id)
    seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
    scenario.sequences.append(seq)
    arb.upload_scenario(scenario, start_idx=0)

    arb.sequence_mode = "SCENARIO"
    arb.scenario_advance_mode = "REPEAT"

    # Setup picosecond
    pspl.duration  = 5e-9
    pspl_attenuation = 6
    pspl.amplitude = polarity*7.5*np.power(10, -pspl_attenuation/20)
    pspl.trigger_source = "EXT"
    pspl.output = True
    #TODO: pspl.trigger_level = 0.1

    # Ramp to the switching field
    mag.set_field(85e-4) # 85G

    # Variable attenuator
    df = pd.read_csv("calibration/RFSA2113SB.tsv", sep="\t")
    attenuator_interp = interp1d(df["Attenuation"], df["Control Voltage"])
    attenuator_lookup = lambda x : float(attenuator_interp(x))

    analog_input = Task()
    read = int32()

    # DAQmx Configure Code
    analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_RSE, 0, 1.0, DAQmx_Val_Volts, None)
    analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , 10)
    analog_input.CfgInputBuffer(10 * 2*reps)
    analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
    analog_input.SetStartTrigRetriggerable(1)

    # DAQmx Start Code
    analog_input.StartTask()

    arb.scenario_start_index = 0
    arb.run()
    attens = np.arange(-16.01, -6, 0.1)
    #attens    = np.arange(-9.01,-6.00,0.5)
    # durations = np.array([5.0e-9])
    durations = 1e-9*np.arange(0.1, 5.01, 0.1)

    volts = 7.5*np.power(10, (-pspl_attenuation+attens)/20)
    buffers = np.empty((len(attens)*len(durations), 2*10*reps))
    idx = 0
    # lock.ao3 = attenuator_lookup(attens[0])

    for dur in tqdm(durations):
        pspl.duration = dur
        time.sleep(0.1) # Allow the PSPL to settle
        for atten in attens:
            lock.ao3 = attenuator_lookup(atten)
            time.sleep(0.02) # Make sure attenuation is set

            arb.advance()
            arb.trigger()
            analog_input.ReadAnalogF64(2*10*reps, -1, DAQmx_Val_GroupByChannel, buffers[idx], 2*10*reps, byref(read), None)

            idx += 1

    try:
        analog_input.StopTask()
    except Exception as e:
        print("Warning failed to stop task.")
        pass
    arb.stop()
    keith.current = 0.0
    mag.zero()
    pspl.output = False
