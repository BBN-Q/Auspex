from pycontrol.instruments.keysight import *
from pycontrol.instruments.stanford import SR830, SR865
from pycontrol.instruments.kepco import BOP2020M
from pycontrol.instruments.magnet import Electromagnet
from pycontrol.instruments.hall_probe import HallProbe
from pycontrol.instruments.picosecond import Picosecond10070A

from PyDAQmx import *

import numpy as np
import time
from tqdm import tqdm
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
    arb  = M8190A("192.168.5.108")
    lock = SR865("USB0::0xB506::0x2000::002638::INSTR")
    bop  = BOP2020M("GPIB0::1::INSTR")
    hp   = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
    mag  = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)
    pspl = Picosecond10070A("Pulse Generator", "GPIB0::24::INSTR")


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

    reset_wf    = arb_pulse(0.75, 3.0/12e9)
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

    reps = 1 << 17
    lockin_settle_delay = 100e-6
    lockin_settle_pts = int(640*np.ceil(lockin_settle_delay * 12e9 / 640))

    scenario = Scenario()
    seq = Sequence(sequence_loop_ct=int(reps/2))
    #First try with reset flipping pulse
    seq.add_waveform(rst_segment_id)
    seq.add_idle(lockin_settle_pts, 0.0)
    seq.add_waveform(nidaq_trig_segment_id)
    seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
    seq.add_waveform(pspl_trig_segment_id)
    seq.add_idle(lockin_settle_pts, 0.0)
    seq.add_waveform(nidaq_trig_segment_id)
    seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
    #second try without
    seq.add_waveform(no_rst_segment_id)
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
    analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_RSE, -10.0,10.0, DAQmx_Val_Volts, None)
    analog_input.CfgSampClkTiming("/Dev1/PFI0", 20000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 2*reps)

    # DAQmx Start Code
    analog_input.StartTask()

    arb.scenario_start_index = 0
    arb.run()
    # attens = np.arange(-7.5,-6,0.01)
    attens    = [-6.01]
    durations = 1e-9*np.arange(0.35, 0.65, 0.01)
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
        time.sleep(0.2)
        arb.advance()
        arb.trigger()
        analog_input.ReadAnalogF64(2*reps, -1, DAQmx_Val_GroupByChannel, buffers[idx], 2*reps, byref(read), None)

        idx += 1

    analog_input.StopTask()
    arb.stop()

    mag.field = 0
    bop.current = 0
    pspl.output = False
