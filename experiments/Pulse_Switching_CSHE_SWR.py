from pycontrol.instruments.keysight import *
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430
from pycontrol.instruments.rfmd import Attenuator

from pycontrol.experiment import FloatParameter, IntParameter, Experiment
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.io import WriteToHDF5

from PyDAQmx import *

import itertools
import numpy as np
import asyncio
import time, sys
import h5py
import matplotlib.pyplot as plt

import analysis.switching as sw
from analysis.h5shell import h5shell

from pycontrol.logging import logger
# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

def arb_voltage_lookup(arb_calib="calibration/AWG_20160718.csv",
                        midpoint_calib="calibration/midpoint_20160718.csv"):
    df_midpoint = pd.read_csv(midpoint_calib, sep=",")
    df_arb = pd.read_csv(arb_calib, sep=",")
    midpoint_lookup = interp1d(df_midpoint["Sample Voltage"],df_midpoint["Midpoint Voltage"])
    arb_control_lookup = interp1d(df_arb["Midpoint Voltage"],df_arb["Control Voltage"])
    sample_volts = []
    control_volts = []
    for volt in df_midpoint['Sample Voltage']:
        mid_volt = midpoint_lookup(volt)
        if (mid_volt > min(df_arb['Midpoint Voltage'])) and (mid_volt < max(df_arb['Midpoint Voltage'])):
            sample_volts.append(volt)
            control_volts.append(arb_control_lookup(mid_volt))
    return interp1d(sample_volts, control_volts)

class SWRExperiment(Experiment):
    """ Experiment class for Switch Rate measurment
    Determine switching probability for V << V0
    with varying V (and durations?)
    """

    # Sample information
    sample         = "CSHE5-C1R3"
    comment        = "Switching Rate for V << V0"

    # Parameters
    field          = FloatParameter(default=0.0, unit="T")
    pulse_duration = FloatParameter(default=10.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=0.1, unit="V")

    attempts       = 1 << 10
    settle_delay    = 50e-6
    measure_current = 3.0e-6
    samps_per_trig  = 5

    polarity        = 1
    # pspl_atten      = 10

    min_daq_voltage = 0.0
    max_daq_voltage = 0.4

    reset_amplitude = 0.7
    reset_duration  = 5.0e-9

    # Things coming back
    daq_buffer     = OutputConnector()

    # Instrument resources
    mag   = AMI430("192.168.5.109")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")
    pspl  = Picosecond10070A("GPIB0::24::INSTR")
    # atten = Attenuator("calibration/RFSA2113SB_HPD_20160706.csv", lock.set_ao2, lock.set_ao3)
    arb   = M8190A("192.168.5.108")
    keith = Keithley2400("GPIB0::25::INSTR")

    def init_instruments(self):

        # ===================
        #    Setup the Keithley
        # ===================

        self.keith.triad()
        self.keith.conf_meas_res(res_range=1e5)
        self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
        self.keith.current = self.measure_current
        self.mag.ramp()

        # ===================
        #    Setup the AWG
        # ===================

        self.arb.set_output(True, channel=1)
        self.arb.set_output(False, channel=2)
        self.arb.sample_freq = 12.0e9
        self.arb.waveform_output_mode = "WSPEED"
        self.arb.set_output_route("DC", channel=1)
        self.arb.voltage_amplitude = 1.0
        self.arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
        self.arb.set_marker_level_high(1.5, channel=1, marker_type="sync")
        self.arb.continuous_mode = False
        self.arb.gate_mode = False

        # ===================
        #   Setup the PSPL
        # ===================

        self.pspl.amplitude = self.polarity*7.5*np.power(10, -self.pspl_atten/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.1
        self.pspl.output = True

        # self.setup_arb(self.pulse_voltage.value)

        # Assign methods
        self.field.assign_method(self.mag.set_field)
        self.pulse_voltage.assign_method(self.setup_arb)

    def setup_arb(self, volt):
        def arb_pulse(amplitude, duration, sample_rate=12e9):
            pulse_points = int(duration*sample_rate)

            if pulse_points < 320:
                wf = np.zeros(320)
            else:
                wf = np.zeros(64*np.ceil(pulse_points/64.0))
            wf[:pulse_points] = amplitude
            return wf

        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        # Reset waveform
        reset_wf    = arb_pulse(-self.polarity*self.reset_amplitude, self.reset_duration)
        wf_data     = M8190A.create_binary_wf_data(reset_wf)
        rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, rst_segment_id)

        # Switching waveform
        switch_wf    = arb_pulse(self.polarity*arb_voltage_lookup()(volt), self.pulse_duration.value)
        wf_data     = M8190A.create_binary_wf_data(switch_wf)
        sw_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, sw_segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
        nidaq_trig_segment_id = self.arb.define_waveform(len(nidaq_trig_wf))
        self.arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

        settle_pts = int(640*np.ceil(self.settle_delay * 12e9 / 640))

        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=int(self.attempts))
        #First try with reset flipping pulse
        seq.add_waveform(rst_segment_id)
        seq.add_idle(settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
        seq.add_waveform(sw_segment_id)
        seq.add_idle(settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
        scenario.sequences.append(seq)
        self.arb.upload_scenario(scenario, start_idx=0)
        self.arb.sequence_mode = "SCENARIO"
        self.arb.scenario_advance_mode = "REPEAT"
        self.arb.scenario_start_index = 0
        self.arb.run()

        # ===================
        #   Setup the NIDAQ
        # ===================

        self.analog_input = Task()
        self.read = int32()
        self.buf_points = 2*self.samps_per_trig*self.attempts
        self.analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff,
            self.min_daq_voltage, self.max_daq_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , self.samps_per_trig)
        self.analog_input.CfgInputBuffer(self.buf_points)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        self.analog_input.SetStartTrigRetriggerable(1)
        self.analog_input.StartTask()

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("state", range(2)))
        descrip.add_axis(DataAxis("attempts", range(self.attempts)))
        self.daq_buffer.set_descriptor(descrip)

    async def run(self):
        """We are no longer using the sweeper."""

        # Keep track of the previous values
        logger.debug("Waiting for filters.")
        await asyncio.sleep(1.0)

        self.arb.advance()
        self.arb.trigger()
        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        await self.daq_buffer.push(buf)
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.002)
        # logger.debug("Stream has filled {} of {} points".format(self.daq_buffer.points_taken, self.daq_buffer.num_points() ))

    def shutdown_instruments(self):
        self.keith.current = 0.0e-5
        # self.mag.zero()
        self.arb.stop()
        self.pspl.output = False
        try:
            self.analog_input.StopTask()
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass

def data_at_volt(fname, volt):
    """ Extract datasets in file fname that have pulse_voltage == volt """
    with h5shell(fname, 'r') as f:
        dsets = [f[k] for k in f.ls('-d')]
        dsets = [dset if abs(float(dset.attrs['pulse_voltage'])-volt)/volt<0.01 for dset in dsets]
        data_mean = [np.mean(dset.value, axis=-1) for dset in dsets]
    return np.concatenate(data_mean,axis=0)

def stop_measure(data, max_points = 1 << 22):
    """ Determine whether we should stop the measurement at a given pulse voltage """
    num_points = int(data.size/2)
    if num_points >= max_points:
        return True
    results = sw.switching_BER(data)
    mean = results[0]
    limit = results[1]
    return limit > mean

def load_SWR_data(fname):
    with h5shell(fname, 'r') as f:
        dsets = [f[k] for k in f.ls('-d')]
        volts = [float(dset.attrs['pulse_voltage']) for dset in dsets]

    unique_volts = []
    for v in sorted(volts):
        if len(unique_volts) == 0:
            unique_volts.append(v)
        else:
            check = [abs((uv-v)/v)<0.01 for uv in unique_volts]
            if not np.any(check):
                unique_volts.append(v)
    data = [data_at_volt(fname,volt) for volt in unique_volts]
    return np.array(unique_volts), np.array(data)

def plot_SWR(volts, results):
    mean = []; limit = []; ci68 = []; ci95 = []
    for datum in results:
        mean.append(datum[0])
        limit.append(datum[1])
        ci68.append(datum[2])
        ci95.append(datum[3])
    mean = np.array(mean)
    limit = np.array(limit)
    fig = plt.figure()
    plt.semilogy(volts, mean, '-o')
    plt.semilogy(volts, limit, linestyle="--")
    plt.fill_between(volts, [ci[0] for ci in ci68], [ci[1] for ci in ci68],  alpha=0.2, edgecolor="none")
    plt.fill_between(volts, [ci[0] for ci in ci95], [ci[1] for ci in ci95],  alpha=0.2, edgecolor="none")
    plt.ylabel("Switching Rate", size=14)
    plt.xlabel("Pulse Voltage (V)", size=14)
    return fig

if __name__ == '__main__':
    exp = SWRExperiment()
    exp.sample = "CSHE5 - C1R3"
    exp.comment = "Switching Rate - AP to P - 5ns"
    exp.polarity = -1 # -1: AP to P; 1: P to AP
    exp.field.value = -0.0074
    exp.attempts = 1 << 15
    exp.pulse_duration.value = 5e-9 # Fixed
    exp.init_instruments()

    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die5-C1R3\CSHE5-C1R3-AP2P_2016-07-26_SWR_5ns.h5")
    edges = [(exp.daq_buffer, wr.data)]
    exp.set_graph(edges)

    
    voltages_list = np.linspace(0.1,0.2,6)

    t1 = [] # Keep track of time
    t2 = []
    for volt in range(voltages_list):
        print("=========================")
        print("Now at: {}.".format(volt))
        t1.append(time.time())
        exp.pulse_voltage.value = volt
        finish = False
        count = 0
        while not finish:
            count = count + 1
            print("Measurement count = %d" %count)
            exp.init_streams()
            exp.reset()
            exp.run_loop()
            data = data_at_volt(wr.filename, volt)
            finish = stop_measure(data, 1<<16)
        t2.append(time.time())
        print("Done one series. Elapsed time: {} min".format((t2[-1]-t1[-1])/60))
        time.sleep(3)

    # Plot data
    volts, data = load_SWR_data(wr.filename)
    results = [sw.switching_BER(datum) for datum in data]
    fig = plot_SWR(volts, results)
