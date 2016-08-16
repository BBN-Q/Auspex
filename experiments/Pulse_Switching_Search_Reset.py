from pycontrol.instruments.keysight import *
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430

from PyDAQmx import *

from pycontrol.experiment import FloatParameter, IntParameter, Experiment
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print
from pycontrol.filters.io import WriteToHDF5
from pycontrol.filters.average import Average
from pycontrol.filters.plot import Plotter

import asyncio
import numpy as np
import time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from analysis.h5shell import h5shell
import pandas as pd
from scipy.interpolate import interp1d

from pycontrol.logging import logger

# Experimental Topology
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# PSPL Trigger -> DAQmx PFI0

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

class ResetSearchExperiment(Experiment):

    daq_buffer = OutputConnector()

    sample = "CSHE2"
    comment = "AWG Reset Amplitude Search"

    field = FloatParameter(default=0, unit="T")
    amplitude = FloatParameter(default=0, unit="V")
    amplitudes = np.arange(-0.7, 0.71, 0.05) # Reset amplitudes
    duration = 5e-9
    reps = 1 << 10
    reps_over = FloatParameter(default=5)
    samps_per_trig = 5
    settle_delay = 50e-6

    measure_current = 3e-6
    # Instruments
    arb   = M8190A("192.168.5.108")
    mag   = AMI430("192.168.5.109")
    keith = Keithley2400("GPIB0::25::INSTR")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

    polarity = -1

    def init_instruments(self):
        # Set up Keithley
        self.keith.triad()
        self.keith.conf_meas_res(res_range=1e6)
        self.keith.conf_src_curr(comp_voltage=0.6, curr_range=1.0e-5)
        self.keith.current = self.measure_current
        self.mag.ramp()

        self.arb.set_output(True, channel=1)
        self.arb.set_output(False, channel=2)
        self.arb.sample_freq = 12.0e9
        self.arb.waveform_output_mode = "WSPEED"
        self.setup_AWG()
        # Set up NIDAQ
        self.analog_input = Task()
        self.read = int32()
        self.buf_points = len(self.amplitudes)*self.samps_per_trig * self.reps
        # DAQmx Configure Code
        self.analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff, 0.0, 0.5, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, self.samps_per_trig)
        self.analog_input.CfgInputBuffer(self.buf_points)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        self.analog_input.SetStartTrigRetriggerable(1)
        # DAQmx Start Code
        self.analog_input.StartTask()

        # Assign methods
        self.field.assign_method(self.mag.set_field)

    def setup_AWG(self):
        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        self.arb.set_output_route("DC", channel=1)
        self.arb.voltage_amplitude = 1.0

        self.arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
        self.arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

        self.arb.continuous_mode = False
        self.arb.gate_mode = False

        def arb_pulse(amplitude, duration, sample_rate=12e9):
            pulse_points = int(duration*sample_rate)

            if pulse_points < 320:
                wf = np.zeros(320)
            else:
                wf = np.zeros(64*int(np.ceil(pulse_points/64.0)))
            wf[:pulse_points] = amplitude
            return wf

        segment_ids = []
        arb_voltage = arb_voltage_lookup()
        for amp in self.amplitudes:
            waveform   = arb_pulse(np.sign(amp)*arb_voltage(abs(amp)), self.duration)
            wf_data    = M8190A.create_binary_wf_data(waveform)
            segment_id = self.arb.define_waveform(len(wf_data))
            segment_ids.append(segment_id)
            self.arb.upload_waveform(wf_data, segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
        nidaq_trig_segment_id = self.arb.define_waveform(len(nidaq_trig_wf))
        self.arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

        settle_pts = int(640*np.ceil(self.settle_delay * 12e9 / 640))
        start_idxs = [0]

        scenario = Scenario()
        for si in segment_ids:
            seq = Sequence(sequence_loop_ct=int(self.reps))
            seq.add_waveform(si) # Apply switching pulse to the sample
            seq.add_idle(settle_pts, 0.0) # Wait for the measurement to settle
            seq.add_waveform(nidaq_trig_segment_id) # Trigger the NIDAQ measurement
            seq.add_idle(1 << 14, 0.0) # bonus non-contiguous memory delay
            scenario.sequences.append(seq)

        self.arb.upload_scenario(scenario, start_idx=start_idxs[-1])
        start_idxs.append(start_idxs[-1] + len(scenario.scpi_strings()))
        # The last entry is eroneous
        start_idxs = start_idxs[:-1]

        self.arb.sequence_mode = "SCENARIO"
        self.arb.scenario_advance_mode = "REPEAT"
        self.arb.stop()
        self.arb.scenario_start_index = 0
        self.arb.run()

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samps_per_trig", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("reps", range(self.reps)))
        descrip.add_axis(DataAxis("amplitude", self.amplitudes))
        self.daq_buffer.set_descriptor(descrip)

    async def run(self):
        # Establish buffers
        buffers = np.empty(self.buf_points)
        self.arb.advance()
        self.arb.trigger()
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                      buffers, self.buf_points, byref(self.read), None)
        logger.debug("Read a buffer of {} points".format(buffers.size))
        await self.daq_buffer.push(buffers)
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
        logger.debug("Stream has filled {} of {} points".format(self.daq_buffer.points_taken, self.daq_buffer.num_points() ))

    def shutdown_instruments(self):
        try:
            self.analog_input.StopTask()
        except Exception as e:
            logger.warning("Warning failed to stop task. This is typical.")
            pass
        self.arb.stop()
        self.keith.current = 0.0
        # mag.zero()

if __name__ == "__main__":
    exp = ResetSearchExperiment()
    exp.sample = "CSHE5-C1R3"
    exp.field.value = -0.0074
    exp.duration = 5e-9
    exp.measure_current = 3e-6
    amps = np.arange(-0.25, 0.26, 0.05)
    amps = np.append(amps, np.flipud(amps))
    exp.amplitudes = amps
    exp.init_streams()
    exp.add_sweep(exp.reps_over, np.linspace(0,9,5))

    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die5-C1R3\CSHE5-C1R3-Search_Reset_2016-07-07.h5")
    # Set up averager and plot
    averager = Average('amplitude')
    fig = Plotter(name="CSHE5-C1R3 - Search Reset", plot_dims=1)
    edges = [(exp.daq_buffer, wr.data), (exp.daq_buffer, averager.data), (averager.final_average,fig.data)]
    exp.set_graph(edges)
    exp.init_instruments()

    exp.init_progressbar(num=1)
    exp.run_sweeps()
    exp.shutdown_instruments()

    f = h5shell(wr.filename,'r')
    dset= f[f.grep('data')[-1]]
    buffers = dset.value
    f.close()
    # Plot the result
    NUM = len(amps)
    buff_mean = np.mean(buffers, axis=(2,3))
    mean_state = np.mean(buff_mean, axis=0)

    fig = plt.figure()
    for i in range(NUM):
        plt.plot(amps[i]*np.ones(buff_mean[:,i].size),
                1e-3*buff_mean[:,i]/max(exp.measure_current,1e-7),
                    '.', color='blue')
    plt.plot(amps, 1e-3*mean_state/max(exp.measure_current,1e-7), '-', color='red')
    plt.xlabel("AWG amplitude (V)", size=14);
    plt.ylabel("Resistance (kOhm)", size=14);
    plt.title("AWG Reset Amplitude Search")
    plt.show()
