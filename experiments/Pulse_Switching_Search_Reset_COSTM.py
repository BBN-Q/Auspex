# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import KeysightM8190A, Scenario, Sequence
from auspex.instruments import SR865
# from auspex.instruments import Keithley2400
from auspex.instruments import AMI430
from auspex.instruments import RFMDAttenuator

from PyDAQmx import *

from auspex.experiment import FloatParameter, Experiment
from auspex.stream import DataAxis, OutputConnector, DataStreamDescriptor
from auspex.filters import WriteToHDF5, Averager, Plotter
from auspex.log import logger

import asyncio
import numpy as np
import time
import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import h5py

# Experimental Topology
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Locking across 1kOhm ref resistor -> DAQmx AI0
#
# Using HP 33150A bias tee to pass 1 MHZ lockin baseband
# Note that PSPL 5575A bias-tees can't pass more than 1-10 kHz

def arb_voltage_lookup(arb_calib="calibration/AWG_20160901.csv"):
    df_arb = pd.read_csv(arb_calib, sep=",")
    return interp1d(df_arb["Amp Out"], df_arb["Control Voltage"])

class ResetSearchLockinExperiment(Experiment):

    voltage          = OutputConnector()
    field            = FloatParameter(default=0, unit="T")
    pulse_duration   = FloatParameter(default=5e-9, unit="s")

    repeats         = 200
    amplitudes      = np.arange(-0.01, 0.011, 0.01) # Reset amplitudes
    samps_per_trig  = 5
    settle_delay    = 50e-6
    circuit_attenuation = 20.0
    res_reference = 1e3
    measure_current = 3e-6
    tc = 50e-6
    fdB = 18

    # Avoid bit depoth problems by scaling here...
    # arb_scale = 0.1

    # Instruments
    arb   = KeysightM8190A("192.168.5.108")
    mag   = AMI430("192.168.5.109")
    # keith = Keithley2400("GPIB0::25::INSTR")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

    polarity = -1

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage'
        descrip.add_axis(DataAxis("sample", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("amplitude", self.amplitudes))
        descrip.add_axis(DataAxis("repeat", range(self.repeats)))
        self.voltage.set_descriptor(descrip)

    def init_instruments(self):
        # ===================
        #    Setup the Lockin
        # ===================
        self.lock.tc = self.tc
        self.lock.filter_slope = self.fdB
        self.lock.amp = self.res_reference * self.measure_current
        time.sleep(0.5)
        # Rescale lockin analogue output for NIDAQ
        self.lock.r_offset_enable = True
        self.lock.auto_offset("R")
        self.lock.r_expand = 10

        self.mag.ramp()

        self.arb.set_output(True, channel=1)
        self.arb.set_output(False, channel=2)
        self.arb.sample_freq = 12.0e9
        self.arb.waveform_output_mode = "WSPEED"
        self.setup_AWG()

        self.analog_input = Task()
        self.read = int32()
        self.buf_points = len(self.amplitudes)*self.samps_per_trig*self.repeats
        self.analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_Diff, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, self.samps_per_trig)
        self.analog_input.CfgInputBuffer(self.buf_points)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        self.analog_input.SetStartTrigRetriggerable(1)
        self.analog_input.StartTask()

        # Assign methods
        self.field.assign_method(self.mag.set_field)
        self.pulse_duration.assign_method(self.setup_AWG)

    def setup_AWG(self, *args):
        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        self.arb.set_output_route("DC", channel=1)
        # self.arb.voltage_amplitude = self.arb_scale

        self.arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
        self.arb.set_marker_level_high(1.5, channel=1, marker_type="sync")

        self.arb.continuous_mode = False
        self.arb.gate_mode = False

        def arb_pulse(amplitude, sample_rate=12e9):
            pulse_points = int(self.pulse_duration.value*sample_rate)

            if pulse_points < 320:
                wf = np.zeros(320)
            else:
                wf = np.zeros(64*int(np.ceil(pulse_points/64.0)))
            wf[:pulse_points] = amplitude
            return wf

        segment_ids = []
        arb_voltage = arb_voltage_lookup()

        scaled_amps = self.amplitudes * np.power(10.0, self.circuit_attenuation/20.0)#self.amplitudes
        for amp in scaled_amps:
            waveform   = arb_pulse(np.sign(amp)*arb_voltage(abs(amp)))
            wf_data    = KeysightM8190A.create_binary_wf_data(waveform)
            segment_id = self.arb.define_waveform(len(wf_data))
            segment_ids.append(segment_id)
            self.arb.upload_waveform(wf_data, segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
        nidaq_trig_segment_id = self.arb.define_waveform(len(nidaq_trig_wf))
        self.arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

        settle_pts = int(640*np.ceil(self.settle_delay * 12e9 / 640))
        start_idxs = [0]

        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=int(self.repeats))
        for si in segment_ids:
            # seq = Sequence(sequence_loop_ct=int(1))
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

    async def run(self):
        # Establish buffers
        buffers = np.empty(self.buf_points)
        self.arb.advance()
        self.arb.trigger()
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                      buffers, self.buf_points, byref(self.read), None)
        logger.debug("Read a buffer of {} points".format(buffers.size))
        await self.voltage.push(buffers)
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

    def shutdown_instruments(self):
        try:
            self.analog_input.StopTask()
        except Exception as e:
            logger.warning("Warning failed to stop task. This is typical.")
            pass

        self.arb.stop()
        self.lock.amp = 0

if __name__ == "__main__":
    sample_name = "CSHE-Die7-C6R7"
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    file_path = "data\CSHE-Switching\{samp:}\{samp:}-SearchReset_{date:}.h5".format(samp=sample_name, date=date)

    exp = ResetSearchLockinExperiment()
    exp.field.value     = 0.007
    exp.pulse_duration.value  = 5e-9
    exp.measure_current = 3e-6
    amps = np.linspace(-0.95, 0.95, 75)
    amps = np.append(amps, np.flipud(amps))
    exp.amplitudes = amps
    exp.init_streams()

    wr         = WriteToHDF5(file_path)
    avg_sample = Averager('sample')
    fig1       = Plotter(name="Sample Averaged", plot_dims=1)
    edges = [(exp.voltage, avg_sample.sink),
             (avg_sample.final_average, wr.sink),
             (avg_sample.partial_average, fig1.sink)]
    exp.set_graph(edges)

    exp.init_progressbar(num=1)
    exp.run_sweeps()

    with h5py.File(wr.filename) as f:
        amps = f['amplitude'].value[:]
        reps = f['repeat'].value[:]
        Vs   = f['data'].value['voltage'][:]
        Vs = Vs.reshape(reps.size, amps.size)
        Vs = Vs>(0.5*(Vs.max()+Vs.min()))
        Vs = 2*Vs-1
        fig = plt.figure(figsize=(4,4))
        plt.ylim(Vs.min()*1.1, Vs.max()*1.1)
        plt.xlabel('Reset Amplitude (V)', size=16)
        plt.ylabel('Final State', size=16)
        plt.title('Reset Pulse Search\n{}'.format(sample_name), size=16)
        plt.plot(amps, Vs.mean(axis=0))
        plt.show()
