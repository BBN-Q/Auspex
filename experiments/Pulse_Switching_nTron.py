# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import KeysightM8190A, Scenario, Sequence
from auspex.instruments.alazar import AlazarATS9870, AlazarChannel
from auspex.instruments import Agilent33220A

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Print, WriteToHDF5, Averager, Plotter, Channelizer, KernelIntegrator
from auspex.log import logger

import itertools
import numpy as np
import asyncio
import time, sys

# import h5py
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.interpolate import interp1d

# import auspex.analysis.switching as sw
from adapt import refine

def switching_pulse(amplitude, duration, waveform_duration=500e-09, sample_rate=12e9):
    # total_points = int(waveform_duration*sample_rate)
    pulse_points = int(duration*sample_rate)
    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude
    wf = np.append(np.zeros(1<<12), wf)
    wf = np.append(wf, np.zeros(1<<12))
    # wf = np.append(wf, np.zeros(12000-len(wf)))
    return wf

def measure_pulse(amplitude, duration, frequency, sample_rate=12e9):
    pulse_points = int(duration*sample_rate)
    if pulse_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.ceil(pulse_points/64.0))
    wf[:pulse_points] = amplitude*np.sin(2.0*np.pi*frequency*np.arange(pulse_points)/sample_rate)
    wf = np.append(np.zeros(1<<13), wf)
    # wf = np.append(wf, np.zeros(12000-len(wf)))
    return wf

class nTronSwitchingExperiment(Experiment):

    # Parameters
    channel_bias         = FloatParameter(default=0.2,  unit="V") # On the 33220A
    gate_bias            = FloatParameter(default=0.0,  unit="V") # On the M8190A
    gate_pulse_amplitude = FloatParameter(default=0.0,  unit="V") # On the M8190A
    gate_pulse_duration  = FloatParameter(default=250e-9, unit="s") # On the M8190A

    gate_pulse_durations   = [0]
    gate_pulse_amplitudes  = [0]
    gate_biases            = [0]

    # Constants (set with attribute access if you want to change these!)
    attempts           = 128 #1 << 8
    samples            = 1024 + 16*20
    measure_amplitude  = 0.2
    measure_duration   = 250.0e-9
    measure_frequency  = 100e6

    # Sweep axes
    gate_amps = np.linspace(0.01, 0.02, 25)
    gate_durs = np.linspace(100.0e-9, 500e-9, 3)

    # Things coming back
    voltage     = OutputConnector()

    # Instrument resources
    arb = KeysightM8190A("192.168.5.108")
    awg = Agilent33220A("192.168.5.198")
    alz = AlazarATS9870("1")

    def init_instruments(self):

        self.awg.function       = 'Pulse'
        self.awg.frequency      = 500e3
        self.awg.pulse_width    = 1200e-9
        self.awg.low_voltage    = 0.0
        self.awg.high_voltage   = self.channel_bias.value
        self.awg.burst_state    = True
        self.awg.burst_cycles   = 1
        self.awg.trigger_source = "External"
        self.awg.output         = True

        self.ch = AlazarChannel({'channel': 1})
        self.alz.add_channel(self.ch)
        alz_cfg = {
            'acquire_mode': 'digitizer',
            'bandwidth': 'Full',
            'clock_type': 'ref',
            'delay': 0.0,
            'enabled': True,
            'label': "Alazar",
            'record_length': self.samples,
            'nbr_segments': len(self.gate_amps), #*len(self.gate_durs),
            'nbr_waveforms': self.attempts,
            'nbr_round_robins': 1,
            'sampling_rate': 1e9,
            'trigger_coupling': 'DC',
            'trigger_level': 250,
            'trigger_slope': 'rising',
            'trigger_source': 'Ext',
            'vertical_coupling': 'AC',
            'vertical_offset': 0.0,
            'vertical_scale': 0.4,
        }
        self.alz.set_all(alz_cfg)
        self.loop.add_reader(self.alz.get_socket(self.ch), self.alz.receive_data, self.ch, self.voltage)

        self.arb.set_output(True, channel=2)
        self.arb.set_output(True, channel=1)
        self.arb.sample_freq = 12.0e9
        self.arb.set_waveform_output_mode("WSPEED", channel=1)
        self.arb.set_waveform_output_mode("WSPEED", channel=2)
        self.arb.set_output_route("DC", channel=1)
        self.arb.set_output_route("DC", channel=2)
        self.arb.set_output_complement(False, channel=1)
        self.arb.set_output_complement(False, channel=2)
        self.arb.voltage_amplitude = 1.0
        self.arb.continuous_mode = False
        self.arb.gate_mode = False
        self.arb.set_marker_level_low(0.0, channel=2, marker_type="sync")
        self.arb.set_marker_level_high(1.5, channel=2, marker_type="sync")

        # self.gate_bias.assign_method(self.setup_arb_gate_bias)
        # self.gate_pulse_amplitude.assign_method(self.setup_arb_gate_pulse_amplitude)
        # self.gate_pulse_duration.assign_method(self.setup_arb_gate_pulse_duration)

        self.setup_arb(self.gate_bias.value, self.gate_pulse_amplitude.value, self.gate_pulse_duration.value) # Sequencing goes here

    # def setup_arb_gate_bias(self, gate_bias):
    #     self.setup_arb(gate_bias, self.gate_pulse_amplitude.value, self.gate_pulse_duration.value)
    #
    # def setup_arb_gate_pulse_amplitude(self, gate_pulse_amplitude):
    #     self.setup_arb(self.gate_bias.value, gate_pulse_amplitude, self.gate_pulse_duration.value)
    #
    # def setup_arb_gate_pulse_duration(self, gate_pulse_duration):
    #     self.setup_arb(self.gate_bias.value, self.gate_pulse_amplitude.value, gate_pulse_duration)

    def setup_arb(self, gate_bias, gate_pulse_amplitude, gate_pulse_duration):
        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        seg_ids_ch1 = []
        seg_ids_ch2 = []

        for amp in self.gate_amps:
            # for dur in self.gate_durs:
                # For the measurements pulses along the channel
            wf      = measure_pulse(amplitude=self.measure_amplitude, duration=self.measure_duration, frequency=self.measure_frequency)
            wf_data = KeysightM8190A.create_binary_wf_data(wf)
            seg_id  = self.arb.define_waveform(len(wf_data), channel=1)
            self.arb.upload_waveform(wf_data, seg_id, channel=1)
            seg_ids_ch1.append(seg_id)

            # For the switching pulses along the gate
            wf      = switching_pulse(amplitude=amp, duration=gate_pulse_duration)
            wf_data = KeysightM8190A.create_binary_wf_data(wf, sync_mkr=1)
            seg_id  = self.arb.define_waveform(len(wf_data), channel=2)
            self.arb.upload_waveform(wf_data, seg_id, channel=2)
            seg_ids_ch2.append(seg_id)

        # Build in a delay between sequences
        settle_pts = int(640*np.ceil(2e-6 * 12e9 / 640))

        print(len(wf))
        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=self.attempts)
        for si in seg_ids_ch1:
            seq.add_waveform(si)
            seq.add_idle(settle_pts, 0.0)
        scenario.sequences.append(seq)
        self.arb.upload_scenario(scenario, start_idx=0, channel=1)

        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=self.attempts)
        for si in seg_ids_ch2:
            seq.add_waveform(si)
            seq.add_idle(settle_pts, 0.0)
        scenario.sequences.append(seq)
        self.arb.upload_scenario(scenario, start_idx=0, channel=2)

        self.arb.set_sequence_mode("SCENARIO", channel=1)
        self.arb.set_scenario_advance_mode("SINGLE", channel=1)
        self.arb.set_scenario_start_index(0, channel=1)
        self.arb.set_sequence_mode("SCENARIO", channel=2)
        self.arb.set_scenario_advance_mode("SINGLE", channel=2)
        self.arb.set_scenario_start_index(0, channel=2)
        self.arb.initiate(channel=1)
        self.arb.initiate(channel=2)
        self.arb.advance()

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("time", 1e-9*np.arange(self.samples)))
        # descrip.add_axis(DataAxis("gate_pulse_duration", self.gate_durs))
        descrip.add_axis(DataAxis("gate_pulse_amplitude", self.gate_amps))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))

        self.voltage.set_descriptor(descrip)

    async def run(self):
        # self.arb.stop()
        self.arb.set_scenario_start_index(0, channel=1)
        self.arb.set_scenario_start_index(0, channel=2)
        self.arb.advance()
        self.alz.acquire()
        await asyncio.sleep(0.2)
        self.arb.trigger()
        await self.alz.wait_for_acquisition(10.0)
        await asyncio.sleep(0.4)
        self.alz.stop()
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
        logger.info("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))


    def shutdown_instruments(self):
        self.awg.output = False

        self.arb.stop()
        self.loop.remove_reader(self.alz.get_socket(self.ch))

        for name, instr in self._instruments.items():
            instr.disconnect()


if __name__ == '__main__':
    exp = nTronSwitchingExperiment()
    plot = Plotter(name="Demod!", plot_mode="real", plot_dims=2)
    plot_ki = Plotter(name="Ki!", plot_mode="real")
    plot_avg = Plotter(name="Avg!", plot_mode="real")
    plot_raw1 = Plotter(name="Raw!", plot_mode="real", plot_dims=1)
    plot_raw2 = Plotter(name="Raw!", plot_mode="real", plot_dims=2)
    demod = Channelizer(frequency=exp.measure_frequency, decimation_factor=4, bandwidth=20e6)
    kernel_params = {}
    kernel_params['kernel'] = 0
    kernel_params['bias'] = 0
    kernel_params['simple_kernel'] = True
    kernel_params['box_car_start'] = 8e-7
    kernel_params['box_car_stop']  = 10.5e-7
    kernel_params['frequency'] = 0.0

    ki = KernelIntegrator(**kernel_params)
    avg = Averager(axis="attempt")
    edges = [(exp.voltage, demod.sink),
            (exp.voltage, plot_raw1.sink),
            (demod.source, ki.sink),
            (ki.source, plot_ki.sink),
            (ki.source, avg.sink),
            (avg.final_average, plot_avg.sink),
            (demod.source, plot.sink)]
    exp.set_graph(edges)

    exp.run_sweeps()
