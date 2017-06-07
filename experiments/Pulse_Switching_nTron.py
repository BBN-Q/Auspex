# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# 0.1 Hz 6dB slope HPF
# 300 kHz 6dB slope LPF

from auspex.instruments import KeysightM8190A, Scenario, Sequence
from auspex.instruments.alazar import AlazarATS9870, AlazarChannel
from auspex.instruments import Agilent33220A

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Print, WriteToHDF5, Averager, Plotter, Channelizer, KernelIntegrator
from auspex.log import logger

import numpy as np
import asyncio
import time, sys, datetime

def switching_pulse(amplitude, duration, delay=25e-9, holdoff=800e-9, total_duration=1200e-09, sample_rate=12e9):
    total_points = int(total_duration*sample_rate)
    pulse_points = int(duration*sample_rate)
    delay_points = int(delay*sample_rate)
    hold_points  = int(holdoff*sample_rate)

    pad_points = hold_points - delay_points - pulse_points
    if total_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.int(np.ceil(total_points/64.0)))
    wf[pad_points:pad_points+pulse_points] = amplitude
    return wf

def measure_pulse(amplitude, duration, frequency, holdoff=800e-9, total_duration=1200e-09, sample_rate=12e9):
    total_points = int(total_duration*sample_rate)
    pulse_points = int(duration*sample_rate)
    hold_points  = int(holdoff*sample_rate)
    if total_points < 320:
        wf = np.zeros(320)
    else:
        wf = np.zeros(64*np.int(np.ceil(total_points/64.0)))
    wf[hold_points:hold_points+pulse_points] = amplitude*np.sin(2.0*np.pi*frequency*np.arange(pulse_points)/sample_rate)
    return wf

class nTronSwitchingExperiment(Experiment):

    # Parameters
    channel_bias         = FloatParameter(default=0.05,  unit="V") # On the 33220A
    gate_bias            = FloatParameter(default=0.0,  unit="V") # On the M8190A

    # Constants (set with attribute access if you want to change these!)
    attempts           = 1 << 8
    samples            = 384 #1024 + 16*20
    measure_amplitude  = 0.1
    measure_duration   = 250.0e-9
    measure_frequency  = 100e6

    # arb
    sample_rate = 12e9
    repeat_time = 4*2.4e-6 # Picked very carefully for 100ns alignment

    # Things coming back
    voltage     = OutputConnector()

    # Instrument resources
    arb = KeysightM8190A("192.168.5.108")
    awg = Agilent33220A("192.168.5.198")
    alz = AlazarATS9870("1")

    def __init__(self, gate_amps, gate_durs):
        self.gate_amps = gate_amps
        self.gate_durs = gate_durs
        super(nTronSwitchingExperiment, self).__init__()

    def init_instruments(self):

        self.awg.function       = 'Pulse'
        self.awg.frequency      = 0.5e6
        self.awg.pulse_width    = 1e-6
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
            'delay': 850e-9,
            'enabled': True,
            'label': "Alazar",
            'record_length': self.samples,
            'nbr_segments': len(self.gate_amps)*len(self.gate_durs),
            'nbr_waveforms': 1,
            'nbr_round_robins': self.attempts,
            'sampling_rate': 1e9,
            'trigger_coupling': 'DC',
            'trigger_level': 125,
            'trigger_slope': 'rising',
            'trigger_source': 'Ext',
            'vertical_coupling': 'AC',
            'vertical_offset': 0.0,
            'vertical_scale': 0.1,
        }
        self.alz.set_all(alz_cfg)
        self.loop.add_reader(self.alz.get_socket(self.ch), self.alz.receive_data, self.ch, self.voltage)

        self.arb.set_output(True, channel=2)
        self.arb.set_output(True, channel=1)
        self.arb.sample_freq = self.sample_rate
        self.arb.set_waveform_output_mode("WSPEED", channel=1)
        self.arb.set_waveform_output_mode("WSPEED", channel=2)
        self.arb.set_output_route("DC", channel=1)
        self.arb.set_output_route("DC", channel=2)
        self.arb.set_output_complement(False, channel=1)
        self.arb.set_output_complement(False, channel=2)
        self.arb.set_voltage_amplitude(1.0, channel=1)
        self.arb.set_voltage_amplitude(1.0, channel=2)
        self.arb.continuous_mode = False
        self.arb.gate_mode = False
        self.arb.set_marker_level_low(0.0, channel=1, marker_type="sync")
        self.arb.set_marker_level_high(1.5, channel=1, marker_type="sync")
        self.arb.set_marker_level_low(0.0, channel=2, marker_type="sync")
        self.arb.set_marker_level_high(1.5, channel=2, marker_type="sync")

        self.setup_arb() #self.gate_bias.value, self.gate_pulse_amplitude.value, self.gate_pulse_duration.value) # Sequencing goes here

    def setup_arb(self): #, gate_bias, gate_pulse_amplitude, gate_pulse_duration):
        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        seg_ids_ch1 = []
        seg_ids_ch2 = []

        # For the measurements pulses along the channel
        wf      = measure_pulse(amplitude=self.measure_amplitude, duration=self.measure_duration, frequency=self.measure_frequency)
        wf_data = KeysightM8190A.create_binary_wf_data(wf, sync_mkr=1)
        seg_id  = self.arb.define_waveform(len(wf_data), channel=1)
        self.arb.upload_waveform(wf_data, seg_id, channel=1)
        seg_ids_ch1.append(seg_id)

        # Build in a delay between sequences
        settle_pts = 640*np.int(np.ceil(self.repeat_time * self.sample_rate / 640))
        # settle_pts2 = 640*np.ceil(8*2.4e-9 * self.sample_rate / 640)

        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=self.attempts*len(self.gate_amps)*len(self.gate_durs))
        for si in seg_ids_ch1:
            seq.add_waveform(si)
            seq.add_idle(settle_pts, 0.0)
        scenario.sequences.append(seq)
        self.arb.upload_scenario(scenario, start_idx=0, channel=1)

        for amp in self.gate_amps:
            for dur in self.gate_durs:
                # For the switching pulses along the gate
                wf      = switching_pulse(amplitude=amp, duration=dur) #self.gate_pulse_duration.value)
                wf_data = KeysightM8190A.create_binary_wf_data(wf)
                seg_id  = self.arb.define_waveform(len(wf_data), channel=2)
                self.arb.upload_waveform(wf_data, seg_id, channel=2)
                seg_ids_ch2.append(seg_id)

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
        if len(self.gate_durs) > 1:
            descrip.add_axis(DataAxis("gate_pulse_duration", self.gate_durs))
        descrip.add_axis(DataAxis("gate_pulse_amplitude", self.gate_amps))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))

        self.voltage.set_descriptor(descrip)

    async def run(self):
        # self.arb.stop()
        self.arb.set_scenario_start_index(0, channel=1)
        self.arb.set_scenario_start_index(0, channel=2)
        self.arb.advance()
        await asyncio.sleep(0.3)
        self.alz.acquire()
        await asyncio.sleep(0.3)
        self.arb.trigger()
        await self.alz.wait_for_acquisition(10.0)
        await asyncio.sleep(0.8)
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
    plot_ki = Plotter(name="Ki!", plot_mode="real", plot_dims=2)
    plot_avg = Plotter(name="Avg!", plot_mode="real")
    plot_raw1 = Plotter(name="Raw!", plot_mode="real", plot_dims=1)
    plot_raw2 = Plotter(name="Raw!", plot_mode="real", plot_dims=2)
    demod = Channelizer(frequency=exp.measure_frequency, decimation_factor=4, bandwidth=20e6)

    ki = KernelIntegrator(kernel=0, bias=0, simple_kernel=True, box_car_start=1e-7, box_car_stop=3.8e-7, frequency=0.0)
    avg = Averager(axis="attempt")

    samp      = "c1r4"
    file_path = f"data\\nTron-Switching\\{samp}\\{samp}-PulseSwitchingShort-{datetime.datetime.today().strftime('%Y-%m-%d')}.h5"
    # file_path = f"data\\nTron-Switching\\{samp}\\{samp}-PulseSwitching-{datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')}.h5"
    wr_int    = WriteToHDF5(file_path, groupname="Integrated", store_tuples=False)
    wr_final  = WriteToHDF5(file_path, groupname="Final", store_tuples=False)
    wr_raw    = WriteToHDF5(file_path, groupname="Raw", store_tuples=False)

    edges = [(exp.voltage, demod.sink),
            # (exp.voltage, wr_raw.sink),
            (exp.voltage, plot_raw1.sink),
            # (exp.voltage, plot_raw2.sink),
            (demod.source, ki.sink),
            # (ki.source, plot_ki.sink),
            (ki.source, avg.sink),
            (ki.source, wr_int.sink),
            (avg.final_average, plot_avg.sink),
            (demod.source, plot.sink),
            ]
    exp.set_graph(edges)

    exp.run_sweeps()
