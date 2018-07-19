# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import KeysightM8190A, Scenario, Sequence
from auspex.instruments import Picosecond10070A
from auspex.instruments import SR865
from auspex.instruments import Keithley2400
from auspex.instruments import AMI430
from auspex.instruments import Attenuator

from PyDAQmx import *

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import WriteToHDF5

from auspex.experiment import FloatParameter, Experiment
from auspex.stream import DataAxis, OutputConnector, DataStreamDescriptor
from auspex.filters.io import WriteToHDF5
from auspex.filters.average import Averager
from auspex.filters.plot import Plotter
from auspex.log import logger

import auspex.analysis.switching as sw
from adapt import refine


import itertools
import numpy as np
import time
import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import h5py

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

def arb_voltage_lookup(arb_calib="calibration/AWG_20160901.csv"):
    df_arb = pd.read_csv(arb_calib, sep=",")
    return interp1d(df_arb["Amp Out"], df_arb["Control Voltage"])

class SwitchingExperiment(Experiment):

    # Parameters and outputs
    field          = FloatParameter(default=0.0, unit="T")
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=0.1, unit="V")
    voltage        = OutputConnector()

    # Constants (set with attribute access if you want to change these!)
    attempts        = 1 << 10
    settle_delay    = 100e-6
    measure_current = 3.0e-6
    samps_per_trig  = 5
    polarity        = 1
    pspl_atten      = 4
    min_daq_voltage = 0.0
    max_daq_voltage = 0.4
    reset_amplitude = 0.2
    reset_duration  = 5.0e-9

    # Instrument Resources
    mag   = AMI430("192.168.5.109")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")
    pspl  = Picosecond10070A("GPIB0::24::INSTR")
    atten = Attenuator("calibration/RFSA2113SB_HPD_20160901.csv", lock.set_ao2, lock.set_ao3)
    arb   = KeysightM8190A("192.168.5.108")
    keith = Keithley2400("GPIB0::25::INSTR")

    def init_streams(self):
        self.voltage.add_axis(DataAxis("sample", range(self.samps_per_trig)))
        self.voltage.add_axis(DataAxis("state", range(2)))
        self.voltage.add_axis(DataAxis("attempt", range(self.attempts)))

    def init_instruments(self):

        # Setup the Keithley
        self.keith.triad()
        self.keith.conf_meas_res(res_range=1e5)
        self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
        self.keith.current = self.measure_current
        self.mag.ramp()

        # Setup the AWG
        self.arb.set_output(True, channel=1)
        self.arb.set_output(False, channel=2)
        self.arb.sample_freq = 12.0e9
        self.arb.waveform_output_mode = "WSPEED"
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
            arb_voltage = arb_voltage_lookup()
            pulse_points = int(duration*sample_rate)
            if pulse_points < 320:
                wf = np.zeros(320)
            else:
                wf = np.zeros(64*np.ceil(pulse_points/64.0))
            wf[:pulse_points] = np.sign(amplitude)*arb_voltage(abs(amplitude))
            return wf

        reset_wf    = arb_pulse(-self.polarity*self.reset_amplitude, self.reset_duration)
        wf_data     = KeysightM8190A.create_binary_wf_data(reset_wf)
        rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, rst_segment_id)

        # no_reset_wf = arb_pulse(0.0, 3.0/12e9)
        # wf_data     = KeysightM8190A.create_binary_wf_data(no_reset_wf)
        # no_rst_segment_id  = self.arb.define_waveform(len(wf_data))
        # self.arb.upload_waveform(wf_data, no_rst_segment_id)

        # Picosecond trigger waveform
        pspl_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), samp_mkr=1)
        pspl_trig_segment_id = self.arb.define_waveform(len(pspl_trig_wf))
        self.arb.upload_waveform(pspl_trig_wf, pspl_trig_segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
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
        seq.add_waveform(pspl_trig_segment_id)
        seq.add_idle(settle_pts, 0.0)
        seq.add_waveform(nidaq_trig_segment_id)
        seq.add_idle(1 << 16, 0.0) # bonus non-contiguous memory delay
        scenario.sequences.append(seq)
        self.arb.upload_scenario(scenario, start_idx=0)
        self.arb.sequence_mode = "SCENARIO"
        self.arb.scenario_advance_mode = "REPEAT"
        self.arb.scenario_start_index = 0
        self.arb.run()

        # Setup the NIDAQ
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

        # Setup the PSPL
        self.pspl.amplitude = self.polarity*7.5*np.power(10, (-self.pspl_atten)/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.1
        self.pspl.output = True

        def set_voltage(voltage):
            # Calculate the voltage controller attenuator setting
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - self.pspl_atten - 0
            if vc_atten <= 6.0:
                raise ValueError("Voltage controlled attenuation under range (6dB).")
            self.atten.set_attenuation(vc_atten)
            time.sleep(0.02)

        # Assign methods
        self.field.assign_method(self.mag.set_field)
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.pulse_voltage.assign_method(set_voltage)

        # Create hooks for relevant delays
        self.pulse_duration.add_post_push_hook(lambda: time.sleep(0.1))

    def run(self):
        """This is run for each step in a sweep."""
        self.arb.advance()
        self.arb.trigger()
        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        self.voltage.push(buf)
        # Seemingly we need to give the filters some time to catch up here...
        time.sleep(0.02)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

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

if __name__ == '__main__':

    exp = SwitchingExperiment()
    exp.field.value     = 0.007
    exp.polarity        = 1 # 1: AP to P; -1: P to AP
    exp.measure_current = 3e-6
    exp.reset_amplitude = 0.78
    exp.reset_duration  = 5.0e-9
    exp.settle_delay    = 200e-6
    exp.pspl_atten      = 3
    max_points          = 100

    sample_name = "CSHE-Die7-C6R7"
    date        = datetime.datetime.today().strftime('%Y-%m-%d')
    pol         = "APtoP" if exp.polarity < 0 else "PtoAP"
    file_path   = "data\CSHE-Switching\{samp:}\{samp:}-PulseSwitching-{pol:}_{date:}.h5".format(pol=pol,samp=sample_name, date=date)

    wr   = WriteToHDF5(file_path)
    avg  = Averager('sample')

    edges = [(exp.voltage, avg.sink),
             (avg.final_average, wr.sink)]
    exp.set_graph(edges)

    # Construct the coarse grid
    coarse_ts = np.linspace(0.1, 10.0, 7)*1e-9
    coarse_vs = np.linspace(0.40, 0.90, 7)
    points    = [coarse_ts, coarse_vs]
    points    = list(itertools.product(*points))

    # Add an extra plotter
    fig1 = MeshPlotter(name="Switching Phase Diagram")

    def refine_func(sweep_axis):
        points, mean = sw.load_switching_data(wr.filename)
        new_points   = refine.refine_scalar_field(points, mean, all_points=False,
                                    criterion="integral", threshold = "one_sigma")
        if len(points) + len(new_points) > max_points:
            print("Reached maximum points ({}).".format(max_points))
            return False
        print("Reached {} points.".format(len(points) + len(new_points)))
        sweep_axis.add_points(new_points)

        # Plot previous mesh
        x = [list(el) for el in points[mesh.simplices,0]]
        y = [list(el) for el in points[mesh.simplices,1]]
        val = [np.mean(vals) for vals in mean[mesh.simplices]]

        desc = DataStreamDescriptor()
        desc.add_axis(sweep_axis)
        exp.push_to_plot(fig1, desc, points)

        time.sleep(1)
        return True

    sweep_axis = exp.add_sweep([exp.pulse_duration, exp.pulse_voltage],
                               points, refine_func=refine_func)

    # Borrow the descriptor from the main sweep and use it for our direct plotter

    exp.add_plotter(fig1, desc)

    exp.run_sweeps()

    points, mean = sw.load_switching_data(wr.filename)
    mesh, scale_factors = sw.scaled_Delaunay(points)
    fig_mesh = sw.phase_diagram_mesh(points, mean, shading='gouraud')
    plt.triplot(mesh.points[:,0]/scale_factors[0],
                mesh.points[:,1]/scale_factors[1], mesh.simplices.copy());

    plt.show()

    # t1 = time.time()
    # for i in range(exp.iterations):
    #     exp.reset()
    #     exp.run_sweeps()
    #     points, mean = sw.load_switching_data(wr.filename)
    #     figs.append(sw.phase_diagram_mesh(points, mean, title="Iteration={}".format(i)))
    #     new_points = refine.refine_scalar_field(points, mean, all_points=False,
    #                                 criterion="integral", threshold = "one_sigma")
    #     if new_points is None:
    #         print("No more points can be added.")
    #         break
    #     print("Added {} new points.".format(len(new_points)))
    #     print("Elapsed time: {}".format((time.time()-t1)/60))
    #     time.sleep(3)
    #     main_sweep.update_values(new_points)

    # t2 = time.time()
    # Shut down
    # exp.shutdown_instruments()
    # For evaluation of adaptive method, plot the mesh
