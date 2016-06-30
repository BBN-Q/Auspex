from pycontrol.instruments.keysight import *
from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430
from pycontrol.instruments.rfmd import Attenuator

from pycontrol.experiment import FloatParameter, IntParameter, Experiment
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print
from pycontrol.filters.io import WriteToHDF5

from PyDAQmx import *

import itertools
import numpy as np
import asyncio
import time, sys
import h5py
import matplotlib.pyplot as plt

import analysis.switching as sw
from adapt import refine

from pycontrol.logging import logger

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

class SwitchingExperiment(Experiment):

    # Sample information
    sample         = "CSHE2-C3R6"
    comment        = "Phase Diagram"
    # Parameters
    field          = FloatParameter(default=0.0, unit="T")
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=0.1, unit="V")

    # Constants (set with attribute access if you want to change these!)
    iteration       = 8
    attempts        = 1 << 10
    settle_delay    = 50e-6
    measure_current = 3.0e-6
    samps_per_trig  = 5

    polarity        = 1
    pspl_atten      = 0

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
    atten = Attenuator("calibration/RFSA2113SB.tsv", lock.set_ao2, lock.set_ao3)
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
                wf = np.zeros(64*np.ceil(pulse_points/64.0))
            wf[:pulse_points] = amplitude
            return wf

        reset_wf    = arb_pulse(-self.polarity*self.reset_amplitude, self.reset_duration)
        wf_data     = M8190A.create_binary_wf_data(reset_wf)
        rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, rst_segment_id)

        no_reset_wf = arb_pulse(0.0, 3.0/12e9)
        wf_data     = M8190A.create_binary_wf_data(no_reset_wf)
        no_rst_segment_id  = self.arb.define_waveform(len(wf_data))
        self.arb.upload_waveform(wf_data, no_rst_segment_id)

        # Picosecond trigger waveform
        pspl_trig_wf = M8190A.create_binary_wf_data(np.zeros(3200), samp_mkr=1)
        pspl_trig_segment_id = self.arb.define_waveform(len(pspl_trig_wf))
        self.arb.upload_waveform(pspl_trig_wf, pspl_trig_segment_id)

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

        # ===================
        #   Setup the PSPL
        # ===================

        self.pspl.amplitude = self.polarity*7.5*np.power(10, (-self.pspl_atten)/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.1
        self.pspl.output = True

        def set_voltage(voltage):
            # Calculate the voltage controller attenuator setting
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - self.pspl_atten - 10
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

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("state", range(2)))
        descrip.add_axis(DataAxis("attempts", range(self.attempts)))
        self.daq_buffer.set_descriptor(descrip)

    async def run(self):
        """This is run for each step in a sweep."""
        self.arb.advance()
        self.arb.trigger()
        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        await self.daq_buffer.push(buf)
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
        logger.debug("Stream has filled {} of {} points".format(self.daq_buffer.points_taken, self.daq_buffer.num_points() ))

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
    exp.sample = "CSHE2-C3R6"
    exp.comment = "Phase Diagram -  AP to P - Interations = 12"
    exp.polarity = 1 # 1: AP to P; -1: P to AP
    exp.iteration = 12
    exp.field.value = 0.0126
    exp.measure_current = 3e-6
    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die2-C3R6\CSHE2-C3R6-AP2P_2016-06-29.h5")
    # pr = Print()
    edges = [(exp.daq_buffer, wr.data)]
    exp.set_graph(edges)
    exp.init_instruments()

    coarse_ts = 1e-9*np.linspace(0.1, 5, 10) # List of durations
    coarse_vs = np.linspace(0.5, 1.15, 10)
    points    = [coarse_ts, coarse_vs]
    points    = list(itertools.product(*points))

    main_sweep = exp.add_unstructured_sweep([exp.pulse_duration, exp.pulse_voltage], points)
    figs = []
    t1 = time.time()
    for i in range(exp.iteration):
        exp.reset()
        exp.run_sweeps()
        points, mean = sw.load_switching_data(wr.filename)
        figs.append(sw.phase_diagram_mesh(points, mean, title="Iteration={}".format(i)))
        new_points = refine.refine_scalar_field(points, mean, all_points=False,
                                    criterion="integral", threshold = "one_sigma")
        if new_points is None:
            print("No more points can be added.")
            break
        #
        print("Added {} new points.".format(len(new_points)))
        print("Elapsed time: {}".format((time.time()-t1)/60))
        time.sleep(3)
        main_sweep.update_values(new_points)

    t2 = time.time()
    # Shut down
    exp.shutdown_instruments()
    # For evaluation of adaptive method, plot the mesh
    mesh, scale_factors = sw.scaled_Delaunay(points)
    fig_mesh = sw.phase_diagram_mesh(points, mean, shading='gouraud')
    plt.triplot(mesh.points[:,0]/scale_factors[0],
                mesh.points[:,1]/scale_factors[1], mesh.simplices.copy());

    # plt.show()
