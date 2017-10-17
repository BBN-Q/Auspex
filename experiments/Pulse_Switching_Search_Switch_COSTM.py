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
# from auspex.instruments import Keithley2400
from auspex.instruments import AMI430
from auspex.instruments import RFMDAttenuator

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import WriteToHDF5, ProgressBar

from PyDAQmx import *

import asyncio
import numpy as np
import time
import matplotlib.pyplot as plt
from auspex.analysis.h5shell import h5shell

from auspex.log import logger

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Lockin offset/expand (x10) analog output -> DAQmx ai0
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

class SwitchSearchLockinExperiment(Experiment):
    voltage = OutputConnector()

    sample = "CSHE2"
    comment = "Search PSPL Switch Voltage"

    field = FloatParameter(default=0.0, unit="T")
    pulse_voltage  = FloatParameter(default=0, unit="V")
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")

    # Default values for lockin measurement. These will need to be changed in a notebook to match the MR and switching current of the sample being measured
    res_reference = 1e3
    sample_resistance = 50
    measure_current = 10e-6
    fdB = 18
    tc = 100e-3

    circuit_attenuation = 20.0
    pspl_base_attenuation = 30.0

    attempts = 1 << 8 # Number of attemps
    samps_per_trig = 10 # Samples per trigger

    # Instruments
    arb   = KeysightM8190A("192.168.5.108")
    pspl  = Picosecond10070A("GPIB0::24::INSTR")
    mag   = AMI430("192.168.5.109")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")
    atten = RFMDAttenuator("calibration/RFSA2113SB_HPD_20160901.csv")

    min_daq_voltage = 0
    max_daq_voltage = 10

    def init_instruments(self):
        # ===================
        #    Setup the Lockin
        # ===================
        self.lock.tc = self.tc
        self.lock.filter_slope = self.fdB
        self.lock.amp = (self.res_reference + self.sample_resistance) * self.measure_current
        sense_vals = np.array(self.lock.SENSITIVITY_VALUES)
        self.lock.sensitivity = sense_vals[np.argmin(np.absolute(sense_vals-2*self.sample_resistance*self.measure_current*np.ones(sense_vals.size)))]
        time.sleep(20 * self.lock.measure_delay())

        # Rescale lockin analogue output for NIDAQ
        self.lock.r_offset_enable = True
        self.lock.r_expand = 100
        self.lock.auto_offset("R")
        self.lock.r_offset = 0.99*self.lock.r_offset
        #self.lock.r_offset = 100 * ((self.sample_resistance*self.measure_current/self.lock.sensitivity) - (0.05/self.lock.r_expand))
        time.sleep(20 * self.lock.measure_delay())

        # Ramp magnet to set point
        self.mag.ramp()

        #Set attenuator control methods
        self.atten.set_supply_method(self.lock.set_ao2)
        self.atten.set_control_method(self.lock.set_ao3)

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
        # PSPL Max amplitude is 7.5 V
        self.pspl.amplitude = 7.5*np.power(10, -self.pspl_base_attenuation/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.1
        self.pspl.output = True

        self.setup_daq()

        def set_voltage(voltage):
            # Calculate the voltage controller attenuator setting
            self.pspl.amplitude = np.sign(voltage)*7.5*np.power(10, -self.pspl_base_attenuation/20.0)
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - self.pspl_base_attenuation - self.circuit_attenuation

            if vc_atten <= self.atten.minimum_atten():
                logger.error("Voltage controlled attenuation {} under range.".format(vc_atten))
                raise ValueError("Voltage controlled attenuation {} under range.".format(vc_atten))

            if self.atten.maximum_atten() < vc_atten:
                logger.error("Voltage controlled attenuation {} over range.".format(vc_atten))
                raise ValueError("Voltage controlled attenuation {} over range.".format(vc_atten))

            self.atten.set_attenuation(vc_atten)
            time.sleep(0.02)

        # Assign methods
        self.field.assign_method(self.mag.set_field)
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.pulse_voltage.assign_method(set_voltage)

        # Create hooks for relevant delays
        self.pulse_duration.add_post_push_hook(lambda: time.sleep(0.1))

    def setup_daq(self):
        self.arb.abort()
        self.arb.delete_all_waveforms()
        self.arb.reset_sequence_table()

        # Picosecond trigger waveform
        pspl_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), samp_mkr=1)
        pspl_trig_segment_id = self.arb.define_waveform(len(pspl_trig_wf))
        self.arb.upload_waveform(pspl_trig_wf, pspl_trig_segment_id)

        # NIDAQ trigger waveform
        nidaq_trig_wf = KeysightM8190A.create_binary_wf_data(np.zeros(3200), sync_mkr=1)
        nidaq_trig_segment_id = self.arb.define_waveform(len(nidaq_trig_wf))
        self.arb.upload_waveform(nidaq_trig_wf, nidaq_trig_segment_id)

        settle_pts = int(640*np.ceil(self.lock.measure_delay() * 12e9 / 640))
        scenario = Scenario()
        seq = Sequence(sequence_loop_ct=int(self.attempts))
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
        self.buf_points = self.samps_per_trig*self.attempts
        self.analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_Diff,
            self.min_daq_voltage, self.max_daq_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , self.samps_per_trig)
        self.analog_input.CfgInputBuffer(self.buf_points)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
        self.analog_input.SetStartTrigRetriggerable(1)
        self.analog_input.StartTask()

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("sample", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("attempts", range(self.attempts)))
        self.voltage.set_descriptor(descrip)

    async def run(self):
        self.arb.advance()
        self.arb.trigger()
        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        logger.debug("Read a buffer of {} points".format(buf.size))
        await self.voltage.push(buf)
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points()))

    def shutdown_instruments(self):
        try:
            self.analog_input.StopTask()
        except Exception as e:
            logger.warning("Warning failed to stop task, which is quite typical (!)")

        self.arb.stop()
        self.pspl.output = False
        self.lock.amp = 0

if __name__=='__main__':
    exp = SwitchSearchLockinExperiment()
    exp.sample = "CSHE2-C4R2"
    exp.field.value = 0.0133
    exp.measure_current = 3e-6
    exp.init_streams()
    volts = np.arange(-0.7, -0.1, 0.1)
    volts = np.append(volts, -1*np.flipud(volts))
    volts = np.append(-volts, np.flipud(volts))
    durs = 1e-9*np.array([3,5])
    exp.add_sweep(exp.pulse_voltage, volts)
    exp.add_sweep(exp.pulse_duration, durs)

    # Set up measurement network
    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die2-C4R2\CSHE2-C4R2-Search_Switch_2016-06-30.h5")
    # pbar = ProgressBar(num=2)
    # edges = [(exp.voltage, wr.data), (exp.voltage, pbar.data)]
    edges = [(exp.voltage, wr.data)]
    exp.set_graph(edges)
    exp.init_instruments()

    exp.run_sweeps()
    exp.shutdown_instruments()
    # Get data
    f = h5shell(wr.filename,'r')
    dset= f[f.grep('data')[-1]]
    buffers = dset.value
    f.close()
    # Plot the result
    buff_mean = np.mean(buffers, axis=(2,3))
    figs = []
    for res, dur in zip(buff_mean, durs):
        figs.append(plt.figure())
        plt.plot(volts, 1e-3*res/max(exp.measure_current,1e-7),'-o')
        plt.xlabel("AWG amplitude (V)", size=14);
        plt.ylabel("Resistance (kOhm)", size=14);
        plt.title("PSPL Switch Volt Search (dur={})".format(dur))
    logger.info("Finished experiment.")
    # plt.show()
