# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from PyDAQmx import *
from PyDAQmx.DAQmxCallBack import *

from auspex.instruments import Agilent33220A
from auspex.instruments import Picosecond10070A
from auspex.instruments import SR865
from auspex.instruments import RFMDAttenuator
from auspex.instruments import Keithley2400
from auspex.instruments import AMI430

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Print, WriteToHDF5, Averager, Plotter, Channelizer, KernelIntegrator
from auspex.log import logger

import numpy as np
import asyncio
import time, sys, datetime

# Experimental Topology
# NIDAQ P1.0 -> Agilent33220A Trigger
# NIDAQ P1.1 -> 10dB Attenuator -> PSPL Trigger
# NIDAQ P1.1 -> NIDAQ PFI1
# Agilent -> 10 kOhm -> Bias-T -> 50 Ohm -> nTron
#                       |          |
#                  Prog Atten   NIDAQ AI0
#                       |
#                     PSPL
# Keithley2400 (as current source) -> CSHE top lead
#                                    |
#                                  NiDAQ AI2

class CallbackTask(Task):
    def __init__(self, channels, loop, points_per_trigger, attempts, output_connectors, trigger="/Dev1/PFI1", min_voltage=-10.0, max_voltage=10.0, ai_clock=1e6):
        Task.__init__(self)
        self.channels = channels
        self.loop = loop # the asyncio loop to push data on
        self.points_per_trigger = points_per_trigger # account for double measurement
        self.attempts = attempts
        self.output_connectors = output_connectors

        # Create an analog channel for each entry in channels
        self.num_channels = len(self.channels)
        self.buffer = np.empty(self.points_per_trigger*self.attempts*self.num_channels)

        # Construct our specific task.
        self.buff_points = self.points_per_trigger*self.attempts*self.num_channels
        for c in self.channels:
            self.CreateAIVoltageChan(c, "", DAQmx_Val_Diff, min_voltage, max_voltage, DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", ai_clock, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, self.points_per_trigger)
        self.CfgInputBuffer(self.buff_points)
        self.CfgDigEdgeStartTrig(trigger, DAQmx_Val_Rising)
        self.SetStartTrigRetriggerable(1)
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,self.points_per_trigger*self.attempts,0)
        self.AutoRegisterDoneEvent(0)

        # print("Expected to receive ", self.buff_points, "per point.")
        self.points_read = 0
        self.num_callbacks = 0

    def EveryNCallback(self):
        read = int32()
        self.num_callbacks += 1
        # import pdb; pdb.set_trace()
        # print("read {} so far".format(self.points_read))
        self.ReadAnalogF64(self.points_per_trigger*self.attempts, 5.0, DAQmx_Val_GroupByChannel,
                                        self.buffer, self.buff_points, byref(read), None)
        for i, oc in enumerate(self.output_connectors):
            dat = self.buffer[i*self.points_per_trigger*self.attempts:(i+1)*self.points_per_trigger*self.attempts]
            asyncio.ensure_future(oc.push(dat), loop=self.loop)
        self.points_read += read.value
        # print("now read {}".format(self.points_read))
        return 0

    def DoneCallback(self, status):
        return 0

class ShortProcessSwitchingExperiment(Experiment):

    # Parameters and outputs
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=1, unit="V")
    bias_current   = FloatParameter(default=60e-6, unit="A")
    field          = FloatParameter(default=0.0, unit="T")
    voltage_chan   = OutputConnector()
    #voltage_MTJ    = OutputConnector()
    resistance_MTJ = OutputConnector()

    # Constants (set with attribute access if you want to change these!)
    attempts           = 1 << 4

    # MTJ measurements
    measure_current = 3.0e-6

    # Reference resistances and attenuations
    matching_ref_res = 50
    chan_bias_ref_res = 1e4
    pspl_base_attenuation = 10
    circuit_attenuation = 0

    # Min/Max NIDAQ AI Voltage
    min_daq_voltage = -1
    max_daq_voltage = 1

    # Measurement Sequence timing, clocks in Hz times in seconds
    ai_clock = 0.25e6
    do_clock = 1e5
    run_time = 2e-4
    settle_time = 0.5*run_time
    integration_time = 0.1*run_time
    pspl_delay = 0.25*0.5*run_time

    # Instrument resources, NIDAQ called via PyDAQmx
    arb_cb = Agilent33220A("192.168.5.199") # Channel Bias
    pspl  = Picosecond10070A("GPIB0::24::INSTR") # Gate Pulse
    atten = RFMDAttenuator("calibration/RFSA2113SB_HPD_20160901.csv") # Gate Amplitude control
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR") # Gate Amplitude control
    keith = Keithley2400("GPIB0::25::INSTR")
    mag   = AMI430("192.168.5.109")

    def init_instruments(self):
        # Channel bias arb
        self.arb_cb.output          = False
        self.arb_cb.load_resistance = (self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.function        = 'Pulse'
        self.arb_cb.pulse_period    = 0.99*self.run_time
        self.arb_cb.pulse_width     = self.run_time - self.settle_time
        self.arb_cb.pulse_edge      = 100e-9
        self.arb_cb.low_voltage     = 0.0
        self.arb_cb.high_voltage    = self.bias_current.value*(self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.polarity        = 1
        self.arb_cb.burst_state     = True
        self.arb_cb.burst_cycles    = 1
        self.arb_cb.trigger_source  = "External"
        self.arb_cb.burst_mode      = "Triggered"
        self.arb_cb.output          = True

        # Setup the Keithley
        self.keith.triad()
        self.keith.conf_meas_res(res_range=1e5)
        self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
        self.keith.current = self.measure_current

        # Setup the magnet
        self.mag.ramp()

        # Setup the NIDAQ tasks
        DAQmxResetDevice("Dev1")
        self.nidaq = CallbackTask(["Dev1/ai0"], asyncio.get_event_loop(), int(self.integration_time*self.ai_clock), self.attempts, [self.voltage_chan],
                                 min_voltage=self.min_daq_voltage, max_voltage=self.max_daq_voltage, ai_clock=self.ai_clock)
        self.nidaq.StartTask()


        # Setup DO Triggers, P1:0 triggers AWG, P1:1 triggers PSPL and DAQ analog input
        self.digital_output = Task()
        data_p0 = np.zeros(int(self.do_clock*self.run_time),dtype=np.uint8)
        data_p0[0]=1

        data_p1 = np.zeros(int(self.do_clock*self.run_time),dtype=np.uint8)
        data_p1[int(self.do_clock*(self.pspl_delay))]=1 # PSPL delay

        data = np.hstack([data_p0,data_p1])

        self.digital_output.CreateDOChan("/Dev1/port0/line0:1","",DAQmx_Val_ChanPerLine)
        self.digital_output.CfgSampClkTiming("",self.do_clock, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, int(data.size/2)*self.attempts)
        self.digital_output.WriteDigitalLines(int(self.do_clock*self.run_time),0,1,DAQmx_Val_GroupByChannel,data,None,None)

        # Setup the PSPL
        self.pspl.amplitude = 7.5*np.power(10, (-self.pspl_base_attenuation)/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.5
        self.pspl.output = True

        # Setup PSPL Attenuator Control
        self.atten.set_supply_method(self.lock.set_ao2)
        self.atten.set_control_method(self.lock.set_ao3)

        def set_pulse_voltage(voltage, base_atten=self.pspl_base_attenuation):
            # Calculate the voltage controller attenuator setting
            amplitude = 7.5*np.power(10, -base_atten/20.0)
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - base_atten - self.circuit_attenuation

            if vc_atten <= self.atten.minimum_atten():
                base_atten -= 1
                if base_atten < 0:
                    logger.error("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                set_pulse_voltage(voltage, base_atten=base_atten)
                return

            if self.atten.maximum_atten() < vc_atten:
                base_atten += 1
                if base_atten > 80:
                    logger.error("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                set_pulse_voltage(voltage, base_atten=base_atten)
                return

            #print("PSPL Amplitude: {}, Attenuator: {}".format(amplitude,vc_atten))
            self.atten.set_attenuation(vc_atten)
            self.pspl.amplitude = self.arb_cb.polarity*abs(amplitude)
            time.sleep(0.04)

        def set_awg_current(current):

            if 0 <= current:
                self.arb_cb.polarity = 1
                self.arb_cb.low_voltage = 0
                self.arb_cb.high_voltage = current*self.chan_bias_ref_res
            else :
                self.arb_cb.polarity = -1
                self.arb_cb.low_voltage = current*self.chan_bias_ref_res
                self.arb_cb.high_voltage = 0

            self.pspl.amplitude = self.arb_cb.polarity*abs(self.pspl.amplitude)

            time.sleep(0.04)

        # Assign Methods
        self.bias_current.assign_method(set_awg_current)
        self.pulse_duration.add_post_push_hook(lambda: time.sleep(0.1))
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.pulse_voltage.assign_method(set_pulse_voltage)
        self.field.assign_method(self.mag.set_field)

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage'
        descrip.add_axis(DataAxis("sample", range(int(self.integration_time*self.ai_clock))))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))
        self.voltage_chan.set_descriptor(descrip)

        # descrip = DataStreamDescriptor()
        # descrip.data_name='voltage'
        # descrip.add_axis(DataAxis("sample", range(int(self.integration_time*self.ai_clock))))
        # descrip.add_axis(DataAxis("attempt", range(self.attempts)))
        # self.voltage_MTJ.set_descriptor(descrip)

        descrip = DataStreamDescriptor()
        descrip.data_name='resistance'
        self.resistance_MTJ.set_descriptor(descrip)

    async def run(self):
        self.digital_output.StartTask()
        self.digital_output.WaitUntilTaskDone(2*self.attempts*self.run_time)
        self.digital_output.StopTask()
        await self.resistance_MTJ.push(self.keith.resistance)
        await asyncio.sleep(0.05)
        # print("\t now ", self.nidaq.points_read)
        # logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

    def shutdown_instruments(self):
        self.keith.current = 0.0e-5
        self.mag.zero()
        try:
            for ch in [self.nidaq, self.nidaq_MTJ]:
                ch.StopTask()
                ch.ClearTask()
                self.digital_output.StopTask()
                del ch
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass
        self.arb_cb.output = False
        self.pspl.output = False
        for name, instr in self._instruments.items():
            instr.disconnect()

class ShortProcessSwitchingExperimentReset(Experiment):

    # Parameters and outputs
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=1, unit="V")
    bias_current   = FloatParameter(default=60e-6, unit="A")
    field          = FloatParameter(default=0.0, unit="T")
    voltage        = OutputConnector()

    polarity        = 1
    attempts        = 1 << 4 # How many times to try switching per set of conditions
    measure_current = 10.0e-6 # MTJ sense current
    reset_current   = 0.5e-3 # AWG reset current (remember division)
    MTJ_res         = 100e3 # MTJ Resistance

    # Reference resistances and attenuations
    matching_ref_res      = 50
    chan_bias_ref_res     = 1e4
    reset_bias_ref_res    = 5e3
    pspl_base_attenuation = 10
    circuit_attenuation   = 0

    # Reset Pulse          Switching Pulse
    #                         ___/\____
    # \_______/             /         \
    # A          B         C     D
    #            |------------------------------------->
    #             MEAS Begins (ignore middle) MEAS Ends
    #            | Integ. time|              |integ time|

    # Measurement Sequence timing, clocks in Hz times in seconds
    ai_clock          = 0.25e6
    do_clock          = 0.5e6

    trig_interval     = 200e-6  # The repetition rate for switching attempts
    bias_pulse_width  = 40e-6   # This is how long the bias pulse is
    reset_pulse_width = 40e-6   # This is how long the reset pulse is
    meas_duration     = 140e-6  # This is how long the meas is

    reset_trig_time   = 0e-6    # (A) When we trigger the reset pulse
    meas_trig_time    = 50e-6   # (B) When the measurement trigger begins
    switch_trig_time  = 4e-6    # (C) When we trigger the reset pulse
    pspl_trig_time    = 4e-6    # (D) When we trigger the gate pulse
    integration_time  = 20e-6   # How long to measure for at the beginning and end

    # Instrument resources, NIDAQ called via PyDAQmx
    arb_cb    = Agilent33220A("192.168.5.199") # Channel Bias
    arb_reset = Agilent33220A("192.168.5.198") # Channel reset pulses
    pspl      = Picosecond10070A("GPIB0::24::INSTR") # Gate Pulse
    atten     = RFMDAttenuator("calibration/RFSA2113SB_HPD_20160901.csv") # Gate Amplitude control
    lock      = SR865("USB0::0xB506::0x2000::002638::INSTR") # Gate Amplitude control
    keith     = Keithley2400("GPIB0::25::INSTR")
    mag       = AMI430("192.168.5.109")

    def init_instruments(self):
        # Channel bias arb
        self.arb_cb.output          = False
        self.arb_cb.load_resistance = (self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.function        = 'Pulse'
        self.arb_cb.pulse_period    = 0.99*self.trig_interval
        self.arb_cb.pulse_width     = self.bias_pulse_width
        self.arb_cb.pulse_edge      = 100e-9
        self.arb_cb.low_voltage     = 0.0
        self.arb_cb.high_voltage    = self.bias_current.value*(self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.polarity        = self.polarity
        self.arb_cb.burst_state     = True
        self.arb_cb.burst_cycles    = 1
        self.arb_cb.trigger_source  = "External"
        self.arb_cb.burst_mode      = "Triggered"
        self.arb_cb.output          = True

        # MTJ reset arb
        self.arb_reset.output          = False
        self.arb_reset.load_resistance = (self.reset_bias_ref_res+self.matching_ref_res)
        self.arb_reset.function        = 'Pulse'
        self.arb_reset.pulse_period    = 0.99*self.trig_interval
        self.arb_reset.pulse_width     = self.reset_pulse_width
        self.arb_reset.pulse_edge      = 100e-9
        self.arb_reset.low_voltage     = 0.0
        self.arb_reset.high_voltage    = self.reset_current*(self.reset_bias_ref_res+self.matching_ref_res)
        self.arb_reset.polarity        = -self.polarity
        self.arb_reset.burst_state     = True
        self.arb_reset.burst_cycles    = 1
        self.arb_reset.trigger_source  = "External"
        self.arb_reset.burst_mode      = "Triggered"
        self.arb_reset.output          = True

        # Setup the Keithley
        self.keith.triad()
        self.keith.conf_meas_res(res_range=1e5)
        self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-4)
        self.keith.current = self.measure_current

        # Setup the magnet
        self.mag.ramp()

        # Setup the NIDAQ tasks
        DAQmxResetDevice("Dev1")
        self.nidaq = CallbackTask(["Dev1/ai2"], asyncio.get_event_loop(), int(self.meas_duration*self.ai_clock), self.attempts, [self.voltage],
                                 trigger="/Dev1/PFI12", min_voltage=-self.MTJ_res*self.measure_current, max_voltage=self.MTJ_res*self.measure_current,
                                 ai_clock=self.ai_clock)
        self.nidaq.StartTask()

        self.digital_output = Task()
        do_reset = np.zeros(int(self.do_clock*self.trig_interval),dtype=np.uint8)
        do_reset[int(self.do_clock*self.reset_trig_time)]=1
        do_bias = np.zeros(int(self.do_clock*self.trig_interval),dtype=np.uint8)
        do_bias[int(self.do_clock*self.switch_trig_time)]=1
        do_pspl = np.zeros(int(self.do_clock*self.trig_interval),dtype=np.uint8)
        do_pspl[int(self.do_clock*self.pspl_trig_time)]=1
        do_meas = np.zeros(int(self.do_clock*self.trig_interval),dtype=np.uint8)
        do_meas[int(self.do_clock*self.meas_trig_time)]=1
        data = np.hstack([do_reset, do_bias, do_pspl, do_meas])

        self.digital_output.CreateDOChan("/Dev1/port0/line0:3","",DAQmx_Val_ChanPerLine)
        self.digital_output.CfgSampClkTiming("",self.do_clock, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, int(data.size/4)*self.attempts)
        self.digital_output.WriteDigitalLines(int(self.do_clock*self.trig_interval),0,1,DAQmx_Val_GroupByChannel,data,None,None)

        # Setup the PSPL
        self.pspl.amplitude = 7.5*np.power(10, (-self.pspl_base_attenuation)/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.5
        self.pspl.output = True

        # Setup PSPL Attenuator Control
        self.atten.set_supply_method(self.lock.set_ao2)
        self.atten.set_control_method(self.lock.set_ao3)

        def set_pulse_voltage(voltage, base_atten=self.pspl_base_attenuation):
            # Calculate the voltage controller attenuator setting
            amplitude = 7.5*np.power(10, -base_atten/20.0)
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - base_atten - self.circuit_attenuation

            if vc_atten <= self.atten.minimum_atten():
                base_atten -= 1
                if base_atten < 0:
                    logger.error("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                set_pulse_voltage(voltage, base_atten=base_atten)
                return

            if self.atten.maximum_atten() < vc_atten:
                base_atten += 1
                if base_atten > 80:
                    logger.error("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                set_pulse_voltage(voltage, base_atten=base_atten)
                return

            #print("PSPL Amplitude: {}, Attenuator: {}".format(amplitude,vc_atten))
            self.atten.set_attenuation(vc_atten)
            self.pspl.amplitude = self.arb_cb.polarity*abs(amplitude)
            time.sleep(0.04)

        def set_awg_current(current):

            if 0 <= current:
                self.arb_cb.polarity = 1
                self.arb_cb.low_voltage = 0
                self.arb_cb.high_voltage = current*self.chan_bias_ref_res
            else :
                self.arb_cb.polarity = -1
                self.arb_cb.low_voltage = current*self.chan_bias_ref_res
                self.arb_cb.high_voltage = 0

            self.pspl.amplitude = self.arb_cb.polarity*abs(self.pspl.amplitude)

            time.sleep(0.04)

        # Assign Methods
        self.bias_current.assign_method(set_awg_current)
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.pulse_voltage.assign_method(set_pulse_voltage)
        self.field.assign_method(self.mag.set_field)

        self.pulse_duration.add_post_push_hook(lambda: time.sleep(0.05))
        self.pulse_voltage.add_post_push_hook(lambda: time.sleep(0.05))
        self.bias_current.add_post_push_hook(lambda: time.sleep(0.05))

    def init_streams(self):
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage'
        descrip.add_axis(DataAxis("sample", range(int(self.meas_duration*self.ai_clock))))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))
        self.voltage.set_descriptor(descrip)

    async def run(self):
        self.digital_output.StartTask()
        self.digital_output.WaitUntilTaskDone(2*self.attempts*self.trig_interval)
        self.digital_output.StopTask()
        await asyncio.sleep(0.05)

    def shutdown_instruments(self):
        self.keith.current = 0.0e-5
        self.keith.triad(down=True)
        self.mag.zero()
        try:
            for ch in [self.nidaq, self.nidaq_MTJ]:
                ch.StopTask()
                ch.ClearTask()
                self.digital_output.StopTask()
                del ch
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass
        self.arb_cb.output = False
        self.pspl.output = False
        for name, instr in self._instruments.items():
            instr.disconnect()

if __name__ == '__main__':
    exp = nTronSwitchingExperiment()
