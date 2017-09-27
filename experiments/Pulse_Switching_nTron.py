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

class CallbackTask(Task):
    def __init__(self, loop, points_per_trigger, attempts, output_connector,
                 chunk_size=None, min_voltage=-10.0, max_voltage=10.0, ai_clock=1e6):
        Task.__init__(self)
        self.loop = loop # the asyncio loop to push data on
        self.points_per_trigger = points_per_trigger
        self.attempts = attempts
        self.output_connector = output_connector

        # Construct our specific task.
        if chunk_size:
            self.buff_points = chunk_size
        else:
            self.buff_points = self.points_per_trigger*self.attempts
        self.buffer = np.empty(self.buff_points)

        self.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_Diff, min_voltage, max_voltage, DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", ai_clock, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, self.points_per_trigger)
        self.CfgInputBuffer(self.buff_points)
        self.CfgDigEdgeStartTrig("/Dev1/PFI1", DAQmx_Val_Rising)
        self.SetStartTrigRetriggerable(1)
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,self.buff_points,0)
        self.AutoRegisterDoneEvent(0)

        # print("Expected to receive ", self.buff_points, "per point.")
        self.points_read = 0
        self.num_callbacks = 0

    def EveryNCallback(self):
        read = int32()
        self.num_callbacks += 1
        self.ReadAnalogF64(self.buff_points, 5.0, DAQmx_Val_GroupByChannel,
                           self.buffer, self.buff_points, byref(read), None)
        asyncio.ensure_future(self.output_connector.push(self.buffer), loop=self.loop)
        self.points_read += read.value
        return 0

    def DoneCallback(self, status):
        return 0

def load_switching_data(filename_or_fileobject, threshold=0.08, group="main", data_name="voltage"):
    data, desc = load_from_HDF5(filename_or_fileobject, reshape=False)
    reps   = desc[group].axis("attempt").points
    dat    = data[group][:].reshape((-1, reps.size))
    probs  = (dat[data_name]>threshold).mean(axis=1)
    durs   = dat['pulse_duration'][:,0]
    amps   = dat['pulse_voltage'][:,0]
    points = np.array([durs, amps]).transpose()
    return points, probs

class nTronSwitchingExperiment(Experiment):

    # Parameters and outputs
    # channel_bias   = FloatParameter(default=100e-6,  unit="A") # On the 33220A
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=1, unit="V")
    voltage        = OutputConnector()

    # Constants (set with attribute access if you want to change these!)
    attempts           = 1 << 4

    # Reference resistances and attenuations
    matching_ref_res = 50
    chan_bias_ref_res = 1e4
    pspl_base_attenuation = 10
    circuit_attenuation = 0
    pulse_polarity = 1

    # Channel Bias current
    channel_bias = 60e-6

    # Min/Max NIDAQ AI Voltage
    min_daq_voltage = 0
    max_daq_voltage = 1e3*channel_bias

    # Measurement Sequence timing, clocks in Hz times in seconds
    ai_clock = 0.25e6
    do_clock = 1e5
    run_time = 2e-4
    settle_time = 0.5*run_time
    integration_time = 0.1*run_time
    ai_delay = 0.25*0.5*run_time

    # Instrument resources, NIDAQ called via PyDAQmx
    arb_cb = Agilent33220A("192.168.5.199") # Channel Bias
    pspl  = Picosecond10070A("GPIB0::24::INSTR") # Gate Pulse
    atten = RFMDAttenuator("calibration/RFSA2113SB_HPD_20160901.csv") # Gate Amplitude control
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR") # Gate Amplitude control

    def init_instruments(self):
        # Channel bias arb
        self.arb_cb.output          = False
        self.arb_cb.load_resistance = (self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.function        = 'Pulse'
        self.arb_cb.pulse_period    = 0.99*self.run_time
        self.arb_cb.pulse_width     = self.run_time - self.settle_time
        self.arb_cb.pulse_edge      = 100e-9
        self.arb_cb.low_voltage     = 0.0
        self.arb_cb.high_voltage    = self.channel_bias*(self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.burst_state     = True
        self.arb_cb.burst_cycles    = 1
        self.arb_cb.trigger_source  = "External"
        self.arb_cb.burst_mode      = "Triggered"
        self.arb_cb.output          = True
        self.arb_cb.polarity        = 1

        # Setup the NIDAQ
        DAQmxResetDevice("Dev1")
        self.nidaq = CallbackTask(asyncio.get_event_loop(), int(self.integration_time*self.ai_clock), self.attempts, self.voltage,
                                 min_voltage=self.min_daq_voltage, max_voltage=self.max_daq_voltage, ai_clock=self.ai_clock)
        self.nidaq.StartTask()

        # Setup DO Triggers, P1:0 triggers AWG, P1:1 triggers PSPL and DAQ analog input
        self.digital_output = Task()
        data_p0 = np.zeros(int(self.do_clock*self.run_time),dtype=np.uint8)
        data_p0[0]=1

        data_p1 = np.zeros(int(self.do_clock*self.run_time),dtype=np.uint8)
        data_p1[int(self.do_clock*(self.ai_delay))]=1

        data = np.append(data_p0,data_p1)

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

        def set_voltage(voltage, base_atten=self.pspl_base_attenuation):
            # Calculate the voltage controller attenuator setting
            amplitude = 7.5*np.power(10, -base_atten/20.0)
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - base_atten - self.circuit_attenuation

            if vc_atten <= self.atten.minimum_atten():
                base_atten -= 1
                if base_atten < 0:
                    logger.error("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                set_voltage(voltage, base_atten=base_atten)
                return

            if self.atten.maximum_atten() < vc_atten:
                base_atten += 1
                if base_atten > 80:
                    logger.error("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                set_voltage(voltage, base_atten=base_atten)
                return

            #print("PSPL Amplitude: {}, Attenuator: {}".format(amplitude,vc_atten))
            self.atten.set_attenuation(vc_atten)
            self.pspl.amplitude = self.pulse_polarity*amplitude
            time.sleep(0.04)

        # Assign Methods
        #self.channel_bias.assign_method(lambda i: self.arb_cb.set_high_voltage(i*self.chan_bias_ref_res))
        self.pulse_duration.add_post_push_hook(lambda: time.sleep(0.1))
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.pulse_voltage.assign_method(set_voltage)


    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage'
        descrip.add_axis(DataAxis("sample", range(int(self.integration_time*self.ai_clock))))
        #descrip.add_axis(DataAxis("state", range(2)))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))
        self.voltage.set_descriptor(descrip)

    async def run(self):

        self.digital_output.StartTask()
        self.digital_output.WaitUntilTaskDone(2*self.attempts*self.run_time)
        self.digital_output.StopTask()
        await asyncio.sleep(0.05)
        # print("\t now ", self.nidaq.points_read)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

    def shutdown_instruments(self):
        try:
            # self.analog_input.StopTask()
            self.nidaq.StopTask()
            self.nidaq.ClearTask()
            self.digital_output.StopTask()
            del self.nidaq
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass
        self.arb_cb.output = False
        self.pspl.output = False
        for name, instr in self._instruments.items():
            instr.disconnect()

class nTronSwitchingExperimentFast(Experiment):

    # Parameters and outputs
    # channel_bias   = FloatParameter(default=100e-6,  unit="A") # On the 33220A
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=1, unit="V")
    channel_bias   = FloatParameter(default=500e-6, unit="A")
    voltage        = OutputConnector()

    # Constants (set with attribute access if you want to change these!)
    attempts = 1 << 6

    # Reference resistances and attenuations
    matching_ref_res      = 50
    chan_bias_ref_res     = 1e4
    pspl_base_attenuation = 10
    circuit_attenuation   = 0
    pulse_polarity        = 1

    # Min/Max NIDAQ AI Voltage
    min_daq_voltage = 0
    max_daq_voltage = 10

    # Measurement Sequence timing, clocks in Hz times in seconds
    ai_clock         = 0.25e6
    do_clock         = 0.5e6
    trig_interval    = 0.2e-3  # The repetition rate for switching attempts
    bias_pulse_width = 40e-6   # This is how long the bias pulse is
    pspl_trig_time   = 4e-6    # When we trigger the gate pulse
    integration_time = 20e-6   # How long to measure for
    ai_delay         = 2e-6    # How long before the measurement begins

    # Instrument resources, NIDAQ called via PyDAQmx
    arb_cb = Agilent33220A("192.168.5.199") # Channel Bias
    pspl   = Picosecond10070A("GPIB0::24::INSTR") # Gate Pulse
    atten  = RFMDAttenuator("calibration/RFSA2113SB_HPD_20160901.csv") # Gate Amplitude control
    lock   = SR865("USB0::0xB506::0x2000::002638::INSTR") # Gate Amplitude control

    def init_instruments(self):
        # Channel bias arb
        self.arb_cb.output          = False
        self.arb_cb.load_resistance = (self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.function        = 'Pulse'
        self.arb_cb.pulse_period    = 0.99*self.trig_interval # Slightly under the trig interval since we are driving with another instrument
        self.arb_cb.pulse_width     = self.bias_pulse_width
        self.arb_cb.pulse_edge      = 100e-9 # Going through the DC port of a bias-tee, no need for fast edges
        self.arb_cb.low_voltage     = 0.0
        self.arb_cb.high_voltage    = self.channel_bias.value*(self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.burst_state     = True
        self.arb_cb.burst_cycles    = 1
        self.arb_cb.trigger_source  = "External"
        self.arb_cb.burst_mode      = "Triggered"
        self.arb_cb.output          = True
        self.arb_cb.polarity        = 1

        # Setup the NIDAQ
        DAQmxResetDevice("Dev1")
        measure_points = int(self.integration_time*self.ai_clock)
        self.nidaq = CallbackTask(asyncio.get_event_loop(), measure_points, self.attempts, self.voltage,
                                #  chunk_size=measure_points*(self.attempts>>2),
                                 min_voltage=self.min_daq_voltage, max_voltage=self.max_daq_voltage, ai_clock=self.ai_clock)
        self.nidaq.StartTask()

        # Setup DO Triggers, P1:0 triggers AWG, P1:1 triggers PSPL and DAQ analog input
        self.digital_output = Task()
        data_p0 = np.zeros(int(self.do_clock*self.trig_interval),dtype=np.uint8)
        data_p0[0]=1

        data_p1 = np.zeros(int(self.do_clock*self.trig_interval),dtype=np.uint8)
        data_p1[int(self.do_clock*(self.pspl_trig_time))]=1

        data = np.append(data_p0,data_p1)

        self.digital_output.CreateDOChan("/Dev1/port0/line0:1","",DAQmx_Val_ChanPerLine)
        self.digital_output.CfgSampClkTiming("",self.do_clock, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, int(data.size/2)*self.attempts)
        self.digital_output.WriteDigitalLines(int(self.do_clock*self.trig_interval),0,1,DAQmx_Val_GroupByChannel,data,None,None)

        # Setup the PSPL`
        self.pspl.amplitude = 7.5*np.power(10, (-self.pspl_base_attenuation)/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.5
        self.pspl.output = True

        # Setup PSPL Attenuator Control
        self.atten.set_supply_method(self.lock.set_ao2)
        self.atten.set_control_method(self.lock.set_ao3)

        # Setup bias current method
        def set_bias(value):
            self.arb_cb.high_voltage = self.channel_bias.value*(self.chan_bias_ref_res+self.matching_ref_res)
            self.arb_cb.low_voltage  = 0.0
        self.channel_bias.assign_method(set_bias)

        def set_voltage(voltage, base_atten=self.pspl_base_attenuation):
            # Calculate the voltage controller attenuator setting
            amplitude = 7.5*np.power(10, -base_atten/20.0)
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - base_atten - self.circuit_attenuation

            if vc_atten <= self.atten.minimum_atten():
                base_atten -= 1
                if base_atten < 0:
                    logger.error("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} under range, PSPL at Max. Decrease circuit attenuation.".format(vc_atten))
                set_voltage(voltage, base_atten=base_atten)
                return

            if self.atten.maximum_atten() < vc_atten:
                base_atten += 1
                if base_atten > 80:
                    logger.error("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                    raise ValueError("Voltage controlled attenuation {} over range, PSPL at Min. Increase circuit attenuation.".format(vc_atten))
                set_voltage(voltage, base_atten=base_atten)
                return

            #print("PSPL Amplitude: {}, Attenuator: {}".format(amplitude,vc_atten))
            self.atten.set_attenuation(vc_atten)
            self.pspl.amplitude = self.pulse_polarity*amplitude
            time.sleep(0.04)

        # Assign Methods
        #self.channel_bias.assign_method(lambda i: self.arb_cb.set_high_voltage(i*self.chan_bias_ref_res))
        self.pulse_duration.add_post_push_hook(lambda: time.sleep(0.1))
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.pulse_voltage.assign_method(set_voltage)


    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage'
        descrip.add_axis(DataAxis("sample", range(int(self.integration_time*self.ai_clock))))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))
        self.voltage.set_descriptor(descrip)

    async def run(self):

        self.digital_output.StartTask()
        self.digital_output.WaitUntilTaskDone(2*self.attempts*self.trig_interval)
        self.digital_output.StopTask()
        await asyncio.sleep(0.05)
        # print("\t now ", self.nidaq.points_read)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

    def shutdown_instruments(self):
        try:
            # self.analog_input.StopTask()
            self.nidaq.StopTask()
            self.nidaq.ClearTask()
            self.digital_output.StopTask()
            del self.nidaq
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass
        self.arb_cb.output = False
        self.pspl.output = False
        for name, instr in self._instruments.items():
            instr.disconnect()


if __name__ == '__main__':
    exp = nTronSwitchingExperiment()
