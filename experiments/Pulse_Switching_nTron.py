# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# 0.1 Hz 6dB slope HPF
# 300 kHz 6dB slope LPF

from PyDAQmx import *
from auspex.instruments import Agilent33220A
from auspex.instruments import Picosecond10070A
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
# NIDAQ P1.1 -> PSPL Trigger
# Agilent -> 10 kOhm -> Bias-T -> 50 Ohm -> nTron
#                       |          |
#                     PSPL      NIDAQ AI0

class nTronSwitchingExperiment(Experiment):

    # Parameters and outputs
    channel_bias   = FloatParameter(default=100e-6,  unit="A") # On the 33220A
    pulse_duration = FloatParameter(default=5.0e-9, unit="s")
    pulse_voltage  = FloatParameter(default=0.1, unit="V")
    voltage        = OutputConnector()

    # Constants (set with attribute access if you want to change these!)
    attempts           = 1 << 4
    samples            = 768

    # Reference resistances
    matching_ref_res = 50
    chan_bias_ref_res = 1e4

    # Instrument resources, NIDAQ called via PyDAQmx
    arb_cb = Agilent33220A("192.168.5.199") # Channel Bias
    pspl  = Picosecond10070A("GPIB0::24::INSTR") # Gate Pulse
    atten = RFMDAttenuator("calibration/RFSA2113SB_HPD_20160901.csv") # Gate Amplitude control


    def init_instruments(self):
        # Channel bias arb
        self.arb_cb.output          = False
        self.arb_cb.load_resistance = (self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.function        = 'Pulse'
        self.arb_cb.pulse_period    = 1e-3
        self.arb_cb.pulse_width     = 0.8e-3
        self.arb_cb.pulse_edge      = 100e-9
        self.arb_cb.low_voltage     = 0.0
        self.arb_cb.high_voltage    = self.channel_bias.value*(self.chan_bias_ref_res+self.matching_ref_res)
        self.arb_cb.burst_state     = True
        self.arb_cb.burst_cycles    = 1
        self.arb_cb.trigger_source  = "External"
        self.arb_cb.output          = True

        # Setup the NIDAQ
        self.analog_input = Task()
        self.read = int32()
        self.buf_points = 2*self.samps_per_trig*self.attempts
        self.analog_input.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_Diff,
            self.min_daq_voltage, self.max_daq_voltage, DAQmx_Val_Volts, None)
        self.analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps , self.samps_per_trig)
        self.analog_input.CfgInputBuffer(self.buf_points)
        self.analog_input.CfgDigEdgeStartTrig("/Dev1/PFI1", DAQmx_Val_Rising)
        self.analog_input.SetStartTrigRetriggerable(1)
        self.analog_input.StartTask()

        # Setup DO Triggers, P1:0 triggers AWG, P1:1 triggers PSPL and DAQ analog input
        self.digital_output = Task()
        data = np.zeros((2*int(self.arb_cb.pulse_period*100e6),),dtype=np.uint8)
        data[0]=1
        data[int((self.arb_cb.pulse_period+0.5*self.arb_cb.pulse_width)*100e6)]=1
        for i in range(0,self.attempts-1):
            data = np.append(data,data)

        self.digital_output.CreateDOChan("Dev1/port1/line0:1","",DAQmx_Val_ChanForAlllines)
        self.digital_output.WriteDigitalLines(int(self.arb_cb.pulse_period*100e6),0,1,DAQmx_Val_GroupByChannel,data,None,None)

        # Setup the PSPL
        self.pspl.amplitude = 7.5*np.power(10, (-self.pspl_base_attenuation)/20.0)
        self.pspl.trigger_source = "EXT"
        self.pspl.trigger_level = 0.1
        self.pspl.output = True

        def set_voltage(voltage):
            # Calculate the voltage controller attenuator setting
            self.pspl.amplitude = self.polarity*7.5*np.power(10, -self.pspl_base_attenuation/20.0)
            vc_atten = abs(20.0 * np.log10(abs(voltage)/7.5)) - self.pspl_base_attenuation - self.circuit_attenuation

            if vc_atten <= self.atten.minimum_atten():
                logger.error("Voltage controlled attenuation {} under range.".format(vc_atten))
                raise ValueError("Voltage controlled attenuation {} under range.".format(vc_atten))

            if self.atten.maximum_atten() < vc_atten:
                logger.error("Voltage controlled attenuation {} over range.".format(vc_atten))
                raise ValueError("Voltage controlled attenuation {} over range.".format(vc_atten))

            self.atten.set_attenuation(vc_atten)
            time.sleep(0.02)

        # What to do in order to change the bias values
        self.channel_bias.assign_method(lambda i: self.arb_cb.set_high_voltage(i*self.chan_bias_ref_res))
        self.channel_bias.add_post_push_hook(lambda: time.sleep(0.1))
        self.pulse_duration.assign_method(self.pspl.set_duration)
        self.pulse_voltage.assign_method(set_voltage)



    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage'
        descrip.add_axis(DataAxis("sample", range(self.samps_per_trig)))
        descrip.add_axis(DataAxis("state", range(2)))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))
        self.voltage.set_descriptor(descrip)

    async def run(self):

        self.digital_output.StartTask()

        while(!self.digital_output.IsTaskDone()):
            time.sleep(self.attempts*self.arb_cb.pulse_period)

        self.digital_ouput.StopTask()

        buf = np.empty(self.buf_points)
        self.analog_input.ReadAnalogF64(self.buf_points, -1, DAQmx_Val_GroupByChannel,
                                        buf, self.buf_points, byref(self.read), None)
        await self.voltage.push(buf)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

    def shutdown_instruments(self):
        try:
            self.analog_input.StopTask()
            self.digital_output.StopTask()
        except Exception as e:
            print("Warning: failed to stop task (this normally happens with no consequences when taking multiple samples per trigger).")
            pass


        for name, instr in self._instruments.items():
            instr.disconnect()


if __name__ == '__main__':
    exp = nTronSwitchingExperiment()
    # plot = Plotter(name="Demod!", plot_mode="real", plot_dims=2)
    # plot_ki = Plotter(name="Ki!", plot_mode="real", plot_dims=2)
    # plot_avg = Plotter(name="Avg!", plot_mode="real")
    plot_raw1 = Plotter(name="Raw!", plot_mode="real", plot_dims=1)
    # plot_raw2 = Plotter(name="Raw!", plot_mode="real", plot_dims=2)
    # demod = Channelizer(frequency=exp.measure_frequency, decimation_factor=4, bandwidth=20e6)

    # ki = KernelIntegrator(kernel=0, bias=0, simple_kernel=True, box_car_start=1e-7, box_car_stop=3.8e-7, frequency=0.0)
    # avg = Averager(axis="attempt")

    # samp      = "c1r4"
    # file_path = f"data\\nTron-Switching\\{samp}\\{samp}-PulseSwitchingShort-{datetime.datetime.today().strftime('%Y-%m-%d')}.h5"
    # file_path = f"data\\nTron-Switching\\{samp}\\{samp}-PulseSwitching-{datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')}.h5"
    # wr_int    = WriteToHDF5(file_path, groupname="Integrated", store_tuples=False)
    # wr_final  = WriteToHDF5(file_path, groupname="Final", store_tuples=False)
    # wr_raw    = WriteToHDF5(file_path, groupname="Raw", store_tuples=False)

    edges = [(exp.voltage, plot_raw1.sink),
            #  (exp.voltage, demod.sink),
            # (exp.voltage, wr_raw.sink),
            # (exp.voltage, plot_raw1.sink),
            # (exp.voltage, plot_raw2.sink),
            # (demod.source, ki.sink),
            # (ki.source, plot_ki.sink),
            # (ki.source, avg.sink),
            # (ki.source, wr_int.sink),
            # (avg.final_average, plot_avg.sink),
            # (demod.source, plot.sink),
            ]
    exp.set_graph(edges)

    exp.run_sweeps()
