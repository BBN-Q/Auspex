# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import TekDPO72004C, TekAWG5014, Labbrick

from auspex.experiment import FloatParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector

import asyncio
import numpy as np
import time
import datetime
import sys

# Experimental Topology
# Labbrick -> Splitter -> atten -> RF_S1 -> RF_Switch -> RF_S2 -> Filters, Amps -> Splitter -> Amp -> Mixer_RF
# 						|																	|
#						Amp -> Mixer_LO														|
# Mixer_Output -> 5GHz Lowpass -> Tektronix DPO72004C 										|
#																					Tektronix DPO72004C
# Tektronix AWG5014 -> Switch Drive 1
# Switch Drive 2 -> 50 Ohm terminator


class RF_Switch_Isolation(Experiment):

	#Define Experiment Axis (Sweep Parameters)
	drive_amp = FloatParameter(default=0.02,unit="volts")

	# Setup Output Connectors (Measurements)
	rise_time	= OutputConnector(unit="seconds")
	rise_time_std = OutputConnector(unit="seconds")
	fall_time	= OutputConnector(unit="seconds")
	fall_time_std = OutputConnector(unit="seconds")
	demod_amp	= OutputConnector(unit="volts")
	demod_amp_std = OutputConnector(unit="volts")
	demod_wf = OutputConnector(unit="volts")
	raw_wf = OutputConnector(unit="volts")

	#Instrument Resources
	sys.path.append("C:/Users/qlab/Documents/dll/")

	awg			= TekAWG5014('192.168.5.100')
	scope		= TekDPO72004C('128.33.89.89')
	brick		= Labbrick('1686')

	def init_streams(self):

		# Since Mux sweeps over channels itself, channel number must be added explicitly as a data axis to each measurement
		self.sheet_res.add_axis(DataAxis("channel",CHAN_LIST))
		self.temp_A.add_axis(DataAxis("channel",CHAN_LIST))
		self.temp_B.add_axis(DataAxis("channel",CHAN_LIST))
		self.sys_time.add_axis(DataAxis("channel",CHAN_LIST))

	def init_instruments(self):

		print("Initializing Instrument: {}".format(self.mux.interface.IDN()))

		self.mux.scanlist = CHAN_LIST
		self.mux.configlist = self.mux.scanlist
		self.mux.resistance_wire = 4
		self.mux.resistance_range = RES_RANGE
		self.mux.resistance_resolution = PLC
		self.mux.resistance_zcomp = ZCOMP

		print("Initializing Instrument: {}".format(self.lakeshore.interface.IDN()))

		self.lakeshore.config_sense_A	= A_CONFIG
		self.lakeshore.config_sense_B	= B_CONFIG
		self.lakeshore.range_htr_1		= 0
		self.lakeshore.range_htr_2		= 0
		self.lakeshore.mout_htr_1		= 0
		self.lakeshore.mout_htr_2		= 0

		self.index.assign_method(int)


	async def run(self):

		self.mux.scan()

		# Everything needs len(chan_list) copies since sheet_res is read in len(chan_list) at a time. This preserves the dimensionality of the data
		await self.temp_A.push([self.lakeshore.Temp("A")]*len(CHAN_LIST))
		await self.temp_B.push([self.lakeshore.Temp("B")]*len(CHAN_LIST))
		await self.sys_time.push([time.time()]*len(CHAN_LIST))

		await asyncio.sleep(4*len(CHAN_LIST)*PLC/60)
		await self.sheet_res.push(self.mux.read())