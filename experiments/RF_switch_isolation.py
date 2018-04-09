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
	width = OutputConnector(unit="seconds")
	width_std = OutputConnector(unit="seconds")
	demod_amp	= OutputConnector(unit="volts")
	demod_amp_std = OutputConnector(unit="volts")
	demod_wf = OutputConnector(unit="volts")
	raw_wf = OutputConnector(unit="volts")

	#Instrument Resources
	sys.path.append("C:/Users/qlab/Documents/dll/") # needed for the Labbrick driver

	awg			= TekAWG5014('192.168.5.100')
	scope		= TekDPO72004C('128.33.89.89')
	brick		= Labbrick('1686')

	#Experiment Configuration
	AWGCHAN = 1 #AWG Output Channel
	DRIVEWF = '50MHz_square' #AWG Waveform

	DEMODCHAN = 4 #Demod signal scope channel
	RAWCHAN = 1 #Raw signal scope channel
	NAVG = 106 #Scope Averages
	NACQ = 10000 #Scope Acquisitions

	TRANSFREQ = 6e9 #Transmission frequency of Labbrick (should be greater than LP Filter)
	TRANSPWR = 0 #Transmission power in dBm (Depends on Amps, Attenuation and Mixer)


	def init_streams_custom(self):

		# Since Scope "sweeps" time on its own, time base must be added explicitly as a data axis for each wf measurement
		# Time base is manually adjusted by experimenter
		TIMEBASE = np.linspace(0, self.scope.record_duration, self.scope.record_length)
		self.demod_wf.add_axis(DataAxis("channel",TIMEBASE))
		self.raw_wf.add_axis(DataAxis("channel",TIMEBASE))

	def init_instruments(self):

		print("Initializing Instrument: {}".format(self.awg.interface.IDN()))

		self.awg.channel = self.AWGCHAN
		self.awg.stop
		self.awg.output = 'OFF'
		self.awg.amplitude = 0.02
		self.awg.offset = 0
		self.awg.runmode = 'CONT'
		self.awg.waveform = self.DRIVEWF

		print("Initializing Instrument: {}".format(self.scope.interface.IDN()))

		self.scope.run = 'OFF'
		self.scope.clear

		self.scope.channel = self.DEMODCHAN
		self.scope.measurement = 1	#Set 1 as Amplitude measurement
		self.scope.measurement_source1 = 'CH{:d}'.format(self.DEMODCHAN)
		self.scope.measurement_type = 'AMP'

		self.scope.measurement = 2	#Set 2 as rise time measurement
		self.scope.measurement_source1 = 'CH{:d}'.format(self.DEMODCHAN)
		self.scope.measurement_type = 'RIS'

		self.scope.measurement = 3	#Set 3 as fall time measurement
		self.scope.measurement_source1 = 'CH{:d}'.format(self.DEMODCHAN)
		self.scope.measurement_type = 'FALL'

		self.scope.measurement = 4	#Set 4 as pulse width meassurement
		self.scope.measurement_source1 = 'CH{:d}'.format(self.DEMODCHAN)
		self.scope.measurement_type = 'PWI'

		self.scope.num_averages = self.NAVG
		self.scope.acquire = 'RUNST' #setup scope for continuous acquisition

		print("Initializing Instrument Labbrick ID: {}".format(self.brick.device_id))

		self.brick.output = 'OFF'
		self.brick.frequency = self.TRANSFREQ
		self.brick.power = self.TRANSPWR
		self.brick.output = 'ON'


		def set_drive_amp(self,amp): 

			self.awg.channel = self.AWGCHAN
			self.awg.stop
			self.awg.output = 'OFF'
			self.awg.amplitude = amp 
			self.awg.offset = amp/2
			self.awg.output = 'ON'


		self.drive_amp.assign_method(set_drive_amp)


	async def run(self):

		self.scope.clear
		self.scope.run = 'ON'
		await asyncio.sleep(0.01)
		self.awg.run

		while self.scope.acquire_num < self.NACQ:
			await asyncio.sleep(0.1)

		self.scope.run = 'OFF'
		self.awg.stop

		self.scope.measurement = 1
		await self.demod_amp.push(self.scope.measurement_mean)
		await self.demod_amp_std.push(self.scope.measurement_std)

		self.scope.measurement = 2
		await self.rise_time.push(self.scope.measurement_mean)
		await self.rise_time_std.push(self.scope.measurement_std)

		self.scope.measurement = 3
		await self.fall_time.push(self.scope.measurement_mean)
		await self.fall_time_std.push(self.scope.measurement_std)

		self.scope.measurement = 4
		await self.demod_amp.push(self.scope.measurement_mean)
		await self.demod_amp_std.push(self.scope.measurement_std)

		self.scope.channel = self.DEMODCHAN
		[time,vals_demod] = self.scope.get_trace
		await self.demod_wf.push(vals_demod)

		self.scope.channel = self.RAWCHAN
		[time,vals_raw] = self.scope.get_trace
		await self.raw_wf.push(vals_raw)




