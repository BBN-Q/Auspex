# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import BNC845, DigitalAttenuator, AgilentE9010A

from auspex.experiment import FloatParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector

import asyncio
import numpy as np
import time
import datetime
import sys

# Experiment sweeps frequency and source power via a programmable attenuator to measure compression of device response with input power and frequency
# Experimental Topology
# BNC Source -> Fixed Attenuator -> Digital Attenuator -> S1 RF switch -> S2 RF Switch -> Spectrum Analyzer

class RF_Switch_Compression(Experiment):

	#Define Experiment Axis (Sweep Parameters)
	inputatten = FloatParameter(default=0,unit="dB")
	inputfreq = FloatParameter(defualt=2e9,unit="Hz")

	# Setup Output Connectors (Measurements)
	outputpow = OutputConnector(unit="dBm")

	#Instrument Resources
	atten = DigitalAttenuator('COM3') #Serial port must be checked with each CPU reboot
	source = BNC845('192.168.5.161') #Use BNC GUI to determine IP
	spec = AgilentE9010A('192.168.5.103')


	def init_instruments(self):

		print("Initializing Instruments")
		self.atten.ch1_attenuation = 30
		self.source.output = False

		self.inputatten.assign_method(self.atten.ch1_attenuation)
		self.inputfreq.assign_method(self.source.frequency)

	async def run(self):

		self.spec.frequency_span = 0
		self.spec.frequency_center = self.inputfreq
		await self.outputpow.push(self.spec.power)

