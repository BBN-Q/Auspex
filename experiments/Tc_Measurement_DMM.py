# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments.agilent import Agilent34970A
from auspex.instruments.lakeshore import LakeShore335

from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.io import WriteToHDF5
from auspex.filters.plot import XYPlotter, Plotter
from auspex.filters.average import Averager
from auspex.analysis.io import load_from_HDF5

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import h5py

from adapt.refine import refine_1D


# from auspex.log import logger
# import logging
# logger.setLevel(logging.DEBUG)

# Experimental Topology
# Sweep dummy index (time proxy) 
# Lakeshore Output -> T senses A and B
# MUX Output -> Sheet resistance of 4 channels
# System time Output -> time of measurement

#Global and instrument configuration variables

# Define Base Temp, Mas Temp, Temp resolution, Resistance noise and max points for Tc refinement
BASETEMP  = 5	  #Kelvin
MAXTEMP	  = 20 	  #Kelvin	
TRES      = 0.05  #Kelvin
RNOISE    = 0.009 #Ohms 
MAXPOINTS = 50

# Configure channels 101:104 for 4 wire resistance measurements
# 100 Ohm range, 10 PLC integration time, 0 Compenstaion ON
CHAN_LIST	= []
SAMPLE_MAP  = {}
RES_RANGE	= "AUTO"
PLC			= 10
ZCOMP		= "ON"

# Configure Tsense A: Diode,2.5V,Kelvin units
# Configure Tsense B: NTC RTD,30 Ohm,Compensation ON,Kelvin units
# This configuration will change with measurement setup
A_CONFIG	= [1,0,0,0,1]
B_CONFIG	= [3,1,0,1,1]

# Configure Heater 1 as 
# Configure Heater 1 output for Closed Loop PID, Sensor B control, Powerup enable off
# Configure Heater 1 range as Medium
# P = 156.7, I = 17.9, D = 100
# This configuration will change with measurement setup

HTR_CONFIG	= [0,2,2,0,2]
HTR_CTRL	= [1,2,0]
HTR_RNG		= 2
HTR_PID 	= [156.7,17.9,100]

class Cooldown(Experiment):

	# Define Experiment Axis
	index		= IntParameter(default=0,unit="none")

	# Setup Output Connectors (Measurements)
	sheet_res	= OutputConnector(unit="Ohm/sq")
	temp_A		= OutputConnector(unit="Kelvin")
	temp_B		= OutputConnector(unit="Kelvin")
	sys_time	= OutputConnector(unit="seconds")

	#Instrument Resources
	mux			= Agilent34970A("GPIB0::10::INSTR")
	lakeshore	= LakeShore335("GPIB0::2::INSTR")

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

# Experimental Topology
# Sweep Temperature with refinement function
# MUX Output -> Sheet resistance of 4 channels

class TcMeas(Experiment):

	# Define Experiment Axis
	temp_set	= FloatParameter(default=0,unit="Kelvin")

	# Setup Output Connectors (Measurements)
	sheet_res	= OutputConnector(unit="Ohm/sq")
	temp_meas	= OutputConnector(unit="Kelvin")

	# Thermal wait time and timeout in seconds, resolution in Kelvin
	T_delta		= 60
	T_res		= 0.05

	#Instrument Resources
	mux			= Agilent34970A("GPIB0::10::INSTR")
	lakeshore	= LakeShore335("GPIB0::2::INSTR")

	def init_streams(self):

		# Since Mux sweeps over channels itself, channel number must be added explicitly as a data axis to each measurement
		self.temp_meas.add_axis(DataAxis("channel",CHAN_LIST))
		self.sheet_res.add_axis(DataAxis("channel",CHAN_LIST))

	def init_instruments(self):

		print("Initializing Instrument: {}".format(self.mux.interface.IDN()))

		self.mux.scanlist = CHAN_LIST
		self.mux.configlist = self.mux.scanlist
		self.mux.resistance_wire = 4
		self.mux.resistance_range = RES_RANGE
		self.mux.resistance_resolution = PLC
		self.mux.resistance_zcomp = ZCOMP

		print("Initializing Instrument: {}".format(self.lakeshore.interface.IDN()))

		self.lakeshore.config_sense_B	= B_CONFIG
		self.lakeshore.config_htr_1		= HTR_CONFIG
		self.lakeshore.control_htr_1	= HTR_CTRL
		self.lakeshore.mout_htr_1		= 0
		self.lakeshore.mout_htr_2		= 0
		self.lakeshore.pid_htr_1		= HTR_PID
		self.lakeshore.range_htr_1		= HTR_RNG


		self.temp_set.assign_method(self.set_temp)

	def set_temp(self,temp):

		# Set temperature and wait for stabilization 

		print("Setting Temperature to: {} K".format(temp))

		self.lakeshore.temp_htr_1 = temp

		T_wait = (abs(temp - self.lakeshore.Temp("B")))*self.T_delta
		waiting = T_wait
		time.sleep(waiting)

		while self.T_res < abs(temp - self.lakeshore.Temp("B")) and waiting < 2*T_wait:
			waiting += self.T_delta
			time.sleep(self.T_delta)

		time.sleep(self.T_delta)
		print("Temperature Stablized at {} K".format(self.lakeshore.Temp("B")))

	async def run(self):

		self.mux.scan()

		print("Measuring ...")

		# Everything needs len(chan_list) copies since sheet_res is read in len(chan_list) at a time. This preserves the dimensionality of the data
		await self.temp_meas.push([self.lakeshore.Temp("B")]*len(CHAN_LIST))

		# Wait for the expected integration time and read. OPC hangs while measuring we have to do it this way.
		# For some reason integration seems to take 2x what I expect, could use investigation
		await asyncio.sleep(4*len(CHAN_LIST)*PLC/60)

		await self.sheet_res.push(self.mux.read())

	def shutdown_instruments(self):

		print("Turning heaters off")
		self.lakeshore.range_htr_1 = 0
		self.lakeshore.range_htr_2 = 0

def load_tc_meas(filename):

	# load data and descriptors with convenience function 
	data, desc = load_from_HDF5(filename)

	# reshape data along sweep axes
	t_pts = data['main']['temp_set'].reshape(desc.data_dims())
	r_pts = data['main']['sheet_res'].reshape(desc.data_dims())

	# take the mean of the data over channel number
	t_pts = t_pts.mean(axis=1)
	r_pts = r_pts.mean(axis=1)

	return t_pts, r_pts


def tc_analysis(filename):

 	print("Analyzing transition data...")

 	data, desc = load_from_HDF5(filename)
 	torder_data = np.sort(data['main'],order='temp_meas')

 	key = ""

 	for ch in CHAN_LIST: 
 		ch_data = torder_data[torder_data['channel']==ch]

 		# Calc transition temperature as max positive derivative
 		dT = np.diff(ch_data['temp_meas'][ch_data['temp_set']>BASETEMP])
 		dR = np.diff(ch_data['sheet_res'][ch_data['temp_set']>BASETEMP])

 		tran_index = np.argmax(np.divide(dR,dT))
 		temps 	   = ch_data['temp_meas'][ch_data['temp_set']>BASETEMP]
 		Tc         = (temps[tran_index]+temps[tran_index+1])/2

 		if RNOISE<dR[tran_index]: 
 			print("Transition in {} measured at {:.2f} K".format(SAMPLE_MAP[ch],Tc))
 			key = "{}, Tc = {:.2f}".format(SAMPLE_MAP[ch],Tc)
 		else:
 			print("No transition detected in {}".format(SAMPLE_MAP[ch])) 
 			key = "{}, No transition".format(SAMPLE_MAP[ch])

 		plt.xlabel("Temperature (K)")
 		plt.ylabel("Sheet Resistance (Ohms/sq)")
 		plt.title(key)
 		plt.plot(ch_data['temp_meas'],ch_data['sheet_res'],'bo')
 		plt.savefig("{}.png".format(SAMPLE_MAP[ch]), bbox_inches='tight')

 	print("Analysis complete")

def main():

	# Define Measurement Channels and sample names
	CHAN_LIST 	= [101,102,103,104]
	RES_RANGE	= 1000
	CDPLC		= 10
	TcPLC		= 100
	SAMPLE_MAP	= {101:'TOX23_NbN',102:'TOX24_NbN',103:'TOX25_NbN',104:'TOX-23_Nb'} 

	# Define Base Temp, Mas Temp, Temp resolution, Resistance noise and max points for Tc refinement
	BASETEMP  = 4	  #Kelvin
	MAXTEMP	  = 25 	  #Kelvin	
	TRES      = 0.05  #Kelvin
	RNOISE    = 0.009 #Ohms 
	MAXPOINTS = 50

	#--------------------------User shouldn't need to edit below here--------------------------------

	names = []
	for i in CHAN_LIST:
		names.append(SAMPLE_MAP[i])

	# Define data file name and path
	sample_name		= ("SAMPLES_"+'_'.join(['{}']*len(names))).format(*names)
	date        	= datetime.datetime.today().strftime('%Y-%m-%d')
	path 			= "Tc_Data\{date:}".format(date=date)

	# Check if already at Base temp
	ls	= LakeShore335("GPIB0::2::INSTR")
	ls.connect()
	t_check = ls.Temp('B')
	ls.disconnect()

	if BASETEMP < t_check:

		# Reset Global config variables
		PLC = CDPLC

		# Create Experiment Object
		cd_exp  = Cooldown()

		# Setup datafile and define which data to write, plot ect.
		cd_file	= "{path:}\{samp:}-Cooldown_{date:}.h5".format(path=path, samp=sample_name, date=date)
		wr = WriteToHDF5(cd_file)

		# Create plots for monitoring. 
		#plt_Avt  = XYPlotter(name="Temperature Sense A", x_series=True, series="inner")
		#plt_Bvt  = XYPlotter(name="Temperature Sense B", x_series=True, series="inner")
		#plt_RvT  = XYPlotter(name="Sample Resistance", x_series=True, series="inner")

		edges = [(cd_exp.sheet_res, wr.sink), (cd_exp.temp_A, wr.sink), (cd_exp.temp_B, wr.sink), (cd_exp.sys_time, wr.sink)]
		cd_exp.set_graph(edges)


		# Add points 10 at a time until base temp is reached
		async def while_temp(sweep_axis, experiment):

			if experiment.lakeshore.Temp("B") < BASETEMP: 
				print("Base Temperature Reached...")
				return False

			print("Running refinement loop: Temp %f, Num_points: %d, last i %d" % (experiment.lakeshore.Temp("B"), sweep_axis.num_points(), sweep_axis.points[-1]))

			last_i = sweep_axis.points[-1]
			sweep_axis.add_points(range(last_i+1,last_i+10))

			return True

		# Defines index as sweep axis where while_temp function determines end condition
		sweep_axis = cd_exp.add_sweep(cd_exp.index, range(3), refine_func=while_temp)

		# Run the experiment
		print("Running Cooldown Log starting from {} K".format(t_check))
		print("Writing Cooldown Data to file: {}".format(wr.filename.value))
		cd_exp.run_sweeps()
		print("Cooldown Logging Completed...")

	else: 
		print("Experiment at base temperature {:.3f} K, skipping Cooldown Log...".format(t_check))

	# Reset Global config variables
	PLC = TcPLC

	# Create Experiment Object
	tc_exp  = TcMeas()

	# Setup datafile and define which data to write, plot ect.
	tc_file	= "{path:}\{samp:}-Tc_{date:}.h5".format(path=path, samp=sample_name, date=date)
	wr = WriteToHDF5(tc_file)
	edges = [(tc_exp.sheet_res, wr.sink),(tc_exp.temp_meas, wr.sink)]
	tc_exp.set_graph(edges)

	# Add points until max points are reached or resolution limit of refine function is hit
	async def transition(sweep_axis, exp):

		points, mean = load_tc_meas(wr.filename.value)

		newpoints = refine_1D(points, mean, all_points=False, criterion="difference", threshold = "one_sigma", resolution = TRES, noise_level = RNOISE)

		if newpoints is None or len(points)+len(newpoints) > MAXPOINTS:

			print("Tc Refinement complete... Exiting")
			return False

		# Always remeasure BASETEMP to ensure lack of thermal hysterysis
		print("Refining Tc Measurement...")
		sweep_axis.add_points(np.insert(newpoints,0,BASETEMP))
		return True

	# Defines index as sweep axis where transition function determines end condition
	sweep_axis = tc_exp.add_sweep(tc_exp.temp_set, range(BASETEMP,MAXTEMP,2), refine_func=transition)

	# Run the experiment
	print("Running Tc Experiment...")
	print("Writing Tc Data to file: {}".format(wr.filename.value))
	tc_exp.run_sweeps()
	print("Tc Experiment Complete")

	#Run post processing analysis
	tc_anlaysis(wr.filename.value)



if __name__ == '__main__':

	main()

