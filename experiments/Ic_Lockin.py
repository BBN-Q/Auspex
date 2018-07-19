# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import SR865
from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import OutputConnector
from auspex.filters import WriteToHDF5, Plotter
from auspex.log import logger

import numpy as np

import time
import datetime

class IcLockinExperiment(Experiment):
	""" Nano-wire Ic measurement using Lockin with DC offset
	"""
	source       = FloatParameter(default=0.0, unit="V")
	voltage      = OutputConnector(unit="V")
	current      = OutputConnector(unit="A")
	resistance   = OutputConnector(unit="Ohm")

	R_ref = 1e3
	sense = 5e-6

	lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

	def init_instruments(self):
		# self.keith.triad()
		# self.keith.conf_meas_res(NPLC=10, res_range=1e5)
		# self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
		# self.keith.current = self.measure_current.value

		# Initialize lockin
		self.lock.amp = self.sense*self.R_ref
		#self.lock.tc  = self.integration_time
		self.delay = self.lock.measure_delay()

		# Define source method
		self.source.assign_method(self.set_source)
		#self.source.add_post_push_hook(lambda: time.sleep(2*self.integration_time))

	def set_source(self,source):

		self.lock.dc = source
		time.sleep(self.lock.measure_delay())


	def run(self):
		"""This is run for each step in a sweep."""

		time.sleep(self.delay)
		R_load = self.lock.mag/(self.sense - self.lock.mag)*self.R_ref
		self.resistance.push(R_load)
		self.current.push(self.lock.dc/(self.R_ref+R_load))
		self.voltage.push(self.lock.dc*R_load/(self.R_ref+R_load))

		logger.debug("Stream has filled {} of {} points".format(self.resistance.points_taken,
																self.resistance.num_points() ))

		#time.sleep(2*self.integration_time) # Give the filters some time to catch up?

	def shutdown_instruments(self):
		self.lock.dc = 0
		self.lock.amp = 0

if __name__ == '__main__':
	sample_name = "Hypress_3_2_500nm_wire"
	date = datetime.datetime.today().strftime('%Y-%m-%d')
	file_path = "data\CRAM01\{samp:}\{samp:}-Ic_{date:}.h5".format(samp=sample_name, date=date)

	exp = IcLockinExperiment()
	wr  = WriteToHDF5(file_path)
	plt = Plotter()

	edges = [(exp.resistance, wr.sink),(exp.voltage, wr.sink),(exp.current, wr.sink),(exp.resistance, plt.sink)]
	exp.set_graph(edges)

	source_pts = np.linspace(0,400e-3,200)
	#fields = np.append(fields, np.flipud(fields))
	main_sweep = exp.add_sweep(exp.source,source_pts)
	exp.run_sweeps()
