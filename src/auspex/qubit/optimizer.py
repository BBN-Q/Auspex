# Copyright 2020 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

try:
    from QGL import *
    from QGL import config as QGLconfig
    from QGL.BasicSequences.helpers import create_cal_seqs, delay_descriptor, cal_descriptor
except:
    print("Could not find QGL")

import auspex.config as config
from auspex.log import logger
from copy import copy, deepcopy
# from adapt.refine import refine_1D
import os
import uuid
import pandas as pd
import networkx as nx
import scipy as sp
import subprocess
import zmq
import json
import datetime
from copy import copy

import time
import bbndb
from auspex.filters import DataBuffer
from .qubit_exp import QubitExperiment
from .pulse_calibration import Calibration, CalibrationExperiment
from . import pipeline
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.analysis.fits import *
from auspex.analysis.CR_fits import *
from auspex.analysis.qubit_fits import *
from auspex.analysis.helpers import normalize_buffer_data
from matplotlib import cm
from scipy.optimize import curve_fit, minimize
import numpy as np
from itertools import product
from collections import Iterable, OrderedDict

available_optimizers = ['SCIPY']

try:
	from bayes_opt import BayesianOptimization
	available_optimizers += ['BAYES']
except ImportError:
	logger.info("Could not import BayesianOptimization package.")


class QubitOptimizer(Calibration):
	"""
	Class for running an optimization over Auspex experiments.

	"""

	def __init__(self, qubits, sequence_function, cost_function, 
				 initial_parameters, other_variables=None, 
				 optimizer="scipy", optim_params=None, output_nodes=None, 
				 stream_selectors=None, do_plotting=True, **kwargs):
		"""Setup an optimization over qubit experiments.

		Args:
			qubits:	The qubit(s) that the optimization is run over.
			sequence_function: A function of the form 

				`sequence_function(*qubits, **params)` 

				that returns a valid QGL sequence for the qubits and initial 
				parameters.
			cost_function: The objective function for the optimization. The input
				for this function comes from the filter pipeline node specified 
				in `output_nodes` or inferred from the qubits (may not be 
				reliable!). This function is responsible for choosing the 
				appropriate quadrature as necessary.
			initial_parameters: A dict of initial parameters for `sequence_function`.
			other_variables: A dict of other Auspex qubit experiment variables 
				(not associated with sequence generation) as keys and initial 
				parameters as values. Example:

				`{"q1 control frequency": 5.4e9, "q2 measure frequency": 6.7e9}`
			optimizer: String which chooses the optimization function. Supported
				values are: "scipy" for scipy.optimize.minimize, "bayes" for 
				the BayesianOptimization package. Others TODO!
			optim_params: Dict of keyword arguments to be passed to the 
				optimization function.

		"""

		self.qubits = list(qubits) if isinstance(qubits, Iterable) else [qubits]
		self.sequence_function  = sequence_function
		self.cost_function      = cost_function
		self.initial_parameters = OrderedDict(initial_parameters)
		self.other_variables    = OrderedDict(other_variables)
		self.optimizer 			= optimizer.upper() 
		self.optim_params		= optim_params

		self.seq_params 		= self.initial_parameters
		self.other_params		= self.other_variables

		self.output_nodes 		= output_nodes
		self.stream_selectors   = stream_selectors
		self.do_plotting        = do_plotting 

		self.cw_mode 			= False
		self.leave_plots_open	= True
		self.axis_descriptor	= None
		self.succeeded 			= False
		self.norm_points		= False
		self.kwargs				= kwargs 
		self.plotters 			= []
		self.fake_data			= []
		self.sample				= None
		self.metafile			= None

		super().__init__()

		if self.optimizer not in available_optimizers:
			raise ValueError(f"Unknown optimizer: {self.optimizer}. Availabe are: {available_optimizers}")
	
	def init_plots(self):
		plot1 = ManualPlotter("Objective", x_label="Iteration", y_label="Value")
		plot1.add_data_trace("Objective", {'color': 'C1'})
		self.plot1 = plot1
		return [plot1]

	def _optimize_function(self):			
		def _func(x):

			self._update_params(x)
			data = self.run_sweeps()
			return self.cost_function(data)
		return _func

	def _update_params(self, p):
		if self.seq_params and self.other_params:
			for idx, k in enumerate(self.seq_params.keys()):
				self.seq_params[k] = p[idx]
			for k in self.other_params.keys():
				idx += 1
				self.other_params[k] = p[idx]
		elif self.seq_params:
			for idx, k in enumerate(self.seq_params.keys()):
				self.seq_params[k] = p[idx]
		elif self.other_params:
			for idx, k in enumerate(self.other_params.keys()):
				self.other_params[k] = p[idx]


	def set_constraints(self, constraints):
		"""Add constraints to the optimization. 

		Args:
			constraints: A dictionary of constraints. The key should match up 
			with the named parameters in `initial_parameters` or `other_variables`. 
			The values should be a list that represents lower and upper bounds. 
			Nonlinear constraints are not (yet!) supported.
		"""
		raise NotImplementedError("Constraints not yet impemented!")

	def parameters(self):
		"""Returns the current set of parameters that are being optimized over"""
		if self.seq_params and self.other_params:
			return OrderedDict({**self.seq_params, **self.other_params})
		elif self.seq_params:
			return self.seq_params
		elif self.other_params:
			return self.other_params

	def calibrate():
		logger.info(f"Not a calibration! Please use {self.__class__.__name___}.optimize")

	def run_sweeps(self):
		
		seq = self.sequence_function(*self.qubits, **self.seq_params)
		metafile = compile_to_hardware(seq, "optim/optim")

		exp       = CalibrationExperiment(self.qubits, self.output_nodes, 
											self.stream_selectors, meta_file, 
											**self.kwargs)

		if len(self.fake_data) > 0:
			raise NotImplementedError("Fake data not yet implemented!")
		self.exp_config(exp)

		exp.run_sweeps()

		data = {}

        #sort nodes by qubit name to match data with metadata when normalizing
        qubit_indices = {q.label: idx for idx, q in enumerate(exp.qubits)}
        exp.output_nodes.sort(key=lambda x: qubit_indices[x.qubit_name])

        for i, (qubit, output_buff, var_buff) in enumerate(zip(exp.qubits,
                                [exp.proxy_to_filter[on] for on in exp.output_nodes])):
            if not isinstance(output_buff, DataBuffer) or isinstance(output_buff, WriteToFile):
                raise ValueError("Could not find data buffer for calibration.")

            dataset, descriptor = output_buff.get_data()
            data[qubit.label] = dataset

        # Return data and variance of the mean
        if len(data) == 1:
            # if single qubit, get rid of dictionary
            data = list(data.values())[0]
            var = list(var.values())[0]
        return data

	def set_fake_data():
		raise NotImplementedError()


	def optimize(self):
		""" Carry out the optimization. """

		if self.optimizer == "SCIPY":

			def plot_callback(xn, result):
				self.plot1['Value'] = (result.nit, result.fun)
			
			x0  = list(self.parameters().values())
			result = minimize(self._optimize_function(), x0, callback=plot_callbabck,
								**self.optim_params)

		if self.optimizer == "BAYES":
			#TODO!
			raise NotImplementedError()




