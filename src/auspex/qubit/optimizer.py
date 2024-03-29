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

try:
    import cma
    available_optimizers += ['CMA']
except ImportError:
    logger.info("Could not import pyCMA optimization package.")

class CloseEnough(Exception):
    pass


class QubitOptimizer(Calibration):
    """
    Class for running an optimization over Auspex experiments.

    """

    def __init__(self, qubits, sequence_function, cost_function, 
                 initial_parameters=None, other_variables=None,
                 optimizer="scipy", optim_params=None, min_cost = None, 
                 output_nodes=None, stream_selectors=None, do_plotting=True, **kwargs):
        """Setup an optimization over qubit experiments.

        Args:
            qubits: The qubit(s) that the optimization is run over.
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
                the BayesianOptimization package
            optim_params: Dict of keyword arguments to be passed to the 
                optimization function.
            min_cost: Minimum value of cost function, optional. 

        """

        self.qubits = list(qubits) if isinstance(qubits, Iterable) else [qubits]
        self.sequence_function  = sequence_function
        self.cost_function      = cost_function
        self.optimizer          = optimizer.upper() 
        self.optim_params       = optim_params

        self.output_nodes       = output_nodes if isinstance(output_nodes, Iterable) else [output_nodes]
        self.stream_selectors   = stream_selectors
        self.do_plotting        = do_plotting 

        self.cw_mode            = False
        self.leave_plots_open   = True
        self.axis_descriptor    = None
        self.succeeded          = False
        self.norm_points        = False
        self.kwargs             = kwargs 
        self.plotters           = []
        self.fake_data          = []
        self.sample             = None
        self.metafile           = None
        self.costs              = []

        self.fake               = False

        self.niterations        = 0
        self.min_cost           = min_cost

        if initial_parameters:
            self.initial_parameters = OrderedDict(initial_parameters)
            self.recompile = True
        else:
            self.initial_parameters = {}
            self.recompile = False
        if other_variables:
            self.other_variables    = OrderedDict(other_variables)
        else:
            self.other_variables = None

        self.seq_params         = self.initial_parameters
        self.other_params       = self.other_variables

        self.param_history = OrderedDict({k: [] for k in self.parameters().keys()})

        self.bounds   = OrderedDict({})

        super().__init__()

        if self.optimizer not in available_optimizers:
            raise ValueError(f"Unknown optimizer: {self.optimizer}. Availabe are: {available_optimizers}")
    
    def init_plots(self):
        plot1 = ManualPlotter("Objective", x_label="Iteration", y_label="Value")
        plot1.add_data_trace("Objective", {'color': 'C1'})
        self.plot1 = plot1

        plot2 = ManualPlotter("Paramters", x_label="Iteration", y_label="Value")
        for idx, key in enumerate(self.parameters().keys()):
            plot2.add_data_trace(key, {'color': f'C{idx}'})
        self.plot2 = plot2

        return [plot1, plot2]

    def update_plots(self):
        iters = np.array(range(1,self.niterations+1))
        self.plot1['Objective'] = (iters, np.array(self.costs))

        for k, v in self.param_history.items():
            self.plot2[k] = (iters, np.array(v))

    #TODO: Multi-qubit fake data generation...
    def setup_fake_data(self, digitizer, fake_data_function):
        self.fake_dig = digitizer
        self.fake_data_fn = fake_data_function
        self.fake = True

    def _optimize_function_scipy(self):           
        def _func(x):
            self._update_params(plist=x)
            data = self.run_sweeps()
            cost = self.cost_function(data)
            self.costs.append(cost)
            if self.do_plotting:
                self.update_plots()
            if self.min_cost:
                if cost < self.min_cost:
                    raise CloseEnough()

            return cost
        return _func

    def _optimize_function_bayes(self):           
        def _func(**x):
            self._update_params(pdict=x)
            data = self.run_sweeps()
            cost = self.cost_function(data)
            self.costs.append(cost)
            if self.do_plotting:
                self.update_plots()
            if self.min_cost:
                if np.abs(cost) < self.min_cost:
                    raise CloseEnough()

            return cost
        return _func

    def _update_params(self, plist=None, pdict=None):
        if pdict:
            for k, v in pdict.items():
                if self.seq_params and k in self.seq_params:
                    self.seq_params[k] = v 
                if self.other_params and k in self.other_params:
                    self.other_params[k] = v
        else:
            if self.seq_params and self.other_params:
                for idx, k in enumerate(self.seq_params.keys()):
                    self.seq_params[k] = plist[idx]
                for k in self.other_params.keys():
                    idx += 1
                    self.other_params[k] = plist[idx]
            elif self.seq_params:
                for idx, k in enumerate(self.seq_params.keys()):
                    self.seq_params[k] = plist[idx]
            elif self.other_params:
                for idx, k in enumerate(self.other_params.keys()):
                    self.other_params[k] = plist[idx]

        for k,v in self.parameters().items():
            self.param_history[k].append(v)


    def set_bounds(self, bounds):
        """Add bounds to the optimization. 

        Args:
            bounds: A dictionary of bounds. The key should match up 
            with the named parameters in `initial_parameters` or `other_variables`. 
            The values should be a list that represents lower and upper bounds 
            in the form (min, max). 
        """
        for k in self.parameters().keys():
            self.bounds[k] = bounds[k]

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
        self.niterations +=1
        if self.recompile or self.niterations == 1:
            seq = self.sequence_function(*self.qubits, **self.seq_params)
            self.metafile = compile_to_hardware(seq, "optim/optim")

        exp       = CalibrationExperiment(self.qubits, self.output_nodes, 
                                            self.stream_selectors, self.metafile, 
                                            **self.kwargs)

        #map the "other" parameters to associated qubit or instrument parameters
        #Examples:
        # "{qubit label} {channel} {attribute}"
        # -or-
        # "{instrument} {channel} {attribute}"
        # -or-
        # "{instrument} {attribute}"
        #Is there a cleaner way to do this?"

        if self.other_params:
            for key, value in self.other_params.items():
                spl = key.split(" ")
                chan = None
                if len(spl) == 3:
                    thing = list(filter(lambda q: q.label==spl[0], self.qubits))
                    
                    if len(thing) == 1:
                        qubit = thing[0]
                        attribute = spl[2]
                        if spl[1] == "measure":
                            qubit = qubit.measure_chan
                        elif spl[1] == "control":
                            pass
                        else:
                            raise ValueError(f"Invalid qubit attribute: {spl[0]} {spl[1]}")

                        if qubit.phys_chan.generator and attribute == "frequency":
                            name  = qubit.phys_chan.generator.label
                            instr = list(filter(lambda x: x.name == name, exp._instruments.values()))[0]
                        else:
                            name, chan = qubit.phys_chan.label.split("-")[0:2]
                            instr = exp._instruments[name]

                            if insinstance(instr, auspex.instruments.APS2) and attribute=="amplitude":
                                chan = [1,2]
                    else:
                        try:
                            instr = list(filter(lambda x: x.name == spl[0], exp._instruments.values()))[0]
                        except IndexError:
                            raise ValueError(f"Unknown qubit or instrument {spl[0]}.")
                        chan = spl[1]
                        attribute = spl[2]
                elif len(spl) == 2:
                    try:
                        instr = list(filter(lambda x: x.name == spl[0], exp._instruments.values()))[0]
                    except IndexError:
                        raise ValueError(f"Unknown instrument {spl[0]}.")
                    attribute = spl[1]
                else:
                    raise ValueError(f"Invalid parameter setting: {key}")
                
                if chan:
                    getattr(instr, "set_"+attribute)(chan, value)
                else:
                    getattr(instr, "set_"+attribute)(value)


        if self.fake:
            fake_data = self.fake_data_fn(**self.parameters())
            self.fake_data.append((self.fake_dig, fake_data))
            for fd in self.fake_data:
                exp.set_fake_data(fd[0], fd[1], random_mag=0.0)
        self.exp_config(exp)

        exp.run_sweeps()

        data = {}

        #sort nodes by qubit name to match data with metadata when normalizing
        qubit_indices = {q.label: idx for idx, q in enumerate(exp.qubits)}
        exp.output_nodes.sort(key=lambda x: qubit_indices[x.qubit_name])

        for i, (qubit, output_buff) in enumerate(zip(exp.qubits,
                                [exp.proxy_to_filter[on] for on in exp.output_nodes])):
            if not isinstance(output_buff, DataBuffer):
                raise ValueError("Could not find data buffer for calibration.")

            dataset, descriptor = output_buff.get_data()
            data[qubit.label] = dataset

        # Return data and variance of the mean
        if len(data) == 1:
            # if single qubit, get rid of dictionary
            data = list(data.values())[0]
        return data

    def optimize(self):
        """ Carry out the optimization. """

        if self.do_plotting:
            self.plotters = self.init_plots()
            self.start_plots()

        if self.optimizer == "SCIPY":

            if self.bounds:
                if "method" not in self.optim_params.keys():
                    raise ValueError("A method must be named for bounded optimization with scipy.")
                if "bounds" in self.optim_params.keys():
                    raise ValueError("Please use the `set_bounds` methods to set bounds with scipy.")
                self.optim_params["bounds"] = [v for v in self.bounds.values()]

            x0  = list(self.parameters().values())
            try:
                if self.optim_params:
                    result = minimize(self._optimize_function_scipy(), x0,**self.optim_params)
                else:
                    result = minimize(self._optimize_function_scipy(), x0)
                self.succeeded = result.success
            except CloseEnough:
                self.succeeded = True
                return self.parameters()

            return {k: result.x[j] for j, k in enumerate(self.parameters().keys())}

        if self.optimizer == "BAYES":
            if not self.bounds:
                raise ValueError("Bayesian optimization requires bounds.")

            if self.min_cost:
                logger.warning("Using `min_cost` with Bayesian optimization is not recommended...")

            optim = BayesianOptimization(f = self._optimize_function_bayes(), pbounds=self.bounds)
            try:
                optim.maximize(**self.optim_params)
            except CloseEnough:
                pass

            self.succeeded = True

            return optim.max['params']

class QubitOptimizerCMA(QubitOptimizer):
    """Particle swarm optimization using the CMA-ES algorithm through pycma.
        See http://cma.gforge.inria.fr/cmaes_sourcecode_page.html
    """

    def __init__(self, qubits, sequence_function, cost_function, 
             initial_parameters=None, other_variables=None, scale=True,
             sigma0=None, parameter_scalers=None, 
             optim_params=None, output_nodes=None, 
             stream_selectors=None, do_plotting=True, **kwargs):
        """Setup an optimization over qubit experiments.

        Args:
            qubits: The qubit(s) that the optimization is run over.
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
            scale: Scale optimization parameters.
            sigma0: Initial standard deviation for all optimization parameters, 
                if none is given all SD's are set to 0.5.
            parameter_scalers: Dictionary of callables to scale the parameters such
                that they all have roughly equal magnitude. If None, we scale everything
                to 1 based on the initial value.
            optim_params: Dict of keyword arguments to be passed to the 
                optimization function.
        """
        if "CMA" not in available_optimizers:
            raise ValueError("pyCMA does not appear to be installed.")

        super().__init__(qubits, sequence_function, cost_function, 
                 initial_parameters, other_variables,"cma", optim_params, 
                 output_nodes, stream_selectors, do_plotting, **kwargs)

        if sigma0:
            self.sigma0 = sigma0
        else:
            self.sigma0 = 0.5

        self.scale = scale

        if parameter_scalers:
            #make sure ordering is consistent
            self.parameter_scalers = list(parameter_scalers[k] for k in self.parameters().keys()) 
        else:
            self.parameter_scalers = list(1.0/v for v in self.parameters().values())

    def set_bounds(self, bounds):
        raise NotImplementedError("Bounds are not implemented for CMA-ES optimization.")

    def _optimize_function(self):           
        def _func(x):
            self._update_params(plist=x)
            data = self.run_sweeps()
            cost = self.cost_function(data)
            self.costs.append(cost)
            if self.do_plotting:
                self.update_plots()
            return cost
        return _func

    def optimize(self):
        """ Carry out the optimization. """

        if self.do_plotting:
            self.plotters = self.init_plots()
            self.start_plots()

        x0 = list(self.parameters().values())
        if self.optim_params:
            es = cma.CMAEvolutionStrategy(x0, self.sigma0, self.optim_params)
        else:
            es = cma.CMAEvolutionStrategy(x0, self.sigma0)

        if self.scale:
            opt_func = cma.fitness_transformations.ScaleCoordinates(
                self._optimize_function(), self.parameter_scalers)
        else:
            opt_func = self._optimize_function()

        es.optimize(opt_func)

        return {k: result.xbest[j] for j, k in enumerate(self.parameters.keys())}
            



