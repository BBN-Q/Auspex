# Copyright 2016 Raytheon BBN Technologies
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
from adapt.refine import refine_1D
import os
import uuid
import pandas as pd
import networkx as nx
import scipy as sp
import subprocess
import zmq
import json
import datetime

import time
import bbndb
from auspex.filters import DataBuffer
from .qubit_exp import QubitExperiment
from . import pipeline
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.analysis.fits import *
from auspex.analysis.qubit_fits import *
from auspex.analysis.helpers import normalize_data
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
from itertools import product

import bbndb

class Calibration(object):

    def __init__(self):
        self.do_plotting = True
        self.uuid = str(uuid.uuid4())

    def init_plots(self):
        """Return a ManualPlotter object so we can plot calibrations. All
        plot lines, glyphs, etc. must be declared up front!"""
        return None

    def start_plots(self):
        # Create the descriptor and set uuids for each plot process
        plot_desc = {p.filter_name: p.desc() for p in self.plotters}

        for p in self.plotters:
            p.uuid = self.uuid
        try:
            time.sleep(1.0)
            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.identity = "Auspex_Experiment".encode()
            socket.connect("tcp://localhost:7761")
            socket.send_multipart([self.uuid.encode(), json.dumps(plot_desc).encode('utf8')])

            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)

            evts = dict(poller.poll(5000))
            if socket in evts:
                try:
                    if socket.recv_multipart()[0] == b'ACK':
                        logger.info("Connection established to plot server.")
                        self.do_plotting = True
                    else:
                        raise Exception("Server returned invalid message, expected ACK.")
                except:
                    logger.info("Could not connect to server.")
                    for p in self.plotters:
                        p.do_plotting = False
            else:
                logger.info("Server did not respond.")
                for p in self.plotters:
                    p.do_plotting = False

        except Exception as e:
            logger.warning(f"Exception {e} occured while contacting the plot server. Is it running?")
            for p in self.plotters:
                p.do_plotting = False

        for p in self.plotters:
            p.start()

    def stop_plots(self):
        for p in self.plotters:
            p.start()

    def calibrate(self):
        if self.do_plotting:
            self.plotters = self.init_plots()
            self.start_plots()

        self._calibrate()

        if self.succeeded:
            self.update_settings()

        if self.do_plotting:
            self.stop_plots()

    def update_settings(self):
        # Must be overriden in child class
        pass

    def descriptor(self):
        return None

    def _calibrate(self):
        """Runs the actual calibration routine, must be overridden to provide any useful functionality.
        This function is responsible for calling self.update_plot()"""
        pass

    def exp_config(self, exp):
        """Any final experiment configuration before it gets run."""
        pass

class QubitCalibration(Calibration):
    calibration_experiment = None
    def __init__(self, qubits, sample_name=None, output_nodes=None, quad="real", auto_rollback=True, do_plotting=True, **kwargs):
        self.qubits           = qubits if isinstance(qubits, list) else [qubits]
        self.qubit            = None if isinstance(qubits, list) else qubits
        self.output_nodes     = output_nodes if isinstance(output_nodes, list) else [output_nodes]
        self.filename         = 'None'
        self.axis_descriptor  = None
        self.leave_plots_open = True
        self.cw_mode          = False
        self.quad             = quad
        self.succeeded        = False
        self.norm_points      = None
        self.auto_rollback    = True # Rollback any db changes upon calibration failure
        self.kwargs           = kwargs
        self.plotters         = []
        self.do_plotting      = do_plotting
        self.fake_data        = None
        self.sample           = None
        try:
            self.quad_fun = {"real": np.real, "imag": np.imag, "amp": np.abs, "phase": np.angle}[quad]
        except:
            raise ValueError('Quadrature to calibrate must be one of ("real", "imag", "amp", "phase").')
        super(QubitCalibration, self).__init__()

        if sample_name:
            if not bbndb.session:
                raise Exception("Attempting to load Calibrations database, \
                    but no database session is open! Have the ChannelLibrary and PipelineManager been created?")
            existing_samples = list(bbndb.session.query(bbndb.calibration.Sample).filter_by(name=sample_name).all())
            if len(existing_samples) == 0:
                logger.info("Creating a new sample in the calibration database.")
                self.sample = bbndb.calibration.Sample(name=sample_name)
                bbndb.session.add(self.sample)
            elif len(existing_samples) == 1:
                self.sample = existing_samples[0]
            else:
                raise Exception("Multiple samples found in calibration database with the same name! How?")

    def sequence(self):
        """Returns the sequence for the given calibration, must be overridden"""
        raise NotImplementedError("Must run a specific qubit calibration.")

    def set_fake_data(self, *args, **kwargs):
        self.fake_data = (args, kwargs)

    def run_sweeps(self):
        meta_file = compile_to_hardware(self.sequence(), fileName=self.filename, axis_descriptor=self.descriptor())
        exp       = CalibrationExperiment(self.qubits, self.output_nodes, meta_file, **self.kwargs)
        if self.fake_data:
            exp.set_fake_data(*self.fake_data[0], **self.fake_data[1])
        self.exp_config(exp)
        exp.run_sweeps()

        data = {}
        var = {}

        for qubit, output_buff, var_buff in zip(exp.qubits,
                                [exp.proxy_to_filter[on] for on in exp.output_nodes],
                                [exp.proxy_to_filter[on] for on in exp.var_buffers]):
            if not isinstance(output_buff, DataBuffer):
                raise ValueError("Could not find data buffer for calibration.")

            dataset, descriptor = output_buff.get_data()
            if self.norm_points:
                buff_data = normalize_data(dataset, zero_id=self.norm_points[qubit.label][0],
                                           one_id=self.norm_points[self.qubit.label][1])
            else:
                buff_data = dataset

            data[qubit.label] = self.quad_fun(buff_data)

            var_dataset, var_descriptor = var_buff.get_data()
            # if 'Variance' in dataset.dtype.names:
            realvar = np.real(var_dataset)
            imagvar = np.imag(var_dataset)
            N = descriptor.metadata["num_averages"]
            if self.quad in ['real', 'imag']:
                var[qubit.label] = self.quad_fun(var_dataset)/N
            elif self.quad == 'amp':
                var[qubit.label] = (realvar + imagvar)/N
            elif self.quad == 'phase':
                # take the approach from Qlab assuming the noise is
                # Gaussian in both quadratures i.e. 'circular' in the IQ plane.
                stddata = np.sqrt(realvar + imagvar)
                stdtheta = 180/np.pi * 2 * np.arctan(stddata/abs(data[qubit.label]))
                var[qubit.label] = (stdtheta**2)/N
            else:
                raise Exception('Variance of {} not available. Choose amp, phase, real or imag'.format(self.quad))

        # Return data and variance of the mean
        if len(data) == 1:
            # if single qubit, get rid of dictionary
            data = list(data.values())[0]
            var = list(var.values())[0]
        return data, var

class CalibrationExperiment(QubitExperiment):

    def __init__(self, qubits, output_nodes, *args, **kwargs):
        self.qubits = qubits
        self.output_nodes = output_nodes
        self.var_buffers = []
        super(CalibrationExperiment, self).__init__(*args, **kwargs)

    def guess_output_nodes(self, graph):
        output_nodes = []
        for qubit in self.qubits:
            ds = nx.descendants(graph, str(self.qubit_proxies[qubit.label]))
            outputs = [graph.nodes[d]['node_obj'] for d in ds if isinstance(graph.nodes[d]['node_obj'], (bbndb.auspex.Write, bbndb.auspex.Buffer))]
            if len(outputs) != 1:
                raise Exception(f"More than one output node found for {qubit}, please explicitly define output node using output_nodes argument.")
            output_nodes.append(outputs[0])
        return output_nodes

    def modify_graph(self, graph):
        """Change the graph as needed. By default we changes all writers to buffers"""
        if None in self.output_nodes:
            self.output_nodes = self.guess_output_nodes(graph)

        for output_node in self.output_nodes:
            if output_node.node_label() not in graph:
                raise ValueError(f"Could not find specified output node {output_node} in graph.")

        for qubit in self.qubits:
            if str(self.qubit_proxies[qubit.label]) not in graph:
                raise ValueError(f"Could not find specified qubit {qubit} in graph.")

        mapping = {}
        for i in range(len(self.output_nodes)):
            output_node = self.output_nodes[i]
            if isinstance(output_node, bbndb.auspex.Write):
                # Change the output node to a buffer
                mapping[output_node] = bbndb.auspex.Buffer(label=output_node.label, qubit_name=output_node.qubit_name)

        # Disable any paths not involving the buffer
        new_graph = nx.DiGraph()
        new_output_nodes = []
        for output_node, qubit in zip(self.output_nodes, self.qubits):
            new_output = mapping[output_node]
            new_output_nodes.append(new_output)

            old_path  = nx.shortest_path(graph, self.qubit_proxies[qubit.label].node_label(), output_node.node_label())
            path      = old_path[:-1] + [new_output.node_label()]
            new_graph.add_path(path)
            for n in old_path[:-1]:
                new_graph.nodes[n]['node_obj'] = graph.nodes[n]['node_obj']
            new_graph.nodes[new_output.node_label()]['node_obj'] = mapping[output_node]

            # Fix connectors
            for i in range(len(path)-1):
                new_graph[path[i]][path[i+1]]['connector_in']  = graph[old_path[i]][old_path[i+1]]['connector_in']
                new_graph[path[i]][path[i+1]]['connector_out'] = graph[old_path[i]][old_path[i+1]]['connector_out']

            if not isinstance(new_graph.nodes(data=True)[path[-2]]['node_obj'], bbndb.auspex.Average):
                raise Exception("There is no averager in line.")
            else:
                vb = bbndb.auspex.Buffer(label=f"{output_node.label}-VarBuffer", qubit_name=output_node.qubit_name)
                self.var_buffers.append(vb)
                new_graph.add_node(vb.node_label(), node_obj=vb)
                new_graph.add_edge(path[-2], vb.node_label(), node_obj=vb, connector_in="sink", connector_out="final_variance")
            # maintain standard plots
            plot_nodes = [output_node for output_node in nx.descendants(graph, path[-2]) if isinstance(graph.nodes[output_node]['node_obj'], bbndb.auspex.Display)]
            for plot_node in plot_nodes:
                plot_path = nx.shortest_path(graph, path[-2], plot_node)
                new_graph = nx.compose(new_graph, graph.subgraph(plot_path))

        self.output_nodes = new_output_nodes
        return new_graph

    def add_cal_sweep(self, method, values):
        par = FloatParameter()
        par.assign_method(method)
        self.add_sweep(par, values)


class CavityTuneup(QubitCalibration):
    def __init__(self, qubit, frequencies, averages=750, **kwargs):
        self.start_frequencies = frequencies
        kwargs['averages'] = averages
        super(CavityTuneup, self).__init__(qubit, **kwargs)
        self.cw_mode = True

    def sequence(self):
        return [[Id(self.qubit), MEAS(self.qubit)]]

    def exp_config(self, exp):
        exp.add_qubit_sweep(self.qubit, "measure", "frequency", self.new_frequencies)
        self.quad_fun = lambda x: x

    def _calibrate(self):
        # all_data = np.empty(dtype=np.complex128)
        self.new_frequencies = self.start_frequencies
        self.frequencies = np.empty(0, dtype=np.complex128)
        self.group_delays = np.empty(0, dtype=np.complex128)
        self.datas = np.empty(0, dtype=np.complex128)
        # orig_avg = self.kwargs['averages']
        # Adaptive refinement to find cavity feature
        # for i in range(self.iterations + 1):
        self.data, _      = self.run_sweeps()
        self.datas        = np.append(self.datas, self.data)
        self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])

        ord = np.argsort(self.frequencies)
        self.datas = self.datas[ord]
        self.frequencies = self.frequencies[ord]

        self.phases = np.unwrap(np.angle(self.datas))
        self.group_delays = -np.diff(self.phases)/np.diff(self.frequencies)
        phase_poly = np.poly1d(np.polyfit(self.frequencies, self.phases, 6))
        # group_delay_poly = phase_poly.deriv()
        # fine_freqs = np.linspace(self.frequencies[0], self.frequencies[-1], self.iterations*len(self.frequencies))
        subtracted = self.phases - phase_poly(self.frequencies)
        group_delay = np.diff(subtracted)/np.diff(self.frequencies)

        # ordering = np.argsort(self.frequencies[:-1])
        self.plot1["Phase"] = (self.frequencies, self.phases)
        self.plot1["Phase Fit"] = (self.frequencies,phase_poly(self.frequencies))
        self.plot1B["Group Delay"] = (self.frequencies[:-1],group_delay)
        self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))

        guess = np.abs(self.frequencies[np.argmax(np.abs(group_delay))])
        self.new_frequencies = np.arange(guess-15e6, guess+15e6, 1e6)
        self.frequencies = np.empty(0, dtype=np.complex128)
        self.group_delays = np.empty(0, dtype=np.complex128)
        self.datas = np.empty(0, dtype=np.complex128)

        self.data, _      = self.run_sweeps()
        self.datas        = np.append(self.datas, self.data)
        self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])

        ord = np.argsort(self.frequencies)
        self.datas = self.datas[ord]
        self.frequencies = self.frequencies[ord]

        self.phases = np.unwrap(np.angle(self.datas))
        self.group_delays = -np.diff(self.phases)/np.diff(self.frequencies)
        phase_poly = np.poly1d(np.polyfit(self.frequencies, self.phases, 6))
        # group_delay_poly = phase_poly.deriv()
        # fine_freqs = np.linspace(self.frequencies[0], self.frequencies[-1], self.iterations*len(self.frequencies))
        subtracted = self.phases - phase_poly(self.frequencies)
        group_delay = np.diff(subtracted)/np.diff(self.frequencies)

        # ordering = np.argsort(self.frequencies[:-1])
        self.plot1["Phase"] = (self.frequencies, self.phases)
        self.plot1["Phase Fit"] = (self.frequencies,phase_poly(self.frequencies))
        self.plot1B["Group Delay"] = (self.frequencies[:-1],group_delay)
        self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))

        guess = np.abs(self.frequencies[np.argmax(np.abs(group_delay))])
        self.new_frequencies = np.arange(guess-4e6, guess+4e6, 0.2e6)
        self.frequencies = np.empty(0, dtype=np.complex128)
        self.group_delays = np.empty(0, dtype=np.complex128)
        self.datas = np.empty(0, dtype=np.complex128)

        self.data, _      = self.run_sweeps()
        self.datas        = np.append(self.datas, self.data)
        self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])

        ord = np.argsort(self.frequencies)
        self.datas = self.datas[ord]
        self.frequencies = self.frequencies[ord]

        self.phases = np.unwrap(np.angle(self.datas))
        self.group_delays = -np.diff(self.phases)/np.diff(self.frequencies)
        phase_poly = np.poly1d(np.polyfit(self.frequencies, self.phases, 6))
        # group_delay_poly = phase_poly.deriv()
        # fine_freqs = np.linspace(self.frequencies[0], self.frequencies[-1], self.iterations*len(self.frequencies))
        subtracted = self.phases - phase_poly(self.frequencies)
        group_delay = np.diff(subtracted)/np.diff(self.frequencies)

        # ordering = np.argsort(self.frequencies[:-1])
        self.plot1["Phase"] = (self.frequencies, self.phases)
        self.plot1["Phase Fit"] = (self.frequencies,phase_poly(self.frequencies))
        self.plot1B["Group Delay"] = (self.frequencies[:-1],group_delay)

        self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))

        shifted_cav = np.real(self.datas) - np.mean(np.real(self.datas))
        guess = np.abs(self.frequencies[np.argmax(np.abs(shifted_cav))])
            # self.kwargs['averages'] = 2000

            # import pdb; pdb.set_trace()
            #
            # self.new_frequencies = refine_1D(self.frequencies, subtracted, all_points=False,
            #                             criterion="difference", threshold = "one_sigma")
            # logger.info(f"new_frequencies {self.new_frequencies}")

        # n, bins = sp.histogram(np.abs(self.frequencies), bins="auto")
        # f_start = bins[np.argmax(n)]
        # f_stop  = bins[np.argmax(n)+1]
        # logger.info(f"Looking in bin from {f_start} to {f_stop}")

        # # self.kwargs['averages'] = orig_avg
        # self.new_frequencies = np.arange(f_start, f_stop, 2e6)
        # self.frequencies = np.empty(0, dtype=np.complex128)
        # self.group_delays = np.empty(0, dtype=np.complex128)
        # self.datas = np.empty(0, dtype=np.complex128)
        #
        # for i in range(self.iterations + 3):
        #     self.data, _      = self.run_sweeps()
        #     self.datas        = np.append(self.datas, self.data)
        #     self.frequencies  = np.append(self.frequencies, self.new_frequencies[:-1])
        #
        #     ord = np.argsort(self.frequencies)
        #     self.datas = self.datas[ord]
        #     self.frequencies = self.frequencies[ord]
        #
        #     self.group_delays = -np.diff(np.unwrap(np.angle(self.datas)))/np.diff(self.frequencies)
        #     # self.group_delays = group_del
        #
        #     # ordering = np.argsort(self.frequencies[:-1])
        #     self.plot3["Group Delay"] = (self.frequencies[1:],self.group_delays)
        #     # self.plot2["Amplitude"] = (self.frequencies,np.abs(self.datas))
        #     # self.kwargs['averages'] = 2000
        #
        #     self.new_frequencies = refine_1D(self.frequencies[:-1], self.group_delays, all_points=False,
        #                                 criterion="integral", threshold = "one_sigma")
        #     logger.info(f"new_frequencies {self.new_frequencies}")
        # #

        # # self.data, _ = self.run_sweeps()
        # # group_delay = -np.diff(np.unwrap(np.angle(self.data)))/np.diff(self.new_frequencies)
        # # self.plot3["Group Delay"] = (self.new_frequencies[1:],group_delay)
        #
        # def lor_der(x, a, x0, width, offset):
        #     return offset-(x-x0)*a/((4.0*((x-x0)/width)**2 + a**2)**2)
        # f0 = np.abs(self.frequencies[np.argmax(np.abs(self.group_delays))])
        # p0 = [np.max(np.abs(self.group_delays))*1e-18, np.abs(f0), 200e6, np.abs(self.group_delays)[0]]
        # popt, pcov = curve_fit(lor_der, np.abs(self.frequencies[1:]), np.abs(self.group_delays), p0=p0)
        # self.plot3["Group Delay Fit"] = ( np.abs(self.frequencies[1:]),  lor_der( np.abs(self.frequencies[1:]), *popt))


    def init_plots(self):
        plot1 = ManualPlotter("Phase", x_label='Frequency (GHz)', y_label='Group Delay')
        plot1.add_data_trace("Phase", {'color': 'C1'})
        plot1.add_fit_trace("Phase Fit", {'color': 'C2'})

        plot1B = ManualPlotter("Group Delay", x_label='Frequency (GHz)', y_label='Group Delay')
        plot1B.add_data_trace("Group Delay", {'color': 'C1'})
        # plot1B.add_fit_trace("Phase Fit", {'color': 'C2'})

        plot2 = ManualPlotter("Amplitude", x_label='Frequency (GHz)', y_label='Amplitude (Arb. Units)')
        plot2.add_data_trace("Amplitude", {'color': 'C2'})

        # plot3 = ManualPlotter("First refined sweep", x_label='Frequency (GHz)', y_label='Group Delay')
        # plot3.add_data_trace("Group Delay", {'color': 'C3'})
        # plot3.add_fit_trace("Group Delay Fit", {'color': 'C4'})
        self.plot1 = plot1
        self.plot1B = plot1B
        self.plot2 = plot2
        # self.plot3 = plot3
        return [plot1, plot1B, plot2] #, plot3]

class QubitTuneup(QubitCalibration):
    def __init__(self, qubit, f_start=5e9, f_stop=6e9, coarse_step=0.1e9, fine_step=1.0e6, averages=500, amp=1.0, **kwargs):
        self.coarse_frequencies = np.arange(f_start, f_stop, coarse_step) - 10.0e6 # Don't stray too close to the carrier tone
        self.fine_frequencies   = np.arange(10.0e6, coarse_step+10.0e6, fine_step)
        self.f_start = f_start
        self.f_stop = f_stop
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.amp = amp
        kwargs['averages'] = averages
        super(QubitTuneup, self).__init__(qubit, **kwargs)

    def sequence(self):
        return [[X(self.qubit, frequency=f, amp=self.amp), MEAS(self.qubit)] for f in self.fine_frequencies]

    def exp_config(self, exp):
        exp.add_qubit_sweep(self.qubit, "control", "frequency", self.coarse_frequencies)
        self.quad_fun = lambda x: x

    def _calibrate(self):
        self.data, _ = self.run_sweeps()
        freqs = np.arange(self.f_start, self.f_stop, self.fine_step)
        self.plot["Data"] = (freqs, self.data)

    def init_plots(self):
        plot = ManualPlotter("Qubit Search", x_label='Frequency (Hz)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data", {'color': 'C1'})
        plot.add_fit_trace("Fit", {'color': 'C1'})
        self.plot = plot
        return [plot]

class RabiAmpCalibration(QubitCalibration):

    amp2offset = 0.5

    def __init__(self, qubit, num_steps=40, **kwargs):
        if num_steps % 2 != 0:
            raise ValueError("Number of steps for RabiAmp calibration must be even!")
        #for now, only do one qubit at a time
        self.num_steps = num_steps
        self.amps = np.hstack((np.arange(-1, 0, 2./num_steps),
                               np.arange(2./num_steps, 1+2./num_steps, 2./num_steps)))
        super(RabiAmpCalibration, self).__init__(qubit, **kwargs)
        self.filename = 'Rabi/Rabi'

    def sequence(self):
        return ([[Xtheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps] +
                [[Ytheta(self.qubit, amp=a), MEAS(self.qubit)] for a in self.amps])

    def _calibrate(self):
        data, _ = self.run_sweeps()
        N = len(data)
        I_fit = RabiAmpFit(self.amps, data[:N//2])
        Q_fit = RabiAmpFit(self.amps, data[:N//2])
        #Arbitary extra division by two so that it doesn't push the offset too far.
        self.pi_amp = I_fit.fit_params["Api"]
        self.pi2_amp = I_fit.fit_params["Api"]/2.0
        self.i_offset = I_fit.fit_params["phi"]*self.amp2offset
        self.q_offset = Q_fit.fit_params["phi"]*self.amp2offset
        logger.info("Found X180 amplitude: {}".format(self.pi_amp))
        logger.info("Shifting I offset by: {}".format(self.i_offset))
        logger.info("Shifting Q offset by: {}".format(self.q_offset))
        finer_amps = np.linspace(np.min(self.amps), np.max(self.amps), 4*len(self.amps))
        self.plot["I Data"] = (self.amps, data[:N//2])
        self.plot["Q Data"] = (self.amps, data[N//2:])
        self.plot["I Fit"] = (finer_amps, I_fit.model(finer_amps))
        self.plot["Q Fit"] = (finer_amps, Q_fit.model(finer_amps))

        if self.pi_amp <= 1.0 and self.pi2_amp <= 1.0:
            self.succeeded = True

    def init_plots(self):
        plot = ManualPlotter("Rabi Amplitude Cal", x_label="I/Q Amplitude", y_label="{} (Arb. Units)".format(self.quad))
        plot.add_data_trace("I Data", {'color': 'C1'})
        plot.add_data_trace("Q Data", {'color': 'C2'})
        plot.add_fit_trace("I Fit", {'color': 'C1'})
        plot.add_fit_trace("Q Fit", {'color': 'C2'})
        self.plot = plot
        return [plot]

    def update_settings(self):
        s = round(self.pi_amp, 5)
        self.qubit.pulse_params['pi2Amp'] = round(self.pi2_amp, 5)
        self.qubit.pulse_params['piAmp'] = round(self.pi_amp, 5)
        awg_chan   = self.qubit.phys_chan
        amp_factor = self.qubit.phys_chan.amp_factor
        awg_chan.I_channel_offset += round(amp_factor*self.amp2offset*self.i_offset, 5)
        awg_chan.Q_channel_offset += round(amp_factor*self.amp2offset*self.i_offset, 5)

        if self.sample:
            c1 = bbndb.calibration.Calibration(value=self.pi2_amp, sample=self.sample, name="Pi2Amp", category="Rabi")
            c2 = bbndb.calibration.Calibration(value=self.pi_amp, sample=self.sample, name="PiAmp", category="Rabi")
            c1.date = c2.date = datetime.datetime.now()
            bbndb.session.add_all([c1, c2])
            bbndb.session.commit()

class RamseyCalibration(QubitCalibration):
    def __init__(self, qubit, delays=np.linspace(0.0, 20.0, 41)*1e-6,
                two_freqs=False, added_detuning=150e3, set_source=True, AIC=True, **kwargs):
        self.delays         = delays
        self.two_freqs      = two_freqs
        self.added_detuning = added_detuning
        self.set_source     = set_source
        self.AIC            = AIC #Akaike information criterion for model choice

        super(RamseyCalibration, self).__init__(qubit, **kwargs)
        self.filename = 'Ramsey/Ramsey'

    def descriptor(self):
        return [delay_descriptor(self.delays)]

    def sequence(self):
        return [[X90(self.qubit), Id(self.qubit, delay), X90(self.qubit), MEAS(self.qubit)] for delay in self.delays]

    def init_plots(self):
        plot = ManualPlotter("Ramsey Fits", x_label='Time (us)', y_label='Amplitude (Arb. Units)')
        plot.add_data_trace("Data 1", {'color': 'black'})
        plot.add_fit_trace("Fit 1", {'color': 'red'})
        plot.add_data_trace("Data 2", {'color': 'green'})
        plot.add_fit_trace("Fit 2", {'color': 'blue'})
        self.plot = plot
        return [plot]

    def exp_config(self, exp):
        rcvr = self.qubit.measure_chan.receiver_chan.receiver
        if self.first_ramsey:
            self.source_proxy = self.qubit.phys_chan.generator # DB object
            self.qubit_source = exp._instruments[self.source_proxy.label] # auspex instrument
            self.orig_freq    = self.source_proxy.frequency
            if self.set_source:
                self.source_proxy.frequency = round(self.orig_freq + self.added_detuning, 10)
                self.qubit_source.frequency = self.source_proxy.frequency
            exp._instruments[rcvr.label].exp_step = 0
        else:
            exp._instruments[rcvr.label].exp_step = 1

    def _calibrate(self):
        self.first_ramsey = True

        if not self.set_source:
            self.qubit.frequency += float(self.added_detuning)
        data, _ = self.run_sweeps()
        try:
            ramsey_fit = RamseyFit(self.delays, data, two_freqs=self.two_freqs, AIC=self.AIC)
            fit_freqs = ramsey_fit.fit_params["f"]
        except Exception as e:
            raise Exception(f"Exception {e} while fitting in {self}")

        # Plot the results
        self.plot["Data 1"] = (self.delays, data)
        finer_delays = np.linspace(np.min(self.delays), np.max(self.delays), 4*len(self.delays))
        self.plot["Fit 1"] = (finer_delays, ramsey_fit.model(finer_delays))

        #TODO: set conditions for success
        fit_freq_A = np.mean(fit_freqs) #the fit result can be one or two frequencies
        if self.set_source:
            self.source_proxy.frequency = round(self.orig_freq + self.added_detuning + fit_freq_A/2, 10)
            self.qubit_source.frequency = self.source_proxy.frequency
        else:
            self.qubit.frequency += float(fit_freq_A/2)

        self.first_ramsey = False

        # if self.plot:
        #     [self.add_manual_plotter(p) for p in self.plot] if isinstance(self.plot, list) else self.add_manual_plotter(self.plot)
        # self.start_manual_plotters()
        data, _ = self.run_sweeps()

        try:
            ramsey_fit = RamseyFit(self.delays, data, two_freqs=self.two_freqs, AIC=self.AIC)
            fit_freqs = ramsey_fit.fit_params["f"]
        except Exception as e:
            raise Exception(f"Exception {e} while fitting in {self}")

        # Plot the results
        self.plot["Data 2"] = (self.delays, data)
        self.plot["Fit 2"]  = (finer_delays, ramsey_fit.model(finer_delays))

        fit_freq_B = np.mean(fit_freqs)
        if fit_freq_B < fit_freq_A:
            self.fit_freq = round(self.orig_freq + self.added_detuning + 0.5*(fit_freq_A + 0.5*fit_freq_A + fit_freq_B), 10)
        else:
            self.fit_freq = round(self.orig_freq + self.added_detuning - 0.5*(fit_freq_A - 0.5*fit_freq_A + fit_freq_B), 10)

    def update_settings(self):
        if self.set_source:
            self.source_proxy.frequency = float(round(self.fit_freq))
            self.qubit_source.frequency = self.source_proxy.frequency
        else:
            self.qubit.frequency += float(round(self.fit_freq - self.orig_freq))
        # update edges where this is the target qubit
        for edge in self.qubit.edge_target:
            edge_source = edge.phys_chan.generator
            edge.frequency = self.source_proxy.frequency + self.qubit_source.frequency - edge_source.frequency
        #         # TODO: fix this for db backend

        # qubit_set_freq = self.saved_settings['instruments'][qubit_source]['frequency'] + self.saved_settings['qubits'][self.qubit.label]['control']['frequency']
        # logger.info("Qubit set frequency = {} GHz".format(round(float(qubit_set_freq/1e9),5)))
        # return ('frequency', qubit_set_freq)

class PhaseEstimation(QubitCalibration):

    amp2offset = 0.5

    def __init__(self, qubit, num_pulses= 1, amplitude= 0.1, direction = 'X',
                    target=np.pi/2, epsilon=1e-2, max_iter=5, **kwargs):
        #for now, only do one qubit at a time
        self.num_pulses = num_pulses
        self.amplitude = amplitude
        self.direction = direction

        self.target = target
        self.epsilon = epsilon
        self.max_iter = max_iter

        super(PhaseEstimation, self).__init__(qubit, **kwargs)

        self.filename = 'PhaseCal/PhaseCal'

    def sequence(self):
        # Determine whether it is a single- or a two-qubit pulse calibration
        if isinstance(self.qubit, list):
            qubit = self.qubit[1]
            cal_pulse = [ZX90_CR(*self.qubit, amp=self.amplitude)]
        else:
            qubit = self.qubit
            cal_pulse = [Xtheta(self.qubit, amp=self.amplitude)]

        # Exponentially growing repetitions of the target pulse, e.g.
        # (1, 2, 4, 8, 16, 32, 64, 128, ...) x X90
        seqs = [cal_pulse*n for n in 2**np.arange(self.num_pulses+1)]
        # measure each along Z or Y
        seqs = [s + m for s in seqs for m in [ [MEAS(qubit)], [X90m(qubit), MEAS(qubit)] ]]
        # tack on calibrations to the beginning
        seqs = [[Id(qubit), MEAS(qubit)], [X(qubit), MEAS(qubit)]] + seqs
        # repeat each
        return [copy(s) for s in seqs for _ in range(2)]

    def _calibrate(self):

        ct = 0
        done = 0

        start_amp = self.amplitude

        phase_error = []

        while not done and ct < self.max_iter:
            ct += 1
            data, var = self.run_sweeps()
            phase, sigma = phase_estimation(data, var)
            self.amplitude, done, error = phase_to_amplitude(phase, sigma, self.amplitude,
                                                self.target, epsilon=self.epsilon)
            phase_error.append(error)

            self.data_plot['data'] = (np.array(range(1, len(data)+1)), data)
            self.plot["angle_estimate"] = (np.array(range(1, len(phase_error)+1)), np.array(phase_error))

        if done == -1:
            self.succeeded = False
        elif done == 1:
            self.succeeded = True
        else:
            raise Exception()

    def init_plots(self):
        data_plot = ManualPlotter("Phase Cal", x_label="Sequence Number", y_label="{} (Arb. Units)".format(self.quad))
        data_plot.add_data_trace("data", {'color': 'C1'})
        plot = ManualPlotter("Phase Angle Error", x_label="Iteration", y_label="Angle (rad.)")
        plot.add_data_trace("angle_estimate", {'color': 'C1'})
        self.plot = plot
        self.data_plot = data_plot
        return [data_plot, plot]

    def update_settings(self):
        logger.warning("Nothing to update.")


class Pi2Calibration(PhaseEstimation):

    def __init__(self, qubit, num_pulses= 1, direction = 'X',
                    epsilon=1e-2, max_iter=5, **kwargs):
        super(Pi2Calibration, self).__init__(qubit, num_pulses=num_pulses,
                        amplitude=qubit.pulse_params['pi2Amp'], direction =direction,
                        target=np.pi/2, epsilon=epsilon, max_iter=max_iter, **kwargs)

    def update_settings(self):
        self.qubit.pulse_params['pi2Amp'] = round(self.amplitude, 5)

        if self.sample:
            c = bbndb.calibration.Calibration(value=self.amplitude, sample=self.sample, name="Pi2Amp", category="PhaseEstimation")
            c.date = datetime.datetime.now()
            bbndb.session.add(c)
            bbndb.session.commit()

class PiCalibration(PhaseEstimation):

    def __init__(self, qubit, num_pulses= 1, direction = 'X',
                    epsilon=1e-2, max_iter=5, **kwargs):
        super(PiCalibration, self).__init__(qubit, num_pulses=num_pulses,
                        amplitude=qubit.pulse_params['piAmp'], direction =direction,
                        target=np.pi, epsilon=epsilon, max_iter=max_iter, **kwargs)

    def update_settings(self):
        self.qubit.pulse_params['piAmp'] = round(self.amplitude, 5)

        if self.sample:
            c = bbndb.calibration.Calibration(value=self.amplitude, sample=self.sample, name="PiAmp", category="PhaseEstimation")
            c.date = datetime.datetime.now()
            bbndb.session.add(c)
            bbndb.session.commit()

# class CRAmpCalibration_PhEst(PhaseEstimation):
#     def __init__(self, qubit_names, num_pulses= 9):
#         super(CRAmpCalibration_PhEst, self).__init__(qubit_names, num_pulses = num_pulses)
#         self.CRchan = ChannelLibraries.EdgeFactory(*self.qubit)
#         self.amplitude = self.CRchan.pulse_params['amp']
#         self.target    = np.pi/2
#         self.edge_name = self.CRchan.label

class DRAGCalibration(QubitCalibration):
    def __init__(self, qubit, deltas = np.linspace(-1,1,21), num_pulses = np.arange(8, 48, 4), **kwargs):
        self.filename = 'DRAG/DRAG'
        self.deltas = deltas
        self.num_pulses = num_pulses
        super(DRAGCalibration, self).__init__(qubit, **kwargs)

    def sequence(self):
        seqs = []
        for n in self.num_pulses:
            seqs += [[X90(self.qubit, drag_scaling = d), X90m(self.qubit, drag_scaling = d)]*n + [X90(self.qubit, drag_scaling = d), MEAS(self.qubit)] for d in self.deltas]
        seqs += create_cal_seqs((self.qubit,),2)
        return seqs

    def init_plots(self):
        plot = ManualPlotter("DRAG Cal", x_label=['DRAG parameter', 'Number of pulses'], y_label=['Amplitude (Arb. Units)', 'Fit DRAG parameter'], numplots = 2)
        cmap = cm.viridis(np.linspace(0, 1, len(self.num_pulses)))
        for n in range(len(self.num_pulses)):
            plot.add_data_trace('Data_{}'.format(n), {'color': list(cmap[n]), 'linestyle': 'None'})
            plot.add_fit_trace('Fit_{}'.format(n), {'color': list(cmap[n])})
        plot.add_data_trace('Data_opt', subplot_num = 1) #TODO: error bars
        self.plot = plot
        return [plot]

    def exp_config(self, exp):
        rcvr = self.qubit.measure_chan.receiver_chan.receiver
        exp._instruments[rcvr.label].exp_step = self.step #where from?

    def _calibrate(self):
        # run twice for different DRAG parameter ranges
        for k in range(2):
            self.step = k
            data, _ = self.run_sweeps()
            finer_deltas = np.linspace(np.min(self.deltas), np.max(self.deltas), 4*len(self.deltas))
            #normalize data with cals
            data = quick_norm_data(data)
            try:
                opt_drag, error_drag, popt_mat = fit_drag(data, self.deltas, self.num_pulses)
                if k==1:
                    self.succeeded = True
            except Exception as e:
                raise Exception(f"Exception {e} while fitting in {self}")

            norm_data = data.reshape((len(self.num_pulses), len(self.deltas)))
            for n in range(len(self.num_pulses)):
                self.plot['Data_{}'.format(n)] = (self.deltas, norm_data[n, :])
                finer_deltas = np.linspace(np.min(self.deltas), np.max(self.deltas), 4*len(self.deltas))
                self.plot['Fit_{}'.format(n)] = (finer_deltas, quadf(finer_deltas, *popt_mat[:, n]))
            self.plot["Data_opt"] = (self.num_pulses, opt_drag) #TODO: add error bars

            if k==0:
                #generate sequence with new pulses and drag parameters
                new_drag_step = 0.25*(max(self.deltas) - min(self.deltas))
                self.deltas = np.linspace(opt_drag[-1] - new_drag_step, opt_drag[-1] + new_drag_step, len(self.deltas))
                new_pulse_step = int(np.floor(2*(max(self.num_pulses)-min(self.num_pulses))/len(self.num_pulses)))
                self.num_pulses = np.arange(max(self.num_pulses) - new_pulse_step, max(self.num_pulses) + new_pulse_step*(len(self.num_pulses)-1), new_pulse_step)

            if not self.leave_plots_open:
                self.plot.set_quit()
        self.opt_drag = round(float(opt_drag[-1]), 5)

    def update_settings(self):
        logger.info(f'{self.qubit.label} DRAG parameter set to {self.opt_drag}')
        self.qubit.pulse_params['drag_scaling'] = self.opt_drag

        if self.sample:
            c = bbndb.calibration.Calibration(value=self.opt_drag, sample=self.sample, name="drag_scaling")
            c.date = datetime.datetime.now()
            bbndb.session.add(c)
            bbndb.session.commit()

# class MeasCalibration(Calibration):
#     def __init__(self, qubit_name):
#         super(MeasCalibration, self).__init__(qubit, **kwargs)
#         self.meas_name = "M-" + qubit_name

# class CLEARCalibration(MeasCalibration):
#     '''
#     Calibration of cavity reset pulse
#     aux_qubit: auxiliary qubit used for CLEAR pulse
#     kappa: cavity linewidth (angular frequency: 1/s)
#     chi: half of the dispershive shift (angular frequency: 1/s)
#     tau: duration of each of the 2 depletion steps (s)
#     alpha: scaling factor
#     T1factor: decay due to T1 between end of msm't and start of Ramsey
#     T2: measured T2*
#     nsteps: calibration steps/sweep
#     cal_steps: choose ranges for calibration steps. 1: +-100%; 0: skip step
#     '''
#     def __init__(self, qubit, aux_qubit, kappa = 2e6, chi = 1e6, t_empty = 200e-9, ramsey_delays=np.linspace(0.0, 50.0, 51)*1e-6, ramsey_freq = 100e3, meas_delay = 0, tau = 200e-9, \
#     alpha = 1, T1factor = 1, T2 = 30e-6, nsteps = 11, eps1 = None, eps2 = None, cal_steps = (1,1,1)):
#         super(CLEARCalibration, self).__init__(qubit, **kwargs)
#         self.filename = 'CLEAR/CLEAR'
#         self.aux_qubit = aux_qubit
#         self.kappa = kappa
#         self.chi = chi
#         self.ramsey_delays = ramsey_delays
#         self.ramsey_freq = ramsey_freq
#         self.meas_delay = meas_delay
#         self.tau = tau
#         self.alpha = alpha
#         self.T1factor = T1factor
#         self.T2 = T2
#         self.nsteps = nsteps
#         if not eps1:
#             # theoretical values as default
#             self.eps1 = (1 - 2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))/(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
#             self.eps2 = 1/(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
#         self.cal_steps = cal_steps

#     def sequence(self, **params):
#         qM = QubitFactory(self.aux_qubit) #TODO: replace with MEAS(q) devoid of digitizer trigger
#         prep = X(self.qubit) if params['state'] else Id(self.qubit)
#         seqs = [[prep, MEAS(qM, amp1 = params['eps1'], amp2 =  params['eps2'], step_length = self.tau), X90(self.qubit), Id(self.qubit,d), U90(self.qubit,phase = self.ramsey_freq*d),
#         Id(self.qubit, self.meas_delay), MEAS(self.qubit)] for d in self.ramsey_delays]
#         seqs += create_cal_seqs((self.qubit,), 2, delay = self.meas_delay)
#         return seqs

#     def init_plots(self): #keep in a single plot?
#         plot_raw = ManualPlotter("CLEAR Ramsey", x_label='Time (us)', y_label='<Z>')
#         plot_res = ManualPlotter("CLEAR Cal", x_label= ['eps1, eps2', 'eps1', 'eps2'], y_label=['Residual photons n0', '', ''], numplots=3)

#         plot_raw.add_data_trace('Data')
#         plot_raw.add_fit_trace('Fit')
#         for sweep_num, state in product([0,1,2], [0,1]):
#             plot_res.add_data_trace('sweep {}, state {}'.format(sweep_num, state), {'color': 'C{}'.format(state+1)}, sweep_num) #TODO: error bar
#             plot_res.add_fit_trace('Fit sweep {}, state {}'.format(sweep_num, state), {'color' : 'C{}'.format(state+1)}, sweep_num) #TODO
#         return [plot_raw, plot_res]


#     def calibrate(self):
#         cal_step = 0
#         for ct in range(3):
#             if not self.cal_steps[ct]:
#                 continue
#             #generate sequence
#             xpoints = np.linspace(1-self.cal_steps[ct], 1+self.cal_steps[ct], self.nsteps)
#             n0vec = np.zeros(self.nsteps)
#             err0vec = np.zeros(self.nsteps)
#             n1vec = np.zeros(self.nsteps)
#             err1vec = np.zeros(self.nsteps)
#             for k in range(self.nsteps):
#                 eps1 = self.eps1 if k==1 else xpoints[k]*self.eps1
#                 eps2 = self.eps2 if k==2 else xpoints[k]*self.eps2
#                 #run for qubit in 0/1
#                 for state in [0,1]:
#                     self.set(eps1 = eps1, eps2 = eps2, state = state, exp_step = cal_step)
#                     #analyze
#                     data, _ = self.run()
#                     norm_data = quick_norm_data(data)
#                     eval('n{}vec'.format(state))[k], eval('err{}vec'.format(state))[k], fit_curve = fit_photon_number(self.ramsey_delays, norm_data, [self.kappa, self.ramsey_freq, 2*self.chi, self.T2, self.T1factor, 0])
#                     #plot
#                     self.plot[0]['Data'] = (self.ramsey_delays, norm_data)
#                     self.plot[0]['Fit'] = fit_curve
#                     self.plot[1]['sweep {}, state 0'.format(ct)] = (xpoints, n0vec)
#                     self.plot[1]['sweep {}, state 1'.format(ct)] = (xpoints, n1vec)
#                     cal_step+=1

#             #fit for minimum photon number
#             popt_0,_ = fit_quad(xpoints, n0vec)
#             popt_1,_ = fit_quad(xpoints, n1vec)
#             finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
#             opt_scaling = np.mean(popt_0[0], popt_1[0])
#             logger.info("Optimal scaling factor for step {} = {}".format(ct+1, opt_scaling))

#             if ct<2:
#                 self.eps1*=opt_scaling
#             if ct!=1:
#                 self.eps2*=opt_scaling
#             self.plot[1]['Fit sweep {}, state 0'.format(ct)] = (finer_xpoints, quadf(finer_xpoints, popt_0))
#             self.plot[1]['Fit sweep {}, state 1'.format(ct)] = (finer_xpoints, quadf(finer_xpoints, popt_1))

#         def update_settings(self):
#             #update library (default amp1, amp2 for MEAS)
#             self.saved_settings['qubits'][self.qubit.label]['measure']['pulse_params']['amp1'] = round(float(self.eps1), 5)
#             self.saved_settings['qubits'][self.qubit.label]['measure']['pulse_params']['amp2'] = round(float(self.eps2), 5)
#             self.saved_settings['qubits'][self.qubit.label]['measure']['pulse_params']['step_length'] = round(float(self.tau), 5)
#             super(CLEARCalibration, self).update_settings()

# '''Two-qubit gate calibrations'''
# class CRCalibration(Calibration):
#     def __init__(self, qubit_names, lengths=np.linspace(20, 1020, 21)*1e-9, phase = 0, amp = 0.8, rise_fall = 40e-9):
#         super(CRCalibration, self).__init__(qubit_names, factory)
#         self.lengths = lengths
#         self.phases = phase
#         self.amps = amp
#         self.rise_fall = rise_fall
#         self.filename = 'CR/CR'
#         self.edge_name = ChannelLibraries.EdgeFactory(*self.qubit).label

#     def init_plots(self):
#         plot = ManualPlotter("CR"+str.lower(self.cal_type.name)+"Fit", x_label=str.lower(self.cal_type.name), y_label='$<Z_{'+self.qubit_names[1]+'}>$', y_lim=(-1.02,1.02))
#         plot.add_data_trace("Data 0", {'color': 'C1'})
#         plot.add_fit_trace("Fit 0", {'color': 'C1'})
#         plot.add_data_trace("Data 1", {'color': 'C2'})
#         plot.add_fit_trace("Fit 1", {'color': 'C2'})
#         return plot

#     def calibrate(self):
#         # generate sequence
#         self.set()
#         # run and load normalized data
#         data, _ = self.run(norm_points = {self.qubit_names[0]: (0, 1), self.qubit_names[1]: (0, 2)})
#         # select target qubit
#         data_t = data[self.qubit_names[1]]
#         # fit
#         self.opt_par, all_params_0, all_params_1 = fit_CR([self.lengths, self.phases, self.amps], data_t, self.cal_type)
#         # plot the result
#         xaxis = self.lengths if self.cal_type==CR_cal_type.LENGTH else self.phases if self.cal_type==CR_cal_type.PHASE else self.amps
#         finer_xaxis = np.linspace(np.min(xaxis), np.max(xaxis), 4*len(xaxis))
#         self.plot["Data 0"] = (xaxis,       data_t[:len(data_t)//2])
#         self.plot["Fit 0"] =  (finer_xaxis, np.polyval(all_params_0, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, *all_params_0))
#         self.plot["Data 1"] = (xaxis,       data_t[len(data_t)//2:])
#         self.plot["Fit 1"] =  (finer_xaxis, np.polyval(all_params_1, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, *all_params_1))
#         return (str.lower(self.cal_type.name), self.opt_par)

#     def update_settings(self):
#         self.saved_settings['edges'][self.edge_name]['pulse_params'][str.lower(self.cal_type.name)] = float(self.opt_par)
#         super(CRCalibration, self).update_settings()

# class CRLenCalibration(CRCalibration):
#     def __init__(self, qubit_names, factory, lengths=np.linspace(20, 1020, 21)*1e-9, phase = 0, amp = 0.8, rise_fall = 40e-9, cal_type = CR_cal_type.LENGTH):
#         self.cal_type = cal_type
#         super(CRLenCalibration, self).__init__(qubit_names, factory, lengths, phase, amp, rise_fall)

#     def sequence(self):
#         qc, qt = self.qubit
#         seqs = [[Id(qc)] + echoCR(qc, qt, length=l, phase = self.phases, amp=self.amps, riseFall=self.rise_fall).seq + [Id(qc), MEAS(qt)*MEAS(qc)]
#         for l in self.lengths]+ [[X(qc)] + echoCR(qc, qt, length=l, phase= self.phases, amp=self.amps, riseFall=self.rise_fall).seq + [X(qc), MEAS(qt)*MEAS(qc)]
#         for l in self.lengths] + create_cal_seqs((qt,qc), 2, measChans=(qt,qc))

#         self.axis_descriptor=[
#             delay_descriptor(np.concatenate((self.lengths, self.lengths))),
#             cal_descriptor(tuple(self.qubit), 2)
#         ]

#         return seqs

# class CRPhaseCalibration(CRCalibration):
#     def __init__(self, qubit_names, factory, phases = np.linspace(0,2*np.pi,21), amp = 0.8, rise_fall = 40e-9, cal_type = CR_cal_type.PHASE):
#         self.cal_type = cal_type
#         super(CRPhaseCalibration, self).__init__(qubit_names, factory, 0, phases, amp, rise_fall)
#         CRchan = ChannelLibraries.EdgeFactory(*self.qubit)
#         self.lengths = CRchan.pulse_params['length']


#     def sequence(self):
#         qc, qt = self.qubit
#         seqs = [[Id(qc)] + echoCR(qc, qt, length=self.lengths, phase=ph, amp=self.amps, riseFall=self.rise_fall).seq + [X90(qt)*Id(qc), MEAS(qt)*MEAS(qc)]
#         for ph in self.phases]+ [[X(qc)] + echoCR(qc, qt, length=self.lengths, phase= ph, amp=self.amps, riseFall=self.rise_fall).seq + [X90(qt)*X(qc), MEAS(qt)*MEAS(qc)]
#         for ph in self.phases] + create_cal_seqs((qt,qc), 2, measChans=(qt,qc))

#         self.axis_descriptor = [
#             {
#                 'name': 'phase',
#                 'unit': 'radians',
#                 'points': list(self.phases)+list(self.phases),
#                 'partition': 1
#             },
#             cal_descriptor(tuple(self.qubit), 2)
#         ]

#         return seqs

# class CRAmpCalibration(CRCalibration):
#     def __init__(self, qubit_names, factory, amp_range = 0.4, amp = 0.8, rise_fall = 40e-9, num_CR = 1, cal_type = CR_cal_type.AMP):
#         self.num_CR = num_CR
#         if num_CR % 2 == 0:
#             logger.error('The number of ZX90 must be odd')
#         self.cal_type = cal_type
#         amps = np.linspace((1-amp_range/2)*amp, (1+amp_range/2)*amp, 21)
#         super(CRAmpCalibration, self).__init__(qubit_names, factory, 0, 0, amps, rise_fall)
#         CRchan = ChannelLibraries.EdgeFactory(*self.qubit)
#         self.lengths = CRchan.pulse_params['length']
#         self.phases = CRchan.pulse_params['phase']

#     def sequence(self):
#         qc, qt = self.qubit
#         seqs = [[Id(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths, phase=self.phases, amp=a, riseFall=self.rise_fall).seq + [Id(qc), MEAS(qt)*MEAS(qc)]
#         for a in self.amps]+ [[X(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths, phase= self.phases, amp=a, riseFall=self.rise_fall).seq + [X(qc), MEAS(qt)*MEAS(qc)]
#         for a in self.amps] + create_cal_seqs((qt,qc), 2, measChans=(qt,qc))

#         self.axis_descriptor = [
#             {
#                 'name': 'amplitude',
#                 'unit': None,
#                 'points': list(self.amps)+list(self.amps),
#                 'partition': 1
#             },
#             cal_descriptor(tuple(self.qubit), 2)
#         ]

#         return seqs

def restrict(phase):
    out = np.mod( phase + np.pi, 2*np.pi, ) - np.pi
    return out

def phase_estimation( data_in, vardata_in, verbose=False):
    """Estimates pulse rotation angle from a sequence of P^k experiments, where
    k is of the form 2^n. Uses the modified phase estimation algorithm from
    Kimmel et al, quant-ph/1502.02677 (2015). Every experiment i doubled.
    vardata should be the variance of the mean"""

    #average together pairs of data points
    avgdata = (data_in[0::2] + data_in[1::2])/2

    # normalize data using the first two pulses to calibrate the "meter"
    data = 1 + 2*(avgdata[2:] - avgdata[0]) / (avgdata[0] - avgdata[1])
    zdata = data[0::2]
    xdata = data[1::2]

    # similar scaling with variances
    vardata = (vardata_in[0::2] + vardata_in[1::2])/2
    vardata = vardata[2:] * 2 / abs(avgdata[0] - avgdata[1])**2
    zvar = vardata[0::2]
    xvar = vardata[1::2]

    phases = np.arctan2(xdata, zdata)
    distances = np.sqrt(xdata**2 + zdata**2)

    curGuess = phases[0]
    phase = curGuess
    sigma = np.pi

    if verbose == True:
        print('Current Guess: %f'%(curGuess))

    for k in range(1,len(phases)):

        if verbose == True:
            print('k: %d'%(k))

        # Each step of phase estimation needs to assign the measured phase to
        # the correct half circle. We will conservatively require that the
        # (x,z) tuple is long enough that we can assign it to the correct
        # quadrant of the circle with 2σ confidence

        if distances[k] < 2*np.sqrt(xvar[k] + zvar[k]):
            logger.info('Phase estimation terminated at %dth pulse because the (x,z) vector is too short'%(k))
            break

        lowerBound = restrict(curGuess - np.pi/2**(k))
        upperBound = restrict(curGuess + np.pi/2**(k))
        possiblesTest = [ restrict((phases[k] + 2*n*np.pi)/2**(k)) for n in range(0,2**(k)+1)]

        if verbose == True:
            logger.info('Lower Bound: %f'%lowerBound)
            logger.info('Upper Bound: %f'%upperBound)

        possibles=[]
        for p in possiblesTest:
            # NOTE: previous code did not handle upperbound == lowerBound
            if lowerBound >= upperBound:
                satisfiesLB = p > lowerBound or p < 0.
                satisfiesUP = p < upperBound or p > 0.
            else:
                satisfiesLB = p > lowerBound
                satisfiesUP = p < upperBound

            if satisfiesLB == True and satisfiesUP == True:
                possibles.append(p)

        curGuess = possibles[0]
        if verbose == True:
            logger.info('Current Guess: %f'%(curGuess))

        phase = curGuess
        sigma = np.maximum(np.abs(restrict(curGuess - lowerBound)), np.abs(restrict(curGuess - upperBound)))

    return phase, sigma

def phase_to_amplitude(phase, sigma, amp, target, epsilon=1e-2):
    # correct for some errors related to 2pi uncertainties
    if np.sign(phase) != np.sign(amp):
        phase += np.sign(amp)*2*np.pi
    angle_error = phase - target;
    logger.info('Angle error: %.4f'%angle_error);

    amp_target = target/phase * amp
    amp_error = amp - amp_target
    logger.info('Set amplitude: %.4f\n'%amp)
    logger.info('Amplitude error: %.4f\n'%amp_error)

    amp = amp_target
    done_flag = 0

    # check for stopping condition
    phase_error = phase - target
    if np.abs(phase_error) < epsilon or np.abs(phase_error/sigma) < 1:
        if np.abs(phase_error) < epsilon:
            logger.info('Reached target rotation angle accuracy');
        elif abs(phase_error/sigma) < 1:
            logger.info('Reached phase uncertainty limit');
        done_flag = 1

    if amp > 1.0 or amp < epsilon:
        logger.warning(f"Phase estimation returned an unreasonable amplitude setting {amp}. Aborting.")
        done_flag = -1

    return amp, done_flag, phase_error

def quick_norm_data(data): #TODO: generalize as in Qlab.jl
    """Rescale data assuming 2 calibrations / single qubit state at the end of the sequence"""
    data = 2*(data-np.mean(data[-4:-2]))/(np.mean(data[-4:-2])-np.mean(data[-2:])) + 1
    data = data[:-4]
    return data
