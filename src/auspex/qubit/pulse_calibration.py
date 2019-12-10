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
from . import pipeline
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.analysis.fits import *
from auspex.analysis.CR_fits import *
from auspex.analysis.qubit_fits import *
from auspex.analysis.helpers import normalize_buffer_data
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
from itertools import product

class Calibration(object):

    def __init__(self):
        self.uuid = str(uuid.uuid4())
        self.context = None
        self.socket = None

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
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.identity = "Auspex_Experiment".encode()
            self.socket.connect("tcp://localhost:7761")
            self.socket.send_multipart([self.uuid.encode(), json.dumps(plot_desc).encode('utf8')])

            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)

            time.sleep(1)
            evts = dict(poller.poll(5000))
            if self.socket in evts:
                try:
                    if self.socket.recv_multipart()[0] == b'ACK':
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
        finally:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()

        for p in self.plotters:
            p.start()

    def stop_plots(self):
        for p in self.plotters:
            p.stop()

    def calibrate(self):
        if self.do_plotting:
            self.plotters = self.init_plots()
            self.start_plots()

        self._calibrate()

        if self.succeeded:
            self.update_settings()
        else:
            raise Exception('Calibration failed!')

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
    def __init__(self, qubits, sample_name=None, output_nodes=None, stream_selectors=None, quad="real", auto_rollback=True, do_plotting=True, **kwargs):
        self.qubits           = qubits if isinstance(qubits, list) else [qubits]
        self.qubit            = None if isinstance(qubits, list) else qubits
        self.output_nodes     = output_nodes if isinstance(output_nodes, list) else [output_nodes]
        self.stream_selectors = stream_selectors if isinstance(stream_selectors, list) else [stream_selectors]
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
        self.fake_data        = []
        self.sample           = None
        self.metafile         = None
        try:
            self.quad_fun = {"real": np.real, "imag": np.imag, "amp": np.abs, "phase": np.angle}[quad]
        except:
            raise ValueError('Quadrature to calibrate must be one of ("real", "imag", "amp", "phase").')
        super(QubitCalibration, self).__init__()
        if not sample_name:
            sample_name = self.qubits[0].label
        if not bbndb.get_cl_session():
            raise Exception("Attempting to load Calibrations database, \
                but no database session is open! Have the ChannelLibrary and PipelineManager been created?")
        existing_samples = list(bbndb.get_cl_session().query(bbndb.calibration.Sample).filter_by(name=sample_name).all())
        if len(existing_samples) == 0:
            logger.info("Creating a new sample in the calibration database.")
            self.sample = bbndb.calibration.Sample(name=sample_name)
            bbndb.get_cl_session().add(self.sample)
        elif len(existing_samples) == 1:
            self.sample = existing_samples[0]
        else:
            raise Exception("Multiple samples found in calibration database with the same name! How?")

    def sequence(self):
        """Returns the sequence for the given calibration, must be overridden"""
        raise NotImplementedError("Must run a specific qubit calibration.")

    def set_fake_data(self, *args, **kwargs):
        self.fake_data.append((args, kwargs))

    def run_sweeps(self):
        if self.metafile:
            meta_file = self.metafile
        else:
            meta_file = compile_to_hardware(self.sequence(), fileName=self.filename, axis_descriptor=self.descriptor())
        exp       = CalibrationExperiment(self.qubits, self.output_nodes, self.stream_selectors, meta_file, **self.kwargs)
        if len(self.fake_data) > 0:
            for fd in self.fake_data:
                exp.set_fake_data(*fd[0], **fd[1], random_mag=0.0)
        self.exp_config(exp)
        exp.run_sweeps()

        data = {}
        var = {}

        #sort nodes by qubit name to match data with metadata when normalizing
        qubit_indices = {q.label: idx for idx, q in enumerate(exp.qubits)}
        exp.output_nodes.sort(key=lambda x: qubit_indices[x.qubit_name])
        exp.var_buffers.sort(key=lambda x: qubit_indices[x.qubit_name])

        for i, (qubit, output_buff, var_buff) in enumerate(zip(exp.qubits,
                                [exp.proxy_to_filter[on] for on in exp.output_nodes],
                                [exp.proxy_to_filter[on] for on in exp.var_buffers])):
            if not isinstance(output_buff, DataBuffer):
                raise ValueError("Could not find data buffer for calibration.")

            dataset, descriptor = output_buff.get_data()
            dataset = self.quad_fun(dataset)

            if self.norm_points:
                buff_data = normalize_buffer_data(dataset, descriptor, i, zero_id=self.norm_points[qubit.label][0],
                                           one_id=self.norm_points[qubit.label][1])
            else:
                buff_data = dataset

            data[qubit.label] = buff_data

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

    def __init__(self, qubits, output_nodes, stream_selectors, *args, **kwargs):
        self.qubits = qubits
        self.output_nodes = output_nodes
        self.input_selectors = stream_selectors # name collision otherwise
        self.var_buffers = []
        if 'disable_plotters' in kwargs:
            self.disable_plotters = kwargs.pop('disable_plotters')
        else:
            self.disable_plotters = False
        super(CalibrationExperiment, self).__init__(*args, **kwargs)

    def guess_output_nodes(self, graph):
        output_nodes = []
        qubit_labels = [q.label for q in self.qubits]
        for qubit in self.qubits:
            stream_sels = [ss for ss in self.stream_selectors if qubit.label in ss.label.split("-")]
            if len(stream_sels) > 1:
                raise Exception(f"More than one stream selector found for {qubit}, please explicitly define output node using output_nodes argument.")
            ds = nx.descendants(graph, stream_sels[0].hash_val)
            outputs = [graph.nodes[d]['node_obj'] for d in ds if isinstance(graph.nodes[d]['node_obj'], (bbndb.auspex.Write, bbndb.auspex.Buffer)) and graph.nodes[d]['node_obj'].qubit_name == qubit.label]
            if len(outputs) > 1:
                raise Exception(f"More than one output node found for {qubit}, please explicitly define output node using output_nodes argument.")
            output_nodes.append(outputs[0])

        return output_nodes

    def modify_graph(self, graph):
        """Change the graph as needed. By default we changes all writers to buffers"""
        if None in self.output_nodes:
            self.output_nodes = self.guess_output_nodes(graph)

        for output_node in self.output_nodes:
            if output_node.hash_val not in graph:
                raise ValueError(f"Could not find specified output node {output_node} in graph.")

        for qubit in self.qubits:
            stream_sels = [ss for ss in self.stream_selectors if qubit.label in ss.label.split("-")]
            if not any([ss.hash_val in graph for ss in stream_sels]):
                raise ValueError(f"Could not find specified qubit {qubit} in graph.")

        mapping = {}
        self.output_connectors = {}
        for i in range(len(self.output_nodes)):
            output_node = self.output_nodes[i]
            if isinstance(output_node, bbndb.auspex.Write):
                # Change the output node to a buffer
                mapping[output_node] = bbndb.auspex.Buffer(label=output_node.label, qubit_name=output_node.qubit_name)

        # Disable any paths not involving the buffer
        new_graph = nx.DiGraph()
        new_output_nodes = []
        new_stream_selectors = []
        connector_by_sel = {}
        for output_node, qubit in zip(self.output_nodes, self.qubits):
            new_output = mapping[output_node]
            new_output_nodes.append(new_output)

            ancestors   = [graph.nodes[n]['node_obj'] for n in nx.ancestors(graph, output_node.hash_val)]
            stream_sels = [a for a in ancestors if isinstance(a, bbndb.auspex.StreamSelect)]
            if len(stream_sels) != 1:
                raise Exception(f"Expected to find one stream selector for {qubit}. Instead found {len(stream_sels)}")
            stream_sel = stream_sels[0]
            new_stream_selectors.append(stream_sel)
            connector_by_sel[stream_sel] = self.connector_by_sel[stream_sel]

            old_path  = nx.shortest_path(graph, stream_sel.hash_val, output_node.hash_val)
            path      = old_path[:-1] + [new_output.hash_val]
            nx.add_path(new_graph, path)
            for n in old_path[:-1]:
                new_graph.nodes[n]['node_obj'] = graph.nodes[n]['node_obj']
            new_graph.nodes[new_output.hash_val]['node_obj'] = mapping[output_node]

            # Fix connectors
            for i in range(len(path)-1):
                new_graph[path[i]][path[i+1]]['connector_in']  = graph[old_path[i]][old_path[i+1]]['connector_in']
                new_graph[path[i]][path[i+1]]['connector_out'] = graph[old_path[i]][old_path[i+1]]['connector_out']

            if not isinstance(new_graph.nodes(data=True)[path[-2]]['node_obj'], bbndb.auspex.Average):
                raise Exception("There is no averager in line.")
            else:
                vb = bbndb.auspex.Buffer(label=f"{output_node.label}-VarBuffer", qubit_name=output_node.qubit_name)
                self.var_buffers.append(vb)
                new_graph.add_node(vb.hash_val, node_obj=vb)
                new_graph.add_edge(path[-2], vb.hash_val, node_obj=vb, connector_in="sink", connector_out="final_variance")
            # maintain standard plots
            if not self.disable_plotters:
                plot_nodes = [output_node for output_node in nx.descendants(graph, path[-2]) if isinstance(graph.nodes[output_node]['node_obj'], bbndb.auspex.Display)]
                for plot_node in plot_nodes:
                    plot_path = nx.shortest_path(graph, path[-2], plot_node)
                    new_graph = nx.compose(new_graph, graph.subgraph(plot_path))

        # Update nodes and connectors
        self.output_nodes = new_output_nodes
        self.stream_selectors = new_stream_selectors
        self.connector_by_sel = connector_by_sel

        for ss in new_stream_selectors:
            self.output_connectors[self.connector_by_sel[ss].name] = self.connector_by_sel[ss]
        for ch in list(self.chan_to_oc):
            if self.chan_to_oc[ch] not in self.output_connectors.values():
                self.chan_to_oc.pop(ch)
                self.chan_to_dig.pop(ch)
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
        I_fit = RabiAmpFit(self.amps, data[N//2:])
        Q_fit = RabiAmpFit(self.amps, data[:N//2])
        #Arbitary extra division by two so that it doesn't push the offset too far.
        self.pi_amp = I_fit.pi_amp
        self.pi2_amp = I_fit.pi_amp/2.0
        self.i_offset = I_fit.fit_params["phi"]*self.amp2offset
        self.q_offset = Q_fit.fit_params["phi"]*self.amp2offset
        logger.info("Found X180 amplitude: {}".format(self.pi_amp))
        logger.info("Shifting I offset by: {}".format(self.i_offset))
        logger.info("Shifting Q offset by: {}".format(self.q_offset))
        finer_amps = np.linspace(np.min(self.amps), np.max(self.amps), 4*len(self.amps))
        if self.do_plotting:
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
            bbndb.get_cl_session().add_all([c1, c2])
            bbndb.get_cl_session().commit()

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
        if self.first_ramsey:
            rcvr = self.qubit.measure_chan.receiver_chan.receiver
            self.source_proxy = self.qubit.phys_chan.generator # DB object
            self.qubit_source = exp._instruments[self.source_proxy.label] # auspex instrument
            if self.set_source:
                self.orig_freq = self.source_proxy.frequency + self.qubit.frequency # real qubit freq.
                self.source_proxy.frequency += self.added_detuning
            else:
                self.orig_freq = self.qubit.frequency
                self.qubit.frequency = round(self.orig_freq+self.added_detuning,10)

    def _calibrate(self):
        self.first_ramsey = True

        data, _ = self.run_sweeps()
        try:
            ramsey_fit = RamseyFit(self.delays, data, two_freqs=self.two_freqs, AIC=self.AIC)
            fit_freqs = ramsey_fit.fit_params["f"]
            fit_err = ramsey_fit.fit_errors["f"]
        except Exception as e:
            raise Exception(f"Exception {e} while fitting in {self}")

        # Plot the results
        if self.do_plotting:
            self.plot["Data 1"] = (self.delays, data)
            finer_delays = np.linspace(np.min(self.delays), np.max(self.delays), 4*len(self.delays))
            self.plot["Fit 1"] = (finer_delays, ramsey_fit.model(finer_delays))

        #TODO: set conditions for success
        fit_freq_A = np.mean(fit_freqs) #the fit result can be one or two frequencies
        fit_err_A = np.sum(fit_err)
        if self.set_source:
            self.source_proxy.frequency = round(self.orig_freq - self.qubit.frequency + self.added_detuning + fit_freq_A/2, 10)
            #self.qubit_source.frequency = self.source_proxy.frequency
        else:
            self.qubit.frequency = round(self.orig_freq + self.added_detuning + fit_freq_A/2, 10)

        self.first_ramsey = False

        data, _ = self.run_sweeps()

        try:
            ramsey_fit = RamseyFit(self.delays, data, two_freqs=self.two_freqs, AIC=self.AIC)
            fit_freqs = ramsey_fit.fit_params["f"]
            fit_err = ramsey_fit.fit_errors["f"]
        except Exception as e:
            if self.set_source:
                self.source_proxy.frequency = self.orig_freq
                self.qubit_source.frequency = self.orig_freq
            else:
                self.qubit.frequency = self.orig_freq
            raise Exception(f"Exception {e} while fitting in {self}")

        # Plot the results
        if self.do_plotting:
            self.plot["Data 2"] = (self.delays, data)
            self.plot["Fit 2"]  = (finer_delays, ramsey_fit.model(finer_delays))

        fit_freq_B = np.mean(fit_freqs)
        fit_err_B = np.sum(fit_err)
        if fit_freq_B < fit_freq_A:
            self.fit_freq = round(self.orig_freq + self.added_detuning + 0.5*(fit_freq_A + 0.5*fit_freq_A + fit_freq_B), 10)
        else:
            self.fit_freq = round(self.orig_freq + self.added_detuning - 0.5*(fit_freq_A - 0.5*fit_freq_A + fit_freq_B), 10)
        self.fit_err = fit_err_A + fit_err_B
        logger.info(f"Found qubit frequency {round(self.fit_freq/1e9,9)} GHz")
        self.succeeded = True #TODO: add bounds

    def update_settings(self):
        if self.set_source:
            self.source_proxy.frequency = float(round(self.fit_freq - self.qubit.frequency))
        elif self.source_proxy is not None:
            self.qubit.frequency = float(round(self.fit_freq - self.source_proxy.frequency))
        else:
            self.qubit.frequency = float(round(self.fit_freq))

        for edge in self.qubit.edge_target:
            if edge.phys_chan.generator is not None:
                edge_source = edge.phys_chan.generator
                if self.set_source:
                    edge_source.frequency = self.source_proxy.frequency + self.qubit.frequency - edge.frequency
                else:
                    edge.frequency = self.source_proxy.frequency + self.qubit.frequency - edge_source.frequency
            else:
                edge.frequency = self.qubit.frequency

        if self.sample:
            frequency = round(self.fit_freq,9)
            frequency_error = round(self.fit_err,9)
            c = bbndb.calibration.Calibration(value=frequency, uncertainty=frequency_error, sample=self.sample, name="Ramsey")
            c.date = datetime.datetime.now()
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

class PhaseEstimation(QubitCalibration):

    amp2offset = 0.5

    def __init__(self, channel, num_pulses= 1, amplitude= 0.1, direction = 'X',
                    target=np.pi/2, epsilon=1e-2, max_iter=5, **kwargs):
        #for now, only do one qubit at a time
        self.num_pulses = num_pulses
        self.amplitude = amplitude
        self.direction = direction

        self.target = target
        self.epsilon = epsilon
        self.max_iter = max_iter

        super(PhaseEstimation, self).__init__(channel, **kwargs)

        self.filename = 'PhaseCal/PhaseCal'

    def sequence(self):
        # Determine whether it is a single- or a two-qubit pulse calibration
        if isinstance(self.qubit, bbndb.qgl.Edge): # slight misnomer...
            qubit = self.qubit.target
            cal_pulse = [ZX90_CR(self.qubit.source, self.qubit.target, amp=self.amplitude)]
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

            if self.do_plotting:
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
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

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
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

class CRAmpCalibration_PhEst(PhaseEstimation):
    def __init__(self, edge, num_pulses= 5, **kwargs):
        super(CRAmpCalibration_PhEst, self).__init__(edge, num_pulses = num_pulses, amplitude=edge.pulse_params['amp'], direction='X', target=np.pi/2, epsilon=1e-2, max_iter=5,**kwargs)
        self.qubits = [edge.target]

    def update_settings(self):
        self.qubit.pulse_params['amp'] = round(self.amplitude, 5)

        if self.sample:
            c = bbndb.calibration.Calibration(value=self.amplitude, sample=self.sample, name="CRamp", category="PhaseEstimation")
            c.date = datetime.datetime.now()
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

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
        label = rcvr.label
        if rcvr.transceiver is not None:
            label = rcvr.transceiver.label
        exp._instruments[label].exp_step = self.step #where from?

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
            if self.do_plotting:
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
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

class CustomCalibration(QubitCalibration):
    def __init__(self, qubits, metafile = None, fit_name = None, fit_param = [], set_param = None, **kwargs):
        if not metafile or not fit_name or not fit_param:
            raise Exception("Please specify experiment metafile, fit function, and desired fit paramter.") #currently save single param.
        try:
            with open(metafile, 'r') as FID:
                self.meta_info = json.load(FID)
        except:
            raise Exception(f"Could note process meta info from file {meta_file}")
        self.fit_name = fit_name
        if not isinstance(fit_param, list):
            fit_param = [fit_param]
        self.fit_param = fit_param
        self.set_param = set_param
        super().__init__(qubits, **kwargs)
        self.metafile = metafile
        self.norm_points = {self.qubits[0].label: (0, 1)} #TODO: generalize

    def _calibrate(self):
        data, _ = self.run_sweeps()  # need to get descriptor
        try:
            self.fit_result = eval(self.fit_name)(np.array(self.meta_info["axis_descriptor"][0]['points']), data)
            self.succeeded = True
            if self.set_param:  # optional set parameter
                self.set_param = self.fit_result.fit_params[self.fit_param]
        except:
            logger.warning(f"{self.fit_name} fit failed.")

    def update_settings(self):
        if self.sample: # make a separate Calibration entry for each fit paramter
            curr_time = datetime.datetime.now()
            for fit_param in self.fit_param:
                c = bbndb.calibration.Calibration(value=self.fit_result.fit_params[fit_param], uncertainty=self.fit_result.fit_errors[fit_param],
                sample=self.sample, name=fit_param, category=self.fit_name)
                c.date = curr_time
                bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

'''Two-qubit gate calibrations'''
class CRCalibration(QubitCalibration):
    """Template for CR calibrations. Currently available steps: length, phase, amplitude

    Args:
        edge: Edge in the channel library defining the connection between control and target qubit
        lengths (array): CR pulse length(s). Longer than 1 for CRLenCalibration
        phases (array): CR pulse phase(s). Longer than 1 for CRPhaseCalibration
        amps (array): CR pulse amp(s). Longer than 1 for CRAmpCalibration
        rise_fall (float): length of rise/fall of CR pulses
        meas_qubits (list): specifies a subset of qubits to be measured (both by default)
    """
    def __init__(self,
                 edge,
                 lengths = np.linspace(20, 1020, 21)*1e-9,
                 phases = [0],
                 amps = [0.8],
                 rise_fall = 40e-9,
                 meas_qubits = None,
                 **kwargs):
        self.lengths   = lengths
        self.phases    = phases
        self.amps      = amps
        self.rise_fall = rise_fall
        self.filename  = 'CR/CR'

        self.edge      = edge
        qubits = meas_qubits if meas_qubits else [edge.source, edge.target]
        super().__init__(qubits, **kwargs)

    def init_plots(self):
        plot = ManualPlotter("CR"+str.lower(self.cal_type.name)+"Fit", x_label=str.lower(self.cal_type.name), y_label='$<Z_{'+self.edge.target.label+'}>$', y_lim=(-1.02,1.02))
        plot.add_data_trace("Data 0", {'color': 'C1'})
        plot.add_fit_trace("Fit 0", {'color': 'C1'})
        plot.add_data_trace("Data 1", {'color': 'C2'})
        plot.add_fit_trace("Fit 1", {'color': 'C2'})

        self.plot = plot
        return [plot]

    def _calibrate(self):
        # run and load normalized data
        qt = self.edge.target
        qs = self.edge.source
        self.qubit = qt

        self.norm_points = {qs.label: (0, 1), qt.label: (0, 1)}
        data, _ =  self.run_sweeps()

        data_t = data[qt.label] if isinstance(data, dict) else data
        # fit
        self.opt_par, all_params_0, all_params_1 = fit_CR([self.lengths, self.phases, self.amps], data_t, self.cal_type)
        # plot the result
        xaxis = self.lengths if self.cal_type==CR_cal_type.LENGTH else self.phases if self.cal_type==CR_cal_type.PHASE else self.amps
        finer_xaxis = np.linspace(np.min(xaxis), np.max(xaxis), 4*len(xaxis))

        if self.do_plotting:
            self.plot["Data 0"] = (xaxis,       data_t[:len(data_t)//2])
            self.plot["Fit 0"] =  (finer_xaxis, np.polyval(all_params_0, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, **all_params_0))
            self.plot["Data 1"] = (xaxis,       data_t[len(data_t)//2:])
            self.plot["Fit 1"] =  (finer_xaxis, np.polyval(all_params_1, finer_xaxis) if self.cal_type == CR_cal_type.AMP else sinf(finer_xaxis, **all_params_1))

        # Optimal parameter within range of original data!
        if self.opt_par > np.min(xaxis) and self.opt_par < np.max(xaxis):
            self.succeeded = True

    def update_settings(self):
        print("updating settings...")
        self.edge.pulse_params[str.lower(self.cal_type.name)] = float(self.opt_par)
        super(CRCalibration, self).update_settings()
        if self.sample:
            c = bbndb.calibration.Calibration(value=float(self.opt_par), sample=self.sample, name="CR"+str.lower(self.cal_type.name))
            c.date = datetime.datetime.now()
            bbndb.get_cl_session().add(c)
            bbndb.get_cl_session().commit()

class CRLenCalibration(CRCalibration):
    cal_type = CR_cal_type.LENGTH

    def __init__(self, edge, lengths=np.linspace(20, 1020, 21)*1e-9, phase=0, amp=0.8, rise_fall=40e-9, meas_qubits=None, **kwargs):
        super().__init__(edge, lengths=lengths, phases=[phase], amps=[amp], rise_fall=rise_fall, meas_qubits = meas_qubits, **kwargs)

    def sequence(self):
        qc = self.edge.source
        qt = self.edge.target
        measBlock = reduce(operator.mul, [MEAS(q) for q in self.qubits])
        seqs = [[Id(qc)] + echoCR(qc, qt, length=l, phase = self.phases[0], amp=self.amps[0], riseFall=self.rise_fall).seq + [Id(qc), measBlock] for l in self.lengths]
        seqs += [[X(qc)] + echoCR(qc, qt, length=l, phase= self.phases[0], amp=self.amps[0], riseFall=self.rise_fall).seq + [X(qc), measBlock] for l in self.lengths]
        seqs += create_cal_seqs(self.qubits, 2)
        return seqs

    def descriptor(self):
         return [
            delay_descriptor(np.concatenate((self.lengths, self.lengths))),
            cal_descriptor(tuple(self.qubits), 2)
        ]


class CRPhaseCalibration(CRCalibration):
    cal_type = CR_cal_type.PHASE

    def __init__(self, edge, length=None, phases=np.linspace(0,2*np.pi,21), amp=0.8, rise_fall=40e-9, **kwargs):
        if not length:
            length = edge.pulse_params['length']
        super().__init__(edge, lengths=[length], phases=phases, amps=[amp], rise_fall=rise_fall, **kwargs)

    def sequence(self):
        qc = self.edge.source
        qt = self.edge.target
        measBlock = reduce(operator.mul, [MEAS(q) for q in self.qubits])
        seqs = [[Id(qc)] + echoCR(qc, qt, length=self.lengths[0], phase=ph, amp=self.amps[0], riseFall=self.rise_fall).seq + [X90(qt)*Id(qc), measBlock] for ph in self.phases]
        seqs += [[X(qc)] + echoCR(qc, qt, length=self.lengths[0], phase=ph, amp=self.amps[0], riseFall=self.rise_fall).seq + [X90(qt)*X(qc), measBlock] for ph in self.phases]
        seqs += create_cal_seqs(self.qubits, 2)
        return seqs

    def descriptor(self):
        return [
            {
                'name': 'phase',
                'unit': 'radians',
                'points': list(self.phases)+list(self.phases),
                'partition': 1
            },
            cal_descriptor(tuple(self.qubits), 2)
        ]

class CRAmpCalibration(CRCalibration):
    cal_type = CR_cal_type.AMP

    def __init__(self, edge, amp_range = 0.4, amp = 0.8, rise_fall = 40e-9, num_CR = 1, **kwargs):
        self.num_CR = num_CR
        length = edge.pulse_params['length']
        phase  = edge.pulse_params['phase']
        if num_CR % 2 == 0:
            logger.error('The number of ZX90 must be odd')
        amps = np.linspace((1-amp_range/2)*amp, (1+amp_range/2)*amp, 21)
        super().__init__(edge, lengths=[length], phases=[phase], amps=amps, rise_fall=rise_fall, **kwargs)

    def sequence(self):
        qc = self.edge.source
        qt = self.edge.target
        measBlock = reduce(operator.mul, [MEAS(q) for q in self.qubits])
        seqs = [[Id(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths[0], phase=self.phases[0], amp=a, riseFall=self.rise_fall).seq + [Id(qc), measBlock]
        for a in self.amps]+ [[X(qc)] + self.num_CR*echoCR(qc, qt, length=self.lengths[0], phase= self.phases[0], amp=a, riseFall=self.rise_fall).seq + [X(qc), measBlock]
        for a in self.amps] + create_cal_seqs(self.qubits, 2)
        return seqs

    def descriptor(self):
        return [{'name': 'amplitude',
                 'unit': None,
                 'points': list(self.amps)+list(self.amps),
                 'partition': 1
                },
                cal_descriptor(tuple(self.qubits), 2)]

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
            logger.info('Reached target rotation angle accuracy. Set amplitude: %.4f\n'%amp)
        elif abs(phase_error/sigma) < 1:
            logger.info('Reached phase uncertainty limit. Set amplitude: %.4f\n'%amp)
        done_flag = 1

    if amp > 1.0 or amp < epsilon:
        logger.warning(f"Phase estimation returned an unreasonable amplitude setting {amp}. Aborting.")
        done_flag = -1

    return amp, done_flag, phase_error

def quick_norm_data(data): #TODO: generalize as in Qlab.jl
    if np.any(np.iscomplex(data)):
        logger.warning("quick_norm_data does not support complex data!")
    """Rescale data assuming 2 calibrations / single qubit state at the end of the sequence"""
    data = 2*(data-np.mean(data[-4:-2]))/(np.mean(data[-4:-2])-np.mean(data[-2:])) + 1
    data = data[:-4]
    return data

class CLEARCalibration(QubitCalibration):
    '''Calibration of cavity reset pulse.

    Args:
        kappa: Cavity linewith (angular frequency: 1/s).
        chi: Half of the dispersive shift (anguler frequency: 1/s).
        t_empty: Time for active depletion (s).
        alpha: Scaling factor.
        T1factor: decay due to T1 between end of measurement and start of Ramsey.
        T2: Measured T2*
        nsteps: number of calibration steps
        ramsey_delays: List of times to use for Ramsey experiment.
        ramsey_freq: Ramsey offset frequency.
        meas_delay: Delay after end of measurement pulse
        preramsey_delay: Delay before start of Ramsey sequence.
        eps1: 1st CLEAR parameter. if set to `None` will use theory values as default for eps1 and eps2.
        eps2: 2nd CLEAR parameter.
        cal_steps (bool, bool, bool): Calibration steps to execute. Currently, the first step sweeps eps1,
        the second eps2, and the third eps1 again in a smaller range.
    '''

    def __init__(self, qubit, kappa = 2*np.pi*2e6, chi = -2*np.pi*1e6, t_empty = 400e-9,
                ramsey_delays=np.linspace(0.0, 2.0, 51)*1e-6, ramsey_freq = 2e6, meas_delay = 0,
                preramsey_delay=0, alpha = 1, T1factor = 1, T2 = 30e-6, nsteps = 5,
                eps1 = None, eps2 = None, cal_steps = (1,1,1), **kwargs):

        self.kappa = kappa
        self.chi = chi
        self.ramsey_delays = ramsey_delays
        self.ramsey_freq = ramsey_freq
        self.meas_delay = meas_delay
        self.preramsey_delay = preramsey_delay
        self.tau = t_empty/2.0
        self.alpha = alpha
        self.T1factor = T1factor
        self.T2 = T2
        self.nsteps = nsteps

        #use theory values as defaults
        if eps1 == None or eps2 == None:
            self.eps1 = ((1 - 2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
                        /(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2)))
            self.eps2 = 1/(1+np.exp(kappa*t_empty/2)-2*np.exp(kappa*t_empty/4)*np.cos(chi*t_empty/2))
            logger.info(f' Using theoretical CLEAR amplitudes: {self.eps1} (eps1), {self.eps2} (eps2)')
        else:
            self.eps1 = eps1
            self.eps2 = eps2

        self.cal_steps = cal_steps
        self.seq_params = {}

        kwargs['disable_plotters'] = True
        super().__init__(qubit, **kwargs)
        self.filename = 'CLEAR/CLEAR'

    def descriptor(self):
        return [delay_descriptor(self.ramsey_delays), cal_descriptor(tuple(self.qubits), 2)]

    def sequence(self):
        if self.seq_params['state']:
            prep = X(self.qubit)
        else:
            prep = Id(self.qubit)

        amp1 = self.alpha * self.seq_params['eps1']
        amp2 = self.alpha * self.seq_params['eps2']


        clear_meas = MEASCLEAR(self.qubit, amp1=amp1, amp2=amp2, step_length=self.seq_params['tau'])
        seqs = [[prep, clear_meas, Id(self.qubit, self.preramsey_delay), X90(self.qubit), Id(self.qubit,d),
                    U90(self.qubit,phase = 2*pi*self.ramsey_freq*d), Id(self.qubit, self.meas_delay), MEAS(self.qubit)]
                        for d in self.ramsey_delays]

        seqs += create_cal_seqs((self.qubit,), 2, delay = self.meas_delay)

        return seqs

    def init_plots(self):
        plot_ramsey = ManualPlotter("CLEAR Ramsey", x_label='Time (us)', y_label='<P(1)>', y_lim=(-0.02,1.02))
        plot_clear = ManualPlotter("CLEAR Calibration", x_label='epsilon', y_label='Residual Photons')

        plot_ramsey.add_data_trace("Data - 0 State", {'color':'C1'})
        plot_ramsey.add_fit_trace("Fit - 0 State", {'color':'C1'})
        plot_ramsey.add_data_trace("Data - 1 State", {'color':'C2'})
        plot_ramsey.add_fit_trace("Fit - 1 State", {'color':'C2'})

        color = 1
        for sweep_num, state in product(range(sum(self.cal_steps)), [0,1]):
            plot_clear.add_data_trace(f"Sweep {sweep_num}, State {state}", {"color": f'C{color}'})
            plot_clear.add_fit_trace(f"Fit Sweep {sweep_num}, State {state}", {"color": f'C{color}'})
            color += 1

        self.plot_ramsey = plot_ramsey
        self.plot_clear = plot_clear

        return [plot_ramsey, plot_clear]

    def _calibrate_one_point(self):
        n0_0 = 0.0
        n0_1 = 0.0
        for state in [0,1]:
            self.seq_params['state'] = state
            data, _ = self.run_sweeps()
            norm_data = quick_norm_data(data)

            # if self.fit_ramsey_freq is None:
            #     fit = RamseyFit(self.ramsey_delays, norm_data)
            #     self.fit_ramsey_freq = fit.fit_params["f"]
            #     logger.info(f"Found Ramsey Frequency of :{self.fit_ramsey_freq/1e3:.3f} kHz.")

            state_data = 0.5*(1 - norm_data) #renormalize data to match convention in CLEAR paper from IBM

            fit = PhotonNumberFit(self.ramsey_delays, state_data, self.T2, self.ramsey_freq*2*np.pi, self.kappa,
                                self.chi, self.T1factor, state)

            self.plot_ramsey[f"Data - {state} State"] = (self.ramsey_delays, state_data)
            self.plot_ramsey[f"Fit - {state} State"] = (self.ramsey_delays, fit.model(self.ramsey_delays))

            if state == 1:
                n0_1 = fit.fit_params["n0"]
            else:
                n0_0 = fit.fit_params["n0"]

        return n0_0, n0_1

    def _calibrate(self):

        #self.fit_ramsey_freq = None
        self.seq_params["tau"] = self.tau
        min_amps = [0, 0, 0.5*self.eps1]
        max_amps = [2*self.eps1, 2*self.eps2, 1.5*self.eps1]
        ind_eff = 0
        for ind,step in enumerate(self.cal_steps):
            if step:
                if ind==1:
                    self.seq_params['eps1'] = self.eps1
                else:
                    self.seq_params['eps2'] = self.eps2
                xpoints = np.linspace(min_amps[ind], max_amps[ind], self.nsteps)
                n0vec = np.zeros(self.nsteps)
                n1vec = np.zeros(self.nsteps)
                for k, xp in enumerate(xpoints):
                    if ind == 1:
                        self.seq_params['eps2'] = xp
                    else:
                        self.seq_params['eps1'] = xp
                    n0vec[k], n1vec[k] = self._calibrate_one_point()
                    self.plot_clear[f'Sweep {ind_eff}, State 0'] = (xpoints, n0vec)
                    self.plot_clear[f'Sweep {ind_eff}, State 1'] = (xpoints, n1vec)

                fit0 = QuadraticFit(xpoints, n0vec)
                fit1 = QuadraticFit(xpoints, n1vec)
                finer_xpoints = np.linspace(np.min(xpoints), np.max(xpoints), 4*len(xpoints))
                self.plot_clear[f'Fit Sweep {ind_eff}, State 0'] = (finer_xpoints, fit0.model(finer_xpoints))
                self.plot_clear[f'Fit Sweep {ind_eff}, State 1'] = (finer_xpoints, fit1.model(finer_xpoints))
                best_guess = 0.5*(fit0.fit_params["x0"]+ fit1.fit_params["x0"])
                logger.info(f"Found best epsilon1 = {best_guess:.6f}")
                if ind == 1:
                    self.eps2 = best_guess
                else:
                    self.eps1 = best_guess
                ind_eff+=1

        self.eps1 = round(float(self.eps1), 5)
        self.eps2 = round(float(self.eps2), 5)

        logger.info(f"Found best CLEAR pulse parameters: eps1 = {self.eps1}, eps2 = {self.eps2}")

        self.succeeded = True #TODO: add bounds

    def update_settings(self):
        self.qubit.measure_chan.pulse_params['amp1'] = self.eps1
        self.qubit.measure_chan.pulse_params['amp2'] = self.eps2
        self.qubit.measure_chan.pulse_params['step_length'] = round(float(self.tau), 9)
