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
from auspex.qubit.qubit_exp import QubitExperiment
from auspex.qubit import pipeline
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

import bbndb

class Calibration(object):

    def __init__(self):
        self.do_plotting = True
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
    def __init__(self, qubits, sample_name=None, output_nodes=None, stream_selectors=None, quad="real",
                    auto_rollback=True, do_plotting=True, **kwargs):
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
        try:
            self.quad_fun = {"real": np.real, "imag": np.imag, "amp": np.abs, "phase": np.angle}[quad]
        except:
            raise ValueError('Quadrature to calibrate must be one of ("real", "imag", "amp", "phase").')
        super(QubitCalibration, self).__init__()

        if sample_name:
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
        meta_file = compile_to_hardware(self.sequence(), fileName=self.filename, axis_descriptor=self.descriptor())
        exp       = CalibrationExperiment(self.qubits, self.output_nodes, self.stream_selectors, meta_file, **self.kwargs)
        if len(self.fake_data) > 0:
            for fd in self.fake_data:
                exp.set_fake_data(*fd[0], **fd[1], random_mag=0.0)
        self.exp_config(exp)
        exp.run_sweeps()

        data = {}
        var = {}
        for i, (qubit, output_buff, var_buff) in enumerate(zip(exp.qubits,
                                [exp.proxy_to_filter[on] for on in exp.output_nodes],
                                [exp.proxy_to_filter[on] for on in exp.var_buffers])):
            if not isinstance(output_buff, DataBuffer):
                raise ValueError("Could not find data buffer for calibration.")

            dataset, descriptor = output_buff.get_data()

            if self.norm_points:
                buff_data = normalize_buffer_data(dataset, descriptor, i, zero_id=self.norm_points[qubit.label][0],
                                           one_id=self.norm_points[qubit.label][1])
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
            stream_sels = [ss for ss in self.stream_selectors if ss.qubit_name == qubit.label]
            if len(stream_sels) > 1:
                raise Exception(f"More than one stream selector found for {qubit}, please explicitly define output node using output_nodes argument.")
            ds = nx.descendants(graph, stream_sels[0].hash_val)
            outputs = [graph.nodes[d]['node_obj'] for d in ds if isinstance(graph.nodes[d]['node_obj'], (bbndb.auspex.Write, bbndb.auspex.Buffer))]
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
            stream_sels = [ss for ss in self.stream_selectors if ss.qubit_name == qubit.label]
            if not any([ss.hash_val in graph for ss in stream_sels]):
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

            ancestors   = [graph.nodes[n]['node_obj'] for n in nx.ancestors(graph, output_node.hash_val)]
            stream_sels = [a for a in ancestors if isinstance(a, bbndb.auspex.StreamSelect)]
            if len(stream_sels) != 1:
                raise Exception(f"Expected to find one stream selector for {qubit}. Instead found {len(stream_sels)}")
            stream_sel = stream_sels[0]

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

        self.output_nodes = new_output_nodes
        return new_graph

    def add_cal_sweep(self, method, values):
        par = FloatParameter()
        par.assign_method(method)
        self.add_sweep(par, values)
