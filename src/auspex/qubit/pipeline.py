import json
import sys
import os
import importlib
import pkgutil
import inspect
import re

import base64
import datetime
import copy

import numpy as np
import networkx as nx

import auspex.config as config
import auspex.instruments
import auspex.filters
from auspex.stream import DataAxis
import bbndb
from auspex.log import logger

pipelineMgr = None

class PipelineManager(object):
    """Create and run Qubit Experiments."""
    def __init__(self):
        global pipelineMgr

        self.pipeline      = None
        self.qubit_proxies = {}
        self.meas_graph    = None

        if bbndb.session:
            self.session = bbndb.session
        else:
            raise Exception("Auspex expects db to be loaded already by QGL")

        pipelineMgr = self

    def create_default_pipeline(self, qubits=None, buffers=False):
        """Look at the QGL channel library and create our pipeline from the current
        qubits."""
        cdb = self.session.query(bbndb.qgl.ChannelDatabase).filter_by(label="working").first()
        if not cdb:
            raise ValueError("Could not find working channel library.")
        
        if not qubits:
            measurements = [c for c in cdb.channels if isinstance(c, bbndb.qgl.Measurement)]
            meas_labels  = [m.label for m in measurements]
            qubits       = [c for c in cdb.channels if "M-"+c.label in meas_labels]
        else:
            meas_labels = ["M-"+q.label for q in qubits]
            measurements = [c for c in cdb.channels if c.label in meas_labels]
        self.qubits = qubits
        self.qubit_proxies = {q.label: bbndb.auspex.QubitProxy(self, q.label) for q in qubits}

        # Build a mapping of qubits to receivers, construct qubit proxies
        receiver_chans_by_qubit = {}
        available_streams_by_qubit = {}
        for m in measurements:
            q = [c for c in cdb.channels if c.label==m.label[2:]][0]
            receiver_chans_by_qubit[q] = m.receiver_chan
            available_streams_by_qubit[q] = m.receiver_chan.receiver.stream_types

        for q, r in receiver_chans_by_qubit.items():
            qp = self.qubit_proxies[q.label]
            qp.available_streams = [st.strip() for st in r.receiver.stream_types.split(",")]
            qp.stream_type = qp.available_streams[-1]

        # generate the pipeline automatically
        self.meas_graph = nx.DiGraph()
        for qp in self.qubit_proxies.values():
            qp.create_default_pipeline(buffers=buffers)

        for el in self.meas_graph.nodes():
            self.session.add(el)
        self.session.commit()
        self.save_pipeline("working")

    def qubit(self, qubit_name):
        return self.qubit_proxies[qubit_name]

    def ls(self):
        i = 0
        for name, time in self.session.query(bbndb.auspex.Connection.pipeline_name, bbndb.auspex.Connection.time).distinct().all():
            print(f"[{i}] {time} -> {name}")
            i += 1

    def save_pipeline(self, name):
        now = datetime.datetime.now()
        cs = [bbndb.auspex.Connection(pipeline_name=name, node1=n1, node2=n2, time=now) for n1, n2 in self.meas_graph.edges()]
        for c in cs:
            self.session.add(c)

    def load_pipeline(self, pipeline_name):
        cs = select(c for c in bbndb.auspex.Connection if c.pipeline_name==pipeline_name)
        if len(cs) == 0:
            print(f"No results for pipeline {pipeline_name}")
            return
        else:
            temp_edges = [(c.node1, c.node2) for c in cs]
            self.meas_graph.clear()
            self.meas_graph.add_edges_from(temp_edges)
            for c in cs:
                c.node1.exp = c.node2.exp = self

    def show_pipeline(self, pipeline_name=None):
        """If a pipeline name is specified query the database, otherwise show the current pipeline."""
        if pipeline_name:
            cs = select(c for c in bbndb.auspex.Connection if c.pipeline_name==pipeline_name)
            if len(cs) == 0:
                print(f"No results for pipeline {pipeline_name}")
                return
            temp_edges = [(c.node1, c.node2) for c in cs]
            graph = nx.DiGraph()
            graph.add_edges_from(temp_edges)
        else:
            graph = self.meas_graph

        if not graph or len(graph.nodes()) == 0:
            print("Could not find any nodes. Has a pipeline been created (try running create_default_pipeline())")
        else:
            labels = {n: n.node_label() for n in graph.nodes()}
            colors = ["#3182bd" if isinstance(n, bbndb.auspex.QubitProxy) else "#ff9933" for n in graph.nodes()]
            self.plot_graph(graph, labels, colors=colors)

    def reset_pipelines(self):
        for qp in self.qubit_proxies.values():
            qp.clear_pipeline()
            qp.auto_create_pipeline()

    def clear_pipelines(self):
        for qp in self.qubit_proxies.values():
            qp.clear_pipeline()

    def show_connectivity(self):
        pass

    def plot_graph(self, graph, labels, prog="dot", colors='r'):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog=prog)

        # Create position copies for shadows, and shift shadows
        pos_shadow = copy.copy(pos)
        pos_labels = copy.copy(pos)
        for idx in pos_shadow.keys():
            pos_shadow[idx] = (pos_shadow[idx][0] + 0.01, pos_shadow[idx][1] - 0.01)
            pos_labels[idx] = (pos_labels[idx][0] + 0, pos_labels[idx][1] + 15 )
        nx.draw_networkx_nodes(graph, pos_shadow, node_size=100, node_color='k', alpha=0.5)
        nx.draw_networkx_nodes(graph, pos, node_size=100, node_color=colors, linewidths=1, alpha=1.0)
        nx.draw_networkx_edges(graph, pos, width=1)
        nx.draw_networkx_labels(graph, pos_labels, labels, font_size=10, bbox=dict(facecolor='white', alpha=0.95), horizontalalignment="center")

        ax = plt.gca()
        ax.axis('off')
        ax.set_xlim((ax.get_xlim()[0]-20.0, ax.get_xlim()[1]+20.0))
        ax.set_ylim((ax.get_ylim()[0]-20.0, ax.get_ylim()[1]+20.0))
        plt.show()

    def __getitem__(self, key):
        return self.qubit(key)