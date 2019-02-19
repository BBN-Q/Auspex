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
from functools import wraps

import numpy as np
import networkx as nx
from IPython.display import HTML, display
from sqlalchemy import inspect

import auspex.config as config
import auspex.instruments
import auspex.filters
from auspex.stream import DataAxis
import bbndb
from auspex.log import logger

pipelineMgr = None

def check_session_dirty(f):
    """Since we can't mix db objects from separate sessions, re-fetch entities by their unique IDs"""
    @wraps(f)
    def wrapper(cls, *args, **kwargs):
        if (len(cls.session.dirty | cls.session.new)) == 0:
            if 'force' in kwargs:
                kwargs.pop('force')
            return f(cls, *args, **kwargs)
        elif 'force' in kwargs and kwargs['force']:
            kwargs.pop('force')
            return f(cls, *args, **kwargs)
        else:
            raise Exception("Uncommitted transactions for working database. Either use force=True or commit/revert your changes.")
    return wrapper

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
        self.qubit_proxies = {q.label: bbndb.auspex.QubitProxy(pipelineMgr=self, qubit_name=q.label) for q in qubits}
        for q in self.qubit_proxies.values():
            q.pipelineMgr = self

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

        # for el in self.meas_graph.nodes():
        #     self.session.add(el)
        self.session.commit()
        self.save_as("working")

    def qubit(self, qubit_name):
        return self.qubit_proxies[qubit_name]

    def ls(self):
        i = 0
        table_code = ""

        for name, time in self.session.query(bbndb.auspex.Connection.pipeline_name, bbndb.auspex.Connection.time).distinct().all():
            y, d, t = map(time.strftime, ["%Y", "%b. %d", "%I:%M:%S %p"])
            table_code += f"<tr><td>{i}</td><td>{y}</td><td>{d}</td><td>{t}</td><td>{name}</td></tr>"
            i += 1
        display(HTML(f"<table><tr><th>id</th><th>Year</th><th>Date</th><th>Time</th><th>Name</th></tr><tr>{table_code}</tr></table>"))

    @check_session_dirty
    def save_as(self, name):
        now = datetime.datetime.now()
        for n1, n2 in self.meas_graph.edges():
            new_node1 = bbndb.copy_sqla_object(n1, self.session)
            new_node2 = bbndb.copy_sqla_object(n2, self.session)
            c = bbndb.auspex.Connection(pipeline_name=name, node1=new_node1, node2=new_node2, time=now,
                                        node1_name=self.meas_graph[n1][n2]["connector_out"],
                                        node2_name=self.meas_graph[n1][n2]["connector_in"])
            self.session.add_all([n1, n2, c])
        self.session.commit()

    @check_session_dirty
    def load(self, pipeline_name):
        cs = self.session.query(bbndb.auspex.Connection).filter_by(pipeline_name=pipeline_name).all()
        if len(cs) == 0:
            raise Exception(f"Could not find pipeline named {pipeline_name}")
        else:
            nodes = []
            new_by_old = {}
            edges = []
            # Find all nodes
            for c in cs:
                nodes.extend([c.node1, c.node2])
            # Copy unique nodes into new objects
            for n in list(set(nodes)):
                new_by_old[n] = bbndb.copy_sqla_object(n, self.session)
                new_by_old[n].pipeline = self
            # Add edges between new objects
            for c in cs:
                edges.append((new_by_old[c.node1], new_by_old[c.node2], {'connector_in':c.node2_name,  'connector_out':c.node1_name}))
            self.session.add_all(new_by_old.values())
            self.session.commit()
            self.meas_graph.clear()
            self.meas_graph.add_edges_from(edges)

    def show_pipeline(self, pipeline_name=None):
        """If a pipeline name is specified query the database, otherwise show the current pipeline."""
        if pipeline_name:
            cs = self.session.query(bbndb.auspex.Connection).filter_by(pipeline_name=pipeline_name).all()
            if len(cs) == 0:
                print(f"No results for pipeline {pipeline_name}")
                return
            temp_edges = [(c.node1, c.node2, {'connector_in':c.node2_name,  'connector_out':c.node1_name}) for c in cs]
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

    def print(self, qubit_name=None):
        if qubit_name:
            nodes = nx.algorithms.dag.descendants(self.meas_graph, self.qubit(qubit_name))
        else:
            nodes = self.meas_graph.nodes()
        table_code = ""

        for node in nodes:
            label = node.label if node.label else "Unlabeled"
            table_code += f"<tr><td><b>{node.node_type}</b> ({node.qubit_name})</td><td></td><td><i>{label}</i></td></td><td></tr>"
            inspr = inspect(node)
            for c in list(node.__mapper__.columns):
                if c.name not in ["id", "label", "qubit_name", "node_type"]:
                    hist = getattr(inspr.attrs, c.name).history
                    dirty = "Yes" if hist.has_changes() else ""
                    table_code += f"<tr><td></td><td>{c.name}</td><td>{getattr(node,c.name)}</td><td>{dirty}</td></tr>"
        display(HTML(f"<table><tr><th>Name</th><th>Attribute</th><th>Value</th><th>Uncommitted Changes</th></tr><tr>{table_code}</tr></table>"))

    def reset_pipelines(self):
        for qp in self.qubit_proxies.values():
            qp.clear_pipeline()
            qp.create_default_pipeline()

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
