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
import operator
from functools import reduce
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

        # Check to see whether there is already a temp database
        available_pipelines = list(set([pn[0] for pn in list(self.session.query(bbndb.auspex.Connection.pipeline_name).all())]))
        if "working" in available_pipelines:
            connections = list(self.session.query(bbndb.auspex.Connection).filter_by(pipeline_name="working").all())
            edges = [(str(c.node1), str(c.node2), {'connector_in':c.node2_name,  'connector_out':c.node1_name}) for c in connections]
            nodes = []
            nodes.extend(list(set([c.node1 for c in connections])))
            nodes.extend(list(set([c.node2 for c in connections])))

            self.meas_graph = nx.DiGraph()
            for node in nodes:
                node.pipelineMgr = self
                self.meas_graph.add_node(str(node), node_obj=node)
            self.meas_graph.add_edges_from(edges)
            bbndb.auspex.__current_pipeline__ = self

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
            new_node1 = bbndb.copy_sqla_object(self.meas_graph.nodes[n1]['node_obj'], self.session)
            new_node2 = bbndb.copy_sqla_object(self.meas_graph.nodes[n2]['node_obj'], self.session)
            c = bbndb.auspex.Connection(pipeline_name=name, node1=new_node1, node2=new_node2, time=now,
                                        node1_name=self.meas_graph[n1][n2]["connector_out"],
                                        node2_name=self.meas_graph[n1][n2]["connector_in"])
            self.session.add_all([self.meas_graph.nodes[n1]['node_obj'], self.meas_graph.nodes[n2]['node_obj'], c])
        self.session.commit()

    @check_session_dirty
    def load(self, pipeline_name):
        cs = self.session.query(bbndb.auspex.Connection).filter_by(pipeline_name=pipeline_name).all()
        if len(cs) == 0:
            raise Exception(f"Could not find pipeline named {pipeline_name}")
        else:
            self.clear_pipelines()
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
                edges.append((str(new_by_old[c.node1]), str(new_by_old[c.node2]), {'connector_in':c.node2_name,  'connector_out':c.node1_name}))
            self.session.add_all(new_by_old.values())
            self.session.commit()
            self.meas_graph.clear()
            for new_node in new_by_old.values():
                self.meas_graph.add_node(str(new_node), node_obj=new_node)
            self.meas_graph.add_edges_from(edges)

    def show_pipeline(self, subgraph=None, pipeline_name=None):
        """If a pipeline name is specified query the database, otherwise show the current pipeline."""
        if subgraph:
            graph = subgraph
        elif pipeline_name:
            cs = self.session.query(bbndb.auspex.Connection).filter_by(pipeline_name=pipeline_name).all()
            if len(cs) == 0:
                print(f"No results for pipeline {pipeline_name}")
                return
            temp_edges = [(str(c.node1), str(c.node2), {'connector_in':c.node2_name,  'connector_out':c.node1_name}) for c in cs]
            nodes = set([c.node1 for c in cs] + [c.node2 for c in cs])
            for node in nodes:
                self.meas_graph.add_node(str(node), node_obj=node)
            graph = nx.DiGraph()
            graph.add_edges_from(temp_edges)
        else:
            graph = self.meas_graph

        if not graph or len(graph.nodes()) == 0:
            print("Could not find any nodes. Has a pipeline been created (try running create_default_pipeline())")
        else:
            from bqplot import Figure, LinearScale
            from bqplot.marks import Graph
            from ipywidgets import Layout, HTML
            from IPython.display import HTML as IPHTML, display

            # nodes     = list(graph.nodes())
            indices   = {n: i for i, n in enumerate(graph.nodes())}
            node_data = [{'label': dat['node_obj'].node_label(), 'data': dat['node_obj'].print(show=False)} for n,dat in graph.nodes(data=True)]
            link_data = [{'source': indices[s], 'target': indices[t]} for s, t in graph.edges()]

            # Update the tooltip chart
            table = HTML("<b>Re-evaluate this plot to see information about filters. Otherwise it will be stale.</b>")
            table.add_class("hover_tooltip")
            display(IPHTML("""
            <style>
                .hover_tooltip table { border-collapse: collapse; padding: 8px; }
                .hover_tooltip th, .hover_tooltip td { text-align: left; padding: 8px; }
                .hover_tooltip tr:nth-child(even) { background-color: #cccccc; padding: 8px; }
            </style>
            """))
            hovered_symbol = ''
            def hover_handler(self, content, hovered_symbol=hovered_symbol, table=table):
                symbol = content.get('data', '')
                
                if(symbol != hovered_symbol):
                    hovered_symbol = symbol
                    table.value = symbol['data']

            node_locations = {}

            qubits = [n for n,dat in graph.nodes(data=True) if isinstance(dat['node_obj'], bbndb.auspex.QubitProxy)]
            loc = {}
            def next_level(nodes, iteration=0, offset=0, accum=[]):
                if len(accum) == 0:
                    loc[nodes[0]] = {'x': 0, 'y': 0}
                    accum = [nodes]
                next_gen_nodes = list(reduce(operator.add, [list(graph.successors(n)) for n in nodes]))
                l = len(next_gen_nodes)
                if l > 0:
                    for k,n in enumerate(next_gen_nodes):
                        loc[n] = {'x': k, 'y': -(iteration+1)}
                    accum.append(next_gen_nodes)
                    return next_level(next_gen_nodes, iteration=iteration+1, offset=2.5*l, accum=accum)
                else:
                    return accum

            hierarchy = [next_level([q]) for q in qubits]
            widest = [max([len(row) for row in qh]) for qh in hierarchy]
            for i in range(1, len(qubits)):
                offset = sum(widest[:i])
                loc[qubits[i]]['x'] += offset
                for n in nx.descendants(graph, qubits[i]):
                    loc[n]['x'] += offset
            
            x = [loc[n]['x'] for n in graph.nodes()]
            y = [loc[n]['y'] for n in graph.nodes()]
            xs = LinearScale(min=min(x)-0.5, max=max(x)+0.5)
            ys = LinearScale(min=min(y)-0.5, max=max(y)+0.5)
            fig_layout = Layout(width='960px', height='500px')
            graph      = Graph(node_data=node_data, link_data=link_data, x=x, y=y, scales={'x': xs, 'y': ys},
                                link_type='line', colors=['orange'] * len(node_data), directed=True)
            fig        = Figure(marks=[graph], layout=fig_layout)
            graph.tooltip = table
            graph.on_hover(hover_handler)
            return fig


    def print(self, qubit_name=None):
        if qubit_name:
            nodes = nx.algorithms.dag.descendants(self.meas_graph, str(self.qubit(qubit_name)))
        else:
            nodes = self.meas_graph.nodes()
        table_code = ""

        for node in nodes:
            node = self.meas_graph.nodes[node]['node_obj']
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

    def __getitem__(self, key):
        return self.qubit(key)
