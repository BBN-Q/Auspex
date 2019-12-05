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
import bbndb.auspex as adb
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
        self.meas_graph    = None

        if not bbndb.get_cl_session():
            raise Exception("Auspex expects db to be created already by QGL. Please create a ChannelLibrary.")

        self.session = bbndb.get_pl_session()

        # Check to see whether there is already a temp database
        available_pipelines = list(set([pn[0] for pn in list(self.session.query(adb.Connection.pipeline_name).all())]))
        if "working" in available_pipelines:
            connections = self.get_connections_by_name('working')
            edges = [(c.node1.hash_val, c.node2.hash_val, {'connector_in':c.node2_name,  'connector_out':c.node1_name}) for c in connections]
            nodes = []
            nodes.extend(list(set([c.node1 for c in connections])))
            nodes.extend(list(set([c.node2 for c in connections])))

            self.meas_graph = nx.DiGraph()
            for node in nodes:
                node.pipelineMgr = self
                self.meas_graph.add_node(node.hash_val, node_obj=node)
            self.meas_graph.add_edges_from(edges)
            adb.__current_pipeline__ = self
        else:
            logger.info("Could not find an existing pipeline. Please create one.")

        pipelineMgr = self

    def add_qubit_pipeline(self, qubit_label, stream_type, auto_create=True, buffers=False):
        # if qubit_label not in self.stream_selectors:
        m = bbndb.qgl.Measurement
        mqs = [l[0] for l in self.session.query(m.label).join(m.channel_db, aliased=True).filter_by(label="working").all()]
        if f'M-{qubit_label}' not in mqs:
            raise Exception(f"Could not find qubit {qubit_label} in pipeline...")

        ss_label = qubit_label+"-"+stream_type
        select = adb.StreamSelect(pipelineMgr=self, stream_type=stream_type, qubit_name=qubit_label, label=ss_label)
        self.session.add(select)
        if not self.meas_graph:
            self.meas_graph = nx.DiGraph()
        self.meas_graph.add_node(select.hash_val, node_obj=select)

        if auto_create:
            select.create_default_pipeline(buffers=buffers)

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

        # Build a mapping of qubits to receivers, construct qubit proxies
        receiver_chans_by_qubit = {}
        receiver_chans_by_qubit_label = {}
        for m in measurements:
            q = [c for c in cdb.channels if c.label==m.label[2:]][0]
            receiver_chans_by_qubit[q] = m.receiver_chan
            receiver_chans_by_qubit_label[q.label] = m.receiver_chan
            
        rx_chans = []
        multiplexed_groups = []
        for q in qubits:
            rx_chan = receiver_chans_by_qubit_label[q.label]
            if rx_chan in rx_chans:
                multiplexed_groups[rx_chans.index(rx_chan)].append(q)
            else:
                rx_chans.append(rx_chan)
                multiplexed_groups.append([q])

        stream_selectors = {}
        for group in multiplexed_groups:
            initial_stream_qubit = group[0]
            labels = [q.label for q in group]
            group_label = '-'.join(labels)
            stream_selectors[group_label] = {'default' : adb.StreamSelect(pipelineMgr=self, label = group_label)}

        for sels in stream_selectors.values():
            sel = sels['default']
            labels = sel.label.split('-')
            source_qubit = labels[0]
            rcvr = receiver_chans_by_qubit_label[source_qubit]
            sel.available_streams = [st.strip() for st in rcvr.receiver.stream_types.split(",")]
            sel.stream_type = sel.available_streams[0]

        # generate the pipeline automatically
        self.meas_graph = nx.DiGraph()
        for sels in stream_selectors.values():
            sel = sels['default']
            qbs = sel.label.split('-')
            for q in qbs:
                sel.qubit_name = q
                sel.create_default_pipeline(buffers=buffers)
            sel.qubit_name = qbs[0]
            self.session.add(sel)

        self._push_meas_graph_to_db(self.meas_graph, "working")
        self.session.commit()

    def recreate_pipeline(self, qubits=None, buffers=False):
        sels = self.get_current_stream_selectors()
        if len(sels) == 0:
            raise Exception("Cannot recreate a pipeline that has not been created. Try create_default_pipeline first.")
        for sel in sels:
            sel.clear_pipeline()
            sel.create_default_pipeline(buffers=buffers)
        self._push_meas_graph_to_db(self.meas_graph, "working")

    def get_stream_selector(self, pipeline_name):
        sels = self.get_current_stream_selectors()
        sels.sort(key=lambda x: x.qubit_name)
        selectors = [sel.hash_val for sel in sels]
        qubit_names = [sel.qubit_name for sel in sels]
        sel_labels = [sel for sel in sels if pipeline_name in sel.label]
        name_f = lambda s: s.qubit_name if qubit_names.count(s.qubit_name) == 1 else s.qubit_name + " " + s.stream_type
        sel_by_name = {name_f(sel): sel for sel in sels}
        
        if pipeline_name in sel_by_name:
            return sel_by_name[pipeline_name]
        elif len(sel_labels)== 1:
            return sel_labels[0]
        else:
            raise Exception(f"Name {pipeline_name} does not specify a pipeline. If there are multiple pipelines for a qubit you must specify 'qubit_name pipeline_name'")
        return sel_by_name[pipeline_name]

    def ls(self):
        i = 0
        table_code = ""

        for name, time in self.session.query(adb.Connection.pipeline_name, adb.Connection.time).distinct().all():
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
            c = adb.Connection(pipeline_name=name, node1=new_node1, node2=new_node2, time=now,
                                        node1_name=self.meas_graph[n1][n2]["connector_out"],
                                        node2_name=self.meas_graph[n1][n2]["connector_in"])
            self.session.add_all([self.meas_graph.nodes[n1]['node_obj'], self.meas_graph.nodes[n2]['node_obj'], c])
        self.session.commit()

    @check_session_dirty
    def load(self, pipeline_name, index=1):
        """Load the latest instance for a particular name. Specifying index = 2 will select the second most recent instance """
        cs = self.get_connections_by_name(pipeline_name, index)
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
                edges.append((new_by_old[c.node1].hash_val, new_by_old[c.node2].hash_val, {'connector_in':c.node2_name,  'connector_out':c.node1_name}))
            self.session.add_all(new_by_old.values())
            self.session.commit()
            self.meas_graph.clear()
            for new_node in new_by_old.values():
                new_node.pipelineMgr = self
                self.meas_graph.add_node(new_node.hash_val, node_obj=new_node)
            self.meas_graph.add_edges_from(edges)

    def get_connections_by_name(self, pipeline_name, index=1):
        cs = self.session.query(adb.Connection).filter_by(pipeline_name=pipeline_name).order_by(adb.Connection.time.desc()).all()
        timestamps = [c.time for c in cs]
        timestamps = sorted(set(timestamps), key=timestamps.index)
        cs = [c for c in cs if c.time == timestamps[index-1]]
        return cs

    def _push_meas_graph_to_db(self, graph, pipeline_name):
        # Clear out existing connections if on working:
        if pipeline_name == "working":
            self.session.query(adb.Connection).filter_by(pipeline_name=pipeline_name).delete()
        now = datetime.datetime.now()
        for n1, n2 in graph.edges():
            new_node1 = bbndb.copy_sqla_object(graph.nodes[n1]['node_obj'], self.session)
            new_node2 = bbndb.copy_sqla_object(graph.nodes[n2]['node_obj'], self.session)
            c = adb.Connection(pipeline_name=pipeline_name, node1=new_node1, node2=new_node2, time=now,
                                        node1_name=graph[n1][n2]["connector_out"],
                                        node2_name=graph[n1][n2]["connector_in"])
            self.session.add(c)

    def get_current_stream_selectors(self):
        return [dat['node_obj'] for n, dat in self.meas_graph.nodes(data=True) if isinstance(dat['node_obj'], adb.StreamSelect)]

    def show_pipeline(self, subgraph=None, pipeline_name=None):
        """If a pipeline name is specified query the database, otherwise show the current pipeline."""
        if subgraph:
            graph = subgraph
        elif pipeline_name:
            cs = self.session.query(adb.Connection).filter_by(pipeline_name=pipeline_name).all()
            if len(cs) == 0:
                print(f"No results for pipeline {pipeline_name}")
                return
            temp_edges = [(c.node1.hash_val, c.node2.hash_val, {'connector_in':c.node2_name,  'connector_out':c.node1_name}) for c in cs]
            nodes = set([c.node1 for c in cs] + [c.node2 for c in cs])
            for node in nodes:
                self.meas_graph.add_node(node.hash_val, node_obj=node)
            graph = nx.DiGraph()
            graph.add_edges_from(temp_edges)
        else:
            graph = self.meas_graph

        if not graph or len(graph.nodes()) == 0:
            raise Exception("Could not find any nodes. Has a pipeline been created (try running create_default_pipeline())")
        else:
            from bqplot import Figure, LinearScale
            from bqplot.marks import Graph, Lines, Label
            from ipywidgets import Layout, HTML
            from IPython.display import HTML as IPHTML, display

            # nodes     = list(dgraph.nodes())
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

            sel_objs = [dat['node_obj'] for n,dat in graph.nodes(data=True) if isinstance(dat['node_obj'], adb.StreamSelect)]
            sel_objs.sort(key=lambda x: x.qubit_name)
            selectors = [sel.hash_val for sel in sel_objs]
            qubit_names = [sel.qubit_name for sel in sel_objs]
            pipeline_names = [sel.qubit_name if qubit_names.count(sel.qubit_name) == 1 else sel.qubit_name + " " + sel.stream_type for sel in sel_objs]

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

            hierarchy = [next_level([q]) for q in selectors]
            widest = [max([len(row) for row in qh]) for qh in hierarchy]
            for i in range(1, len(selectors)):
                offset = sum(widest[:i])
                loc[selectors[i]]['x'] += offset
                for n in nx.descendants(graph, selectors[i]):
                    loc[n]['x'] += offset

            x = [loc[n]['x'] for n in graph.nodes()]
            y = [loc[n]['y'] for n in graph.nodes()]
            xs = LinearScale(min=min(x)-0.5, max=max(x)+0.6)
            ys = LinearScale(min=min(y)-0.5, max=max(y)+0.6)
            fig_layout = Layout(width='960px', height='500px')
            graph      = Graph(node_data=node_data, link_data=link_data, x=x, y=y, scales={'x': xs, 'y': ys},
                                link_type='line', colors=['orange'] * len(node_data), directed=True)
            bgs_lines = []
            middles   = []
            for i in range(len(selectors)):
                if i==0:
                    start = -0.4
                    end = widest[0]-0.6
                elif i == len(selectors):
                    start = sum(widest)-0.4
                    end = max(x)+0.4
                else:
                    start = sum(widest[:i])-0.4
                    end = sum(widest[:i+1])-0.6
                middles.append(0.5*(start+end))
                bgs_lines.append(Lines(x=[start, end], y=[[min(y)-0.5,min(y)-0.5],[max(y)+0.5,max(y)+0.5]], scales= {'x': xs, 'y': ys},
                                      fill='between',   # opacity does not work with this option
                                      fill_opacities = [0.1+0.5*i/len(selectors)],
                                      stroke_width = 0.0
                                     ))
            labels = Label(x=middles, y=[max(y)+0.65 for m in middles], text=pipeline_names, align='middle', scales= {'x': xs, 'y': ys},
                default_size=14, font_weight='bolder', colors=['#4f6367'])

            fig        = Figure(marks=bgs_lines+[graph, labels], layout=fig_layout)
            graph.tooltip = table
            graph.on_hover(hover_handler)
            return fig

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()

    def print(self, pipeline_name=None):
        if pipeline_name:
            nodes = list(nx.algorithms.dag.descendants(self.meas_graph, self.get_stream_selector(pipeline_name).hash_val)) + [self.get_stream_selector(pipeline_name).hash_val]
        else:
            nodes = self.meas_graph.nodes()
        table_code = ""

        for node in nodes:
            if not isinstance(node, adb.StreamSelect):
                node = self.meas_graph.nodes[node]['node_obj']
            label = node.label if node.label else "Unlabeled"
            table_code += f"<tr><td><b>{node.node_type}</b> ({node.qubit_name})</td><td></td><td><i>{label}</i></td></td><td></tr>"
            inspr = inspect(node)
            for c in list(node.__mapper__.columns):
                if c.name not in ["id", "label", "qubit_name", "node_type"]:
                    hist = getattr(inspr.attrs, c.name).history
                    dirty = "Yes" if hist.has_changes() else ""
                    if c.name == "kernel_data":
                        table_code += f"<tr><td></td><td>{c.name}</td><td>Binary Data of length {len(node.kernel)}</td><td>{dirty}</td></tr>"
                    else:
                        table_code += f"<tr><td></td><td>{c.name}</td><td>{getattr(node,c.name)}</td><td>{dirty}</td></tr>"
        display(HTML(f"<table><tr><th>Name</th><th>Attribute</th><th>Value</th><th>Uncommitted Changes</th></tr><tr>{table_code}</tr></table>"))

    def reset_pipelines(self):
        for sel in self.get_current_stream_selectors():
            sel.clear_pipeline()
            sel.create_default_pipeline()

    def clear_pipelines(self):
        for sel in self.get_current_stream_selectors():
            sel.clear_pipeline()
            sel.clear_pipeline()

    def __getitem__(self, key):
        return self.get_stream_selector(key)
