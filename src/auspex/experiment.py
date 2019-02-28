# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import uuid
import json

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    from queue import Queue as Queue
    from threading import Event
    from threading import Thread as Process
    from threading import Thread as Thread
else:
    from multiprocessing import Queue as Queue
    from multiprocessing import Event
    from multiprocessing import Process
    from threading import Thread as Thread

import inspect
import time
import copy
import itertools
import logging
import signal
import numbers
import subprocess
import queue
import re
import cProfile
from functools import partial

import zmq
import numpy as np
import scipy as sp
import networkx as nx

from auspex.instruments.instrument import Instrument
from auspex.parameter import ParameterGroup, FloatParameter, IntParameter, Parameter
from auspex.sweep import Sweeper
from auspex.stream import DataStream, DataAxis, SweepAxis, DataStreamDescriptor, InputConnector, OutputConnector
from auspex.filters import Plotter, MeshPlotter, ManualPlotter, WriteToFile, DataBuffer, Filter
from auspex.log import logger
import auspex.config

def auspex_plot_server():
    client_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"plot_server.py")
    subprocess.Popen(['python', client_path], env=os.environ.copy())

def update_filename(filename, add_date=True):
    """Update the file number and date."""
    basename, _ = os.path.splitext(filename)
    dirname  = os.path.dirname(os.path.abspath(filename))

    if add_date:
        date     = time.strftime("%y%m%d")
        dirname  = os.path.join(dirname, date)
        basename = os.path.join(dirname, os.path.basename(basename))

    # Set the file number to the maximum in the current folder + 1
    filenums = []
    if os.path.exists(dirname):
        for f in os.listdir(dirname):
            if 'auspex' in f and os.path.exists(os.path.join(dirname, f)):
                nums = re.findall('-(\d{4})\.', f)
                if len(nums) > 0:
                    filenums.append(int(nums[0]))

    i = max(filenums) + 1 if filenums else 0
    return "{}-{:04d}".format(basename,i)

class ExperimentGraph(object):
    def __init__(self, edges):
        self.dag = None
        self.edges = []
        self.create_graph(edges)

    def dfs_edges(self):
        # Edge depth-first traversal of the graph

        # Find the input nodes
        input_nodes = [n for n in self.dag.nodes() if self.dag.in_degree(n) == 0]
        logger.debug("Input nodes for DFS are '%s'", input_nodes)

        dfs_edge_iters  = [nx.edge_dfs(self.dag, input_node) for input_node in input_nodes]
        processed_edges = [] # Keep track of what we've initialized

        for ei in dfs_edge_iters:
            for edge in ei:
                if edge not in processed_edges:
                    processed_edges.append(edge)
                    yield edge

    def create_graph(self, edges):
        dag = nx.DiGraph()
        self.edges = []
        for edge in edges:
            obj = DataStream(name="{}_TO_{}".format(edge[0].name, edge[1].name))
            edge[0].add_output_stream(obj)
            edge[1].add_input_stream(obj)
            self.edges.append(obj)
            dag.add_edge(edge[0].parent, edge[1].parent, object=obj)

        self.dag = dag

class MetaExperiment(type):
    """Meta class to bake the instrument objects into a class description
    """

    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding controls to %s", name)
        self._parameters        = {}
        self._instruments       = {}
        self._constants         = {}

        # Beware, passing objects won't work at parse time
        self._output_connectors = {}

        # Parse ourself
        self._exp_src = inspect.getsource(self)

        for k,v in dct.items():
            if isinstance(v, Instrument):
                logger.debug("Found '%s' instrument", k)
                self._instruments[k] = v
            elif isinstance(v, Parameter):
                logger.debug("Found '%s' parameter", k)
                if v.name is None:
                    v.name = k
                self._parameters[k] = v
            elif isinstance(v, OutputConnector):
                logger.debug("Found '%s' output connector.", k)
                self._output_connectors[k] = v
            elif isinstance(v, numbers.Number) or isinstance(v, str):
                self._constants[k] = v
                # Keep track of numerical parameters

class Experiment(metaclass=MetaExperiment):
    """The measurement loop to be run for each set of sweep parameters."""
    def __init__(self):
        super(Experiment, self).__init__()
        # Experiment name
        self.name = None

        # Sweep control
        self.sweeper = Sweeper()

        # This holds the experiment graph
        self.graph = None

        # Should we show the dashboard?
        self.dashboard = False

        # Create and use plots?
        self.do_plotting = False

        # Unique ID for this experiment
        self.uuid = str(uuid.uuid4())

        # Disconnect at the end of experiment?
        self.keep_instruments_connected = False

        # Also keep references to all of the plot filters
        self.plotters = [] # Standard pipeline plotters using streams
        self.extra_plotters = [] # Plotters using streams, but not the pipeline
        self.manual_plotters = [] # Plotters using neither streams nor the pipeline
        self.manual_plotter_callbacks = [] # These are called at the end of run
        self._extra_plots_to_streams = {}

        # Furthermore, keep references to all of the file writers and buffers.
        self.writers = []
        self.buffers = []

        # ExpProgressBar object to display progress bars
        self.progressbar = None

        # indicates whether the instruments are already connected
        self.instrs_connected = False

        # indicates whether this is the first (or only) experiment in a series (e.g. for pulse calibrations)
        self.first_exp = True

        # add date to data files?
        self.add_date = False

        # Things we can't metaclass
        self.output_connectors = {}
        for oc in self._output_connectors.keys():
            a = OutputConnector(name=oc, data_name=oc, unit=self._output_connectors[oc].data_unit, parent=self)
            a.parent = self

            self.output_connectors[oc] = a
            setattr(self, oc, a)

        # Some instruments don't clean up well after themselves, reconstruct them on a
        # per instance basis. These instruments contain a wide variety of complex behaviors
        # and rely on other classes and data structures, so we avoid copying them and
        # run through the constructor instead.
        self._instruments_instance = {}
        for n in self._instruments.keys():
            new_cls = type(self._instruments[n])
            new_inst = new_cls(resource_name=self._instruments[n].resource_name, name=self._instruments[n].name)
            setattr(self, n, new_inst)
            self._instruments_instance[n] = new_inst
        self._instruments = self._instruments_instance

        # We don't want to add parameters to the base class, so do the same here.
        # These aren't very complicated objects, so we'll throw caution to the wind and
        # try copying them directly.
        self._parameters_instance = {}
        for n, v in self._parameters.items():
            new_inst = copy.deepcopy(v)
            setattr(self, n, new_inst)
            self._parameters_instance[n] = new_inst
        self._parameters = self._parameters_instance

        # Based on the logging level, infer whether we want asyncio debug
        do_debug = logger.getEffectiveLevel() <= logging.DEBUG

        # Run the stream init
        self.init_streams()

    def set_graph(self, edges):
        unique_nodes = []
        for eb, ee in edges:
            if eb.parent not in unique_nodes:
                unique_nodes.append(eb.parent)
            if ee.parent not in unique_nodes:
                unique_nodes.append(ee.parent)
        self.nodes = unique_nodes
        self.graph = ExperimentGraph(edges)

    def init_streams(self):
        """Establish the base descriptors for any internal data streams and connectors."""
        pass

    def init_instruments(self):
        """Gets run before a sweep starts"""
        pass

    def shutdown_instruments(self):
        """Gets run after a sweep ends, or when the program is terminated."""
        pass

    def run(self):
        """This is the inner measurement loop, which is the smallest unit that
        is repeated across various sweep variables. For more complicated run control
        than can be provided by the automatic sweeping, the full experimental
        operation should be defined here"""
        pass

    def set_stream_compression(self, compression="zlib"):
        for oc in self.output_connectors.values():
            for os in oc.output_streams:
                os.compression = compression

    def reset(self):
        for edge in self.graph.edges:
            edge.reset()

    def update_descriptors(self):
        logger.debug("Starting descriptor update in experiment.")
        for oc in self.output_connectors.values():
            oc.descriptor._exp_src = self._exp_src
            for k,v in self._parameters.items():
                oc.descriptor.add_param(k, v.value)
                if v.unit is not None:
                    oc.descriptor.add_param('unit_'+k, v.unit)
            for k in self._constants.keys():
                if hasattr(self,k):
                    v = getattr(self,k)
                    oc.descriptor.add_param(k, v)
            oc.descriptor.visited_tuples = []
            oc.update_descriptors()

    def declare_done(self):
        for oc in self.output_connectors.values():
            for os in oc.output_streams:
                # TODO: why does any queue interaction prevent adding out of order?
                while not os.queue.empty():
                    time.sleep(0.05)
                os.push_event("done")
        for p in self.extra_plotters:
            stream = self._extra_plots_to_streams[p]
            stream.push_event("done")

    def sweep(self):
        # Set any static parameters
        static_params = [p for p in self._parameters.values() if p not in self.sweeper.swept_parameters()]
        for p in static_params:
            p.push()

        # Keep track of the previous values
        logger.debug("Waiting for filters.")
        last_param_values = None
        logger.debug("Starting experiment sweep.")

        while True:
            # Increment the sweeper, which returns a list of the current
            # values of the SweepAxes (no DataAxes).
            sweep_values, axis_names = self.sweeper.update()

            if self.sweeper.is_adaptive():
                # Add the new tuples to the stream descriptors
                for oc in self.output_connectors.values():
                    # Obtain the lists of values for any fixed
                    # DataAxes and append them to them to the sweep_values
                    # in preperation for finding all combinations.
                    vals = [a for a in oc.descriptor.data_axis_values()]
                    if sweep_values:
                        vals  = [[v] for v in sweep_values] + vals
                    # Find all coordinate tuples and update the list of
                    # tuples that the experiment has probed.
                    nested_list    = list(itertools.product(*vals))
                    flattened_list = [tuple((val for sublist in line for val in sublist)) for line in nested_list]
                    oc.descriptor.visited_tuples = oc.descriptor.visited_tuples + flattened_list

                    # Since the filters are in separate processes, pass them the same
                    # information so that they may perform the same operations.
                    oc.push_event("new_tuples", (axis_names, sweep_values,))

            # Run the procedure
            self.run()

            # See if the axes want to extend themselves. They will push updates
            # directly to the output_connecters as messages that will be passed
            # through the filter pipeline.
            self.sweeper.check_for_refinement(self.output_connectors)

            # Finish up, checking to see whether we've received all of our data
            if self.sweeper.done():
                self.declare_done()
                break

    # def filters_finished(self):
    #     return all([n.finished_processing.is_set() for n in self.other_nodes if isinstance(n, Filter)])

    def connect_instruments(self):
        # Connect the instruments to their resources
        if not self.instrs_connected:
            for instrument in self._instruments.values():
                instrument.connect()
            self.instrs_connected = True

    def disconnect_instruments(self):
        # Connect the instruments to their resources
        for instrument in self._instruments.values():
            instrument.disconnect()
        self.instrs_connected = False

    # def run_sweeps(self):
    #     if auspex.config.profile:
    #         cProfile.runctx('self._run_sweeps()', globals(), locals(), 'prof-run_sweeps.prof')
    #     else:
    #         self._run_sweeps()

    def final_init(self):
        # Call any final initialization on the filter pipeline
        for n in self.nodes + self.extra_plotters:
            if n != self and hasattr(n, 'final_init'):
                n.final_init()

    def run_sweeps(self):
        # Propagate the descriptors through the network
        self.update_descriptors()
        # Make sure we are starting from scratch... is this necessary?
        self.reset()

        #Make sure we have axes.
        if not any([oc.descriptor.axes for oc in self.output_connectors.values()]):
            logger.warning("There do not appear to be any axes defined for this experiment!")

        # Go find any writers
        self.writers = [n for n in self.nodes if isinstance(n, WriteToFile)]
        self.buffers = [n for n in self.nodes if isinstance(n, DataBuffer)]
        if self.name:
            for w in self.writers:
                w.filename.value = os.path.join(os.path.dirname(w.filename.value), self.name)
        self.filenames = [w.filename.value for w in self.writers]

        # Auto increment the filenames
        for filename in set(self.filenames):
            wrs = [w for w in self.writers if w.filename.value == filename]
            inc_filename = update_filename(filename, add_date=self.add_date)
            for w in wrs:
                w.filename.value = inc_filename
        self.filenames = [w.filename.value for w in self.writers]

        # Remove the nodes with 0 dimension
        self.nodes = [n for n in self.nodes if not(hasattr(n, 'input_connectors') and  n.input_connectors['sink'].descriptor.num_dims()==0)]

        # Go and find any plotters
        self.standard_plotters = [n for n in self.nodes if isinstance(n, (Plotter, MeshPlotter))]
        self.plotters = copy.copy(self.standard_plotters)

        # We might have some additional plotters that are separate from
        # The asyncio filter pipeline
        self.plotters.extend(self.extra_plotters)

        # These use neither streams nor the filter pipeline
        self.plotters.extend(self.manual_plotters)

        # Last minute init
        self.final_init()

        # Launch plot servers.
        if len(self.plotters) > 0:
            self.connect_to_plot_server()

        time.sleep(0.1)
        #connect all instruments
        self.connect_instruments()

        try:
            #initialize instruments
            self.init_instruments()

            def catch_ctrl_c(signum, frame):
                logger.info("Caught SIGINT. Shutting down.")
                self.declare_done() # Ask nicely

                self.shutdown()
                raise NameError("Shutting down.")
                sys.exit(0)

            signal.signal(signal.SIGINT, catch_ctrl_c)

            # We want to wait for the sweep method above,
            # not the experiment's run method, so replace this
            # in the list of tasks.
            self.other_nodes = self.nodes[:]
            self.other_nodes.extend(self.extra_plotters)
            self.other_nodes.remove(self)

            # If we are launching the process dashboard,
            # setup the bokeh server and establish a queue for
            # filters to push data back to our thread below for
            # communication to the bokeh server instance.

            if self.dashboard:
                from .dashboard_server import BokehServerProcess
                from bokeh.plotting import Figure
                from bokeh.client import push_session
                from bokeh.layouts import row, column
                from bokeh.io import curdoc
                from tornado import gen
                from bokeh.models import ColumnDataSource, Legend
                from bokeh.palettes import viridis, Category20 #here viridis is a function that takes\
                #parameter n and provides a palette with n equally(almost) spaced colors.

                bokeh_process = BokehServerProcess(notebook=False)
                bokeh_process.run()
                perf_queue = Queue()

                cpu_fig  = Figure(plot_width=1000, plot_height=300, x_axis_type="datetime", title="CPU Usage")
                mem_fig  = Figure(plot_width=1000, plot_height=300, x_axis_type="datetime", title="Memory Usage")
                vmem_fig = Figure(plot_width=1000, plot_height=300, x_axis_type="datetime", title="Data Processed")

                cpu_fig.xaxis[0].axis_label  = 'Time (s)'
                mem_fig.xaxis[0].axis_label  = 'Time (s)'
                vmem_fig.xaxis[0].axis_label = 'Time (s)'
                cpu_fig.yaxis[0].axis_label  = 'CPU (%)'
                mem_fig.yaxis[0].axis_label  = 'Memory (MB)'
                vmem_fig.yaxis[0].axis_label = 'Throughput (MB)'

                colors       = Category20[20] if len(self.other_nodes) <= 20 else viridis(len(self.other_nodes))
                data_sources = {str(n): ColumnDataSource(data=dict(time=[], cpu=[], mem=[], vmem=[], proc=[])) for n in self.other_nodes}
                cpu_plots    = {str(n): cpu_fig.line(x='time', y='cpu', color=colors[i], line_width=4, source=data_sources[str(n)]) for i, n in enumerate(self.other_nodes)}
                mem_plots    = {str(n): mem_fig.line(x='time', y='mem', color=colors[i], line_width=4, source=data_sources[str(n)]) for i, n in enumerate(self.other_nodes)}
                vmem_plots   = {str(n): vmem_fig.line(x='time', y='proc', color=colors[i], line_width=4, source=data_sources[str(n)]) for i, n in enumerate(self.other_nodes)}

                legend_1 = Legend(items=[(n , [l]) for n, l in mem_plots.items()], location=(0, 0))
                legend_2 = Legend(items=[(n , [l]) for n, l in vmem_plots.items()], location=(0, 0))
                legend_3 = Legend(items=[(n , [l]) for n, l in cpu_plots.items()], location=(0, 0))
                mem_fig.add_layout(legend_1, 'right')
                vmem_fig.add_layout(legend_2, 'right')
                cpu_fig.add_layout(legend_3, 'right')
                container = column([cpu_fig, mem_fig, vmem_fig])
                doc = curdoc()

                exit_perf = Event()

                @gen.coroutine
                def update_perf(name, time_val, cpu, mem, vmem, proc):
                    data_sources[name].stream(dict(time=[time_val], cpu=[cpu], mem=[mem], vmem=[vmem], proc=[proc]))

                def wait_for_perf_updates(q, exit):
                    while not exit.is_set():
                        messages = []

                        while not exit.is_set():
                            try:
                                messages.append(q.get(False))
                            except queue.Empty as e:
                                time.sleep(0.05)
                                break

                        for message in messages:
                            filter_name, time_val, cpu, mem_info, processed = message
                            mem = mem_info[0]/2.**20
                            vmem = mem_info[1]/2.**20
                            proc = processed/2.**20
                            update_perf(filter_name, time_val, cpu, mem, vmem, proc)

                perf_thread = Thread(target=wait_for_perf_updates, args=(perf_queue, exit_perf))
                perf_thread.start()

                for f in self.other_nodes:
                    f.perf_queue = perf_queue

                doc.clear()
                doc.add_root(container)
                session = push_session(doc)
                session.show(container)

            # Start the filter processes
            for n in self.other_nodes:
                n.start()

            # Run the main experiment loop
            self.sweep()

            # Make sure any last minute plotting needs are met
            for plot, callback in zip(self.manual_plotters, self.manual_plotter_callbacks):
                if callback:
                    callback(plot)

            # Wait for the
            time.sleep(0.1)
            times = {n: 0 for n in self.other_nodes}
            dones = {n: False for n in self.other_nodes}
            while False in dones.values():
                time.sleep(1)
                for n in self.other_nodes:
                    if not n.done.is_set():
                        times[n] += 1
                        logger.info(f"{str(n)} not done. Waited {times[n]} times. Is the pipeline backed up at IO stage?")
                    else:
                        dones[n] = True
                        n.join(timeout=0.1)
                # We've had enough...
                if any([t > 10 for t in times.values()]):
                    break

            for buff in self.buffers:
                buff.output_data, buff.descriptor = buff.get_data()

            if self.dashboard:
                exit_perf.set()
                perf_thread.join()

        except Exception as e:
            logger.warning("Encountered error in run sweeps after initializing experiments")
            raise e
        finally:
            self.shutdown()

    def start_manual_plotters(self):
        for mp in self.manual_plotters:
            mp.start()

    def stop_manual_plotters(self):
        for mp in self.manual_plotters:
            mp.stop()

    def shutdown(self):
        logger.debug("Shutting Down!")

        for n in self.other_nodes:
            n.exit.set()
            n.join(0.1)
            if n.is_alive():
                logger.info(f"Terminating {str(n)} aggressively")
                n.terminate()

        logger.debug("Shutting down instruments")
        self.shutdown_instruments()

        if not self.keep_instruments_connected:
            logger.debug("Disconnecting instruments")
            self.disconnect_instruments()

    def add_axis(self, axis, position=0):
        for oc in self.output_connectors.values():
            logger.debug("Adding axis %s to connector %s.", axis, oc.name)
            oc.descriptor.add_axis(axis, position=position)

    def add_sweep(self, parameters, sweep_list, refine_func=None, callback_func=None, metadata=None):
        ax = SweepAxis(parameters, sweep_list, refine_func=refine_func, callback_func=callback_func, metadata=metadata)
        ax.experiment = self
        self.sweeper.add_sweep(ax)
        self.add_axis(ax)
        if ax.unstructured:
            for p, v in zip(parameters, sweep_list[0]):
                p.value = v
        else:
            parameters.value = sweep_list[0]
        return ax

    def clear_sweeps(self):
        """Delete all sweeps present in this experiment."""
        logger.debug("Removing all axes from experiment.")
        self.sweeper.axes = []
        for oc in self.output_connectors.values():
            oc.descriptor.axes = []

    def pop_sweep(self, name):
        """Remove sweep that has a given name."""
        names = [_.name for _ in self.sweeper.axes]
        if name not in names:
            raise KeyError("Could not remove sweep named {}; does not appear to be present.".format(name))
        self.sweeper.axes = [_ for _ in self.sweeper.axes if _.name != name]
        for oc in self.output_connectors.values():
            oc.descriptor.axes = [_ for _ in oc.descriptor.axes if _.name != name]
        logger.debug("Removed sweep {} from experiment".format(name))

    def add_direct_plotter(self, plotter):
        """A plotter that lives outside the filter pipeline, intended for advanced
        use cases when plotting data during refinement."""
        plotter_stream = DataStream()
        plotter.sink.add_input_stream(plotter_stream)
        self.extra_plotters.append(plotter)
        self._extra_plots_to_streams[plotter] = plotter_stream

    def add_manual_plotter(self, plotter, callback=None):
        self.manual_plotters.append(plotter)
        self.manual_plotter_callbacks.append(callback)

    def push_to_plot(self, plotter, data):
        """Push data to a direct plotter."""
        stream = self._extra_plots_to_streams[plotter]
        stream.push_direct(data)

    def connect_to_plot_server(self):
        logger.debug("Found %d plotters", len(self.plotters))

        # Create the descriptor and set uuids for each plot process
        plot_desc = {p.filter_name: p.desc() for p in self.plotters}
        for p in self.plotters:
            p.uuid = self.uuid

        try:
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

        except:
            logger.warning("Exception occured while contacting the plot server. Is it running?")
            for p in self.plotters:
                p.do_plotting = False

        time.sleep(0.5)
