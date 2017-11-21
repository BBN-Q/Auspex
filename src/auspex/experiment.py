# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import inspect
import time
import copy
import itertools
import logging
import asyncio
import signal
import sys
import numbers
import os
import subprocess

import numpy as np
import scipy as sp
import networkx as nx
import h5py
from tqdm import tqdm, tqdm_notebook

from auspex.instruments.instrument import Instrument
from auspex.parameter import ParameterGroup, FloatParameter, IntParameter, Parameter
from auspex.sweep import Sweeper
from auspex.stream import DataStream, DataAxis, SweepAxis, DataStreamDescriptor, InputConnector, OutputConnector
from auspex.filters import Plotter, XYPlotter, MeshPlotter, ManualPlotter, WriteToHDF5, DataBuffer, Filter
from auspex.log import logger
import auspex.config

class ExpProgressBar(object):
    """ Display progress bar(s) on the terminal.

    num: number of progress bars to be display, \
    corresponding to the number of axes (counting from outer most)

        For running in Jupyter Notebook:
    Needs to open '_tqdm_notebook.py',\
    search for 'n = int(s[:npos])'\
    then replace it with 'n = float(s[:npos])'
    """
    def __init__(self, stream=None, num=0, notebook=False):
        super(ExpProgressBar,self).__init__()
        logger.debug("initialize the progress bars.")
        self.stream = stream
        self.num = num
        self.notebook = notebook
        # self.reset(stream=stream)

    def reset(self, stream=None):
        """ Reset the progress bar(s) """
        logger.debug("Update stream descriptor for progress bars.")
        if stream is not None:
            self.stream = stream
        if self.stream is None:
            logger.warning("No stream is associated with the progress bars!")
            self.axes = []
        else:
            self.axes = self.stream.descriptor.axes
        self.num = min(self.num, len(self.axes))
        self.totals = [self.stream.descriptor.num_points_through_axis(axis) for axis in range(self.num)]
        self.chunk_sizes = [max(1,self.stream.descriptor.num_points_through_axis(axis+1)) for axis in range(self.num)]
        logger.debug("Reset the progress bars to initial states.")
        self.bars   = []
        for i in range(self.num):
            if self.notebook:
                self.bars.append(tqdm_notebook(total=self.totals[i]/self.chunk_sizes[i]))
            else:
                self.bars.append(tqdm(total=self.totals[i]/self.chunk_sizes[i]))

    def close(self):
        """ Close all progress bar(s) """
        logger.debug("Close all the progress bars.")
        for bar in self.bars:
            if self.notebook:
                bar.sp(close=True)
            else:
                bar.close()

    def update(self):
        """ Update the status of the progress bar(s) """
        if self.stream is None:
            logger.warning("No stream is associated with the progress bars!")
            num_data = 0
        else:
            num_data = self.stream.points_taken
        logger.debug("Update the progress bars.")
        for i in range(self.num):
            if num_data == 0:
                # Reset the progress bar with a new one
                if self.notebook:
                    self.bars[i].sp(close=True)
                    self.bars[i] = tqdm_notebook(total=self.totals[i]/self.chunk_sizes[i])
                else:
                    self.bars[i].close()
                    self.bars[i] = tqdm(total=self.totals[i]/self.chunk_sizes[i])
            pos = int(10*num_data / self.chunk_sizes[i])/10.0 # One decimal is good enough
            if pos > self.bars[i].n:
                self.bars[i].update(pos - self.bars[i].n)
            num_data = num_data % self.chunk_sizes[i]

class ExperimentGraph(object):
    def __init__(self, edges, loop):
        self.dag = None
        self.edges = []
        self.loop = loop
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
            obj = DataStream(name="{}_TO_{}".format(edge[0].name, edge[1].name),
                             loop=self.loop)
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

        # This holds a reference to a matplotlib server instance
        # for plotting, if there is one.
        self.matplot_server_thread = None
        # If this is True, don't close the plot server thread so that
        # we might push additional plots after run_sweeps is complete.
        self.leave_plot_server_open = False

        self.keep_instruments_connected = False

        # Also keep references to all of the plot filters
        self.plotters = [] # Standard pipeline plotters using streams
        self.extra_plotters = [] # Plotters using streams, but not the pipeline
        self.manual_plotters = [] # Plotters using neither streams nor the pipeline
        self.manual_plotter_callbacks = [] # These are called at the end of run
        self._extra_plots_to_streams = {}

        # Furthermore, keep references to all of the file writers.
        # If multiple writers request acces to the same filename, they
        # should share the same file object and write in separate
        # hdf5 groups.
        self.writers = []
        self.buffers = []

        # ExpProgressBar object to display progress bars
        self.progressbar = None

        # indicates whether the instruments are already connected
        self.instrs_connected = False

        # indicates whether this is the first (or only) experiment in a series (e.g. for pulse calibrations)
        self.first_exp = True

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

        # Create the asyncio measurement loop
        self.loop = asyncio.get_event_loop()

        # Based on the logging level, infer whether we want asyncio debug
        do_debug = logger.getEffectiveLevel() <= logging.DEBUG
        self.loop.set_debug(do_debug)

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
        self.graph = ExperimentGraph(edges, self.loop)

    def init_streams(self):
        """Establish the base descriptors for any internal data streams and connectors."""
        pass

    def init_instruments(self):
        """Gets run before a sweep starts"""
        pass

    def shutdown_instruments(self):
        """Gets run after a sweep ends, or when the program is terminated."""
        pass

    def init_progressbar(self, num=0, notebook=False):
        """ initialize the progress bars."""
        oc = list(self.output_connectors.values())
        if len(oc)>0:
            self.progressbar = ExpProgressBar(oc[0].output_streams[0], num=num, notebook=notebook)
        else:
            logger.warning("No stream is found for progress bars. Create a dummy bar.")
            self.progressbar = ExpProgressBar(None, num=num, notebook=notebook)

    async def run(self):
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
            # if not self.sweeper.is_adaptive():
            #     oc.descriptor.visited_tuples = oc.descriptor.expected_tuples(with_metadata=True, as_structured_array=False)
            # else:
            oc.descriptor.visited_tuples = []
            oc.update_descriptors()

    async def declare_done(self):
        for oc in self.output_connectors.values():
            for os in oc.output_streams:
                await os.push_event("done")
        for p in self.extra_plotters:
            stream = self._extra_plots_to_streams[p]
            await stream.push_event("done")

    async def sweep(self):
        # Set any static parameters
        static_params = [p for p in self._parameters.values() if p not in self.sweeper.swept_parameters()]
        for p in static_params:
            p.push()

        # Keep track of the previous values
        logger.debug("Waiting for filters.")
        await asyncio.sleep(0.1)
        last_param_values = None
        logger.debug("Starting experiment sweep.")

        done = True
        while True:
            # Increment the sweeper, which returns a list of the current
            # values of the SweepAxes (no DataAxes).
            sweep_values = await self.sweeper.update()

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

            # Run the procedure
            # logger.debug("Starting a new run.")
            await self.run()

            # See if the axes want to extend themselves
            refined_axis = await self.sweeper.check_for_refinement()
            if refined_axis is not None:
                for oc in self.output_connectors.values():
                     await oc.push_event("refined", refined_axis)

            # Update progress bars
            if self.progressbar is not None:
                self.progressbar.update()

            # Finish up, checking to see whether we've received all of our data
            if self.sweeper.done():
                sleep_time = 0
                while not self.filters_finished():
                    await asyncio.sleep(1)
                    sleep_time += 1
                    if sleep_time == 5:
                        logger.info("Still waiting for filters to finish. Did the experiment produce the expected amount of data?")
                        for n in self.nodes:
                            if isinstance(n, Filter):
                                logger.info("  {} done: {}".format(n, n.finished_processing))
                        print({n: n.finished_processing for n in self.nodes if isinstance(n, Filter)})

                    if sleep_time >= 20:
                        logger.warning("Filters not stopped after 20 seconds, bailing.")
                        break
                await self.declare_done()
                break

    def filters_finished(self):
        return all([n.finished_processing for n in self.nodes if isinstance(n, Filter)])

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

    def run_sweeps(self):
        # Propagate the descriptors through the network
        self.update_descriptors()
        # Make sure we are starting from scratch... is this necessary?
        self.reset()
        # Update the progress bar if need be
        if self.progressbar is not None:
            self.progressbar.reset()

        #Make sure we have axes.
        if not any([oc.descriptor.axes for oc in self.output_connectors.values()]):
            logger.warning("There do not appear to be any axes defined for this experiment!")

        # Go find any writers
        self.writers = [n for n in self.nodes if isinstance(n, WriteToHDF5)]
        self.buffers = [n for n in self.nodes if isinstance(n, DataBuffer)]
        if self.name:
            for w in self.writers:
                w.filename.value = os.path.join(os.path.dirname(w.filename.value), self.name)
        self.filenames = [w.filename.value for w in self.writers]
        self.files = []

        # Check for redundancy in filenames, and share plot file objects
        for filename in set(self.filenames):
            wrs = [w for w in self.writers if w.filename.value == filename]

            # Let the first writer with this filename create the file...
            wrs[0].file = wrs[0].new_file()
            self.files.append(wrs[0].file)

            # Make the rest of the writers use this same file object
            for w in wrs[1:]:
                w.file = wrs[0].file
                w.filename.value = wrs[0].filename.value

        # Remove the nodes with 0 dimension
        self.nodes = [n for n in self.nodes if not(hasattr(n, 'input_connectors') and        n.input_connectors['sink'].descriptor.num_dims()==0)]

        # Go and find any plotters
        self.standard_plotters = [n for n in self.nodes if isinstance(n, (Plotter, MeshPlotter, XYPlotter))]
        self.plotters = copy.copy(self.standard_plotters)

        # We might have some additional plotters that are separate from
        # The asyncio filter pipeline
        self.plotters.extend(self.extra_plotters)

        # These use neither streams nor the filter pipeline
        self.plotters.extend(self.manual_plotters)

        # Call any final initialization on the filter pipeline
        for n in self.nodes + self.extra_plotters:
            n.experiment = self
            n.loop       = self.loop
            # n.executor   = self.executor
            if hasattr(n, 'final_init'):
                n.final_init()

        # Launch plot servers.
        if len(self.plotters) > 0:
            self.init_plot_servers()
        time.sleep(1)
        #connect all instruments
        self.connect_instruments()
        #initialize instruments
        self.init_instruments()

        def catch_ctrl_c(signum, frame):
            logger.info("Caught SIGINT. Shutting down.")
            self.shutdown()
            raise NameError("Shutting down.")
            sys.exit(0)

        signal.signal(signal.SIGINT, catch_ctrl_c)

        # We want to wait for the sweep method above,
        # not the experiment's run method, so replace this
        # in the list of tasks.
        other_nodes = self.nodes[:]
        other_nodes.extend(self.extra_plotters)
        other_nodes.remove(self)
        tasks = [n.run() for n in other_nodes]

        tasks.append(self.sweep())
        try:
            self.loop.run_until_complete(asyncio.gather(*tasks))
            self.loop.run_until_complete(asyncio.sleep(1))
        except Exception as e:
            logger.exception("message")

        for plot, callback in zip(self.manual_plotters, self.manual_plotter_callbacks):
            if callback:
                callback(plot)

        self.shutdown()

    def shutdown(self):
        logger.debug("Shutting Down!")
        for f in self.files:
            try:
                logger.debug("Closing %s", f)
                f.close()
                del f
            except:
                logger.debug("File probably already closed...")

        if hasattr(self, 'plot_server'):
            try:
                if len(self.plotters) > 0: #and not self.leave_plot_server_open:
                    self.plot_server.stop()
            except:
                logger.warning("Could not stop plot server gracefully...")

        self.shutdown_instruments()

        if not self.keep_instruments_connected:
            self.disconnect_instruments()

    def add_axis(self, axis):
        for oc in self.output_connectors.values():
            logger.debug("Adding axis %s to connector %s.", axis, oc.name)
            oc.descriptor.add_axis(axis)

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

    async def push_to_plot(self, plotter, data):
        """Push data to a direct plotter."""

        stream = self._extra_plots_to_streams[plotter]
        await stream.push_direct(data)

    def init_plot_servers(self):
        logger.debug("Found %d plotters", len(self.plotters))

        from .plotting import MatplotServerThread
        plot_desc = {p.name: p.desc() for p in self.standard_plotters}
        if not hasattr(self, "plot_server"):
            self.plot_server = MatplotServerThread(plot_desc)
        if len(self.plotters) > len(self.standard_plotters) and not hasattr(self, "extra_plot_server"):
            extra_plot_desc = {p.name: p.desc() for p in self.extra_plotters + self.manual_plotters}
            self.extra_plot_server = MatplotServerThread(extra_plot_desc, status_port = self.plot_server.status_port+2, data_port = self.plot_server.data_port+2)
        for plotter in self.standard_plotters:
            plotter.plot_server = self.plot_server
        for plotter in self.extra_plotters + self.manual_plotters:
            plotter.plot_server = self.extra_plot_server
        time.sleep(0.5)
        # Kill a previous plotter if desired.
        if auspex.config.single_plotter_mode and auspex.config.last_plotter_process:
            pros = [auspex.config.last_plotter_process]
            if (not self.leave_plot_server_open or self.first_exp) and auspex.config.last_extra_plotter_process:
                pros += [auspex.config.last_extra_plotter_process]
            for pro in pros:
                if hasattr(os, 'setsid'): # Doesn't exist on windows
                    try:
                        os.kill(pro.pid, 0) # Raises an error if the PID doesn't exist
                        os.killpg(os.getpgid(pro.pid), signal.SIGTERM) # Proceed to kill process group
                    except OSError:
                        logger.debug("No plotter to kill.")
                else:
                    try:
                        pro.kill()
                    except:
                        logger.debug("No plotter to kill.")

        client_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"matplotlib-client.py")
        #if not auspex.config.last_plotter_process:
        if hasattr(os, 'setsid'):
            auspex.config.last_plotter_process = subprocess.Popen(['python', client_path, 'localhost'],
                                                                env=os.environ.copy(), preexec_fn=os.setsid)
        else:
            auspex.config.last_plotter_process = subprocess.Popen(['python', client_path, 'localhost'],
                                                                env=os.environ.copy())
        if hasattr(self, 'extra_plot_server') and (not auspex.config.last_extra_plotter_process or not self.leave_plot_server_open or self.first_exp):
            if hasattr(os, 'setsid'):
                auspex.config.last_extra_plotter_process = subprocess.Popen(['python', client_path, 'localhost', str(self.extra_plot_server.status_port), str(self.extra_plot_server.data_port)], env=os.environ.copy(), preexec_fn=os.setsid)
            else:
                auspex.config.last_extra_plotter_process = subprocess.Popen(['python', client_path, 'localhost', str(self.extra_plot_server.status_port), str(self.extra_plot_server.data_port)], env=os.environ.copy())
