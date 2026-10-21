# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Plotter', 'ManualPlotter', 'MeshPlotter']

import time
import zmq
import json
import uuid
import sys, os
import numpy as np

from .filter import Filter
from auspex.parameter import Parameter, IntParameter
from auspex.log import logger
from auspex.stream import InputConnector, OutputConnector

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    import threading as mp
    from queue import Queue
else:
    import multiprocessing as mp
    from multiprocessing import Queue

class Plotter(Filter):
    sink      = InputConnector()
    plot_dims = IntParameter(value_range=(0,1,2), snap=1, default=0) # 0 means auto
    plot_mode = Parameter(allowed_values=["real", "imag", "real/imag", "amp/phase", "quad"], default="quad")

    def __init__(self, *args, name="", plot_dims=None, plot_mode=None, **plot_args):
        super(Plotter, self).__init__(*args, name=name)
        if plot_dims:
            self.plot_dims.value = plot_dims
        if plot_mode:
            self.plot_mode.value = plot_mode
        self.plot_args = plot_args
        self.full_update_interval = 0.5
        self.update_interval = 2.0 # slower for partial updates
        self.last_update = time.time()
        self.last_full_update = time.time()

        self._final_buffer = Queue()
        self.final_buffer = None

        self.quince_parameters = [self.plot_dims, self.plot_mode]

        # Unique id for plot server
        self.uuid = None

        # Should we actually produce plots?
        self.do_plotting = True

    def send(self, message):
        if self.do_plotting:
            data = message['data']
            msg  = message['msg']
            name = message['name']

            msg_contents = [self.uuid.encode(), msg.encode(), name.encode()]

            # We might be sending multiple axes, series, etc.
            # Just add them succesively to a multipart message.
            for dat in data:
                md = dict(
                    dtype = str(dat.dtype),
                    shape = dat.shape,
                )
                msg_contents.extend([json.dumps(md).encode(), np.ascontiguousarray(dat)])
            self.socket.send_multipart(msg_contents)

    def get_final_plot(self, quad_funcs=[np.abs, np.angle]):
        if not self.done.is_set():
            raise Exception("Cannot get final plot since plotter is not done or was not run.")

        from bqplot import LinearScale, ColorScale, ColorAxis, Axis, Lines, Figure, Tooltip, HeatMap
        from bqplot.toolbar import Toolbar
        from ipywidgets import VBox, HBox

        if self.final_buffer is None:
            self.final_buffer = self._final_buffer.get()
        if self.plot_dims.value == 2:
            raise NotImplementedError("2 dimensional get_final_plot not yet implemented.")
        elif self.plot_dims.value == 1:
            figs = []
            for quad_func in quad_funcs:
                sx   = LinearScale()
                sy   = LinearScale()
                ax   = Axis(label=self.axis_label(-1), scale=sx)
                ay   = Axis(label=f"{self.descriptor.data_name} ({self.descriptor.data_unit})", scale=sy, orientation='vertical')
                line = Lines(x=self.x_values, y=quad_func(self.final_buffer), scales={'x': sx, 'y': sy})
                fig  = Figure(marks=[line], axes=[ax, ay], title=self.filter_name)
                figs.append(fig)
        if len(figs) <= 2:
            return HBox(figs)
        elif len(figs) == 4:
            return VBox([HBox([figs[0], figs[1]]), HBox([figs[2], figs[3]])])
        elif len(figs) == 3 or len(figs) > 4:
            raise Exception("Please use 1, 2, or 4 quadrature functions.")


    def desc(self):
        d =    {'plot_type': 'standard',
                'plot_mode': self.plot_mode.value,
                'plot_dims': int(self.plot_dims.value),
                'x_min':     float(min(self.x_values)),
                'x_max':     float(max(self.x_values)),
                'x_len':     int(self.descriptor.axes[-1].num_points()),
                'x_label':   self.axis_label(-1),
                'y_label':   "{} ({})".format(self.descriptor.data_name, self.descriptor.data_unit)
                }
        if self.plot_dims.value == 2:
            d['y_label']    = self.axis_label(-2)
            d['data_label'] = "{} ({})".format(self.descriptor.data_name, self.descriptor.data_unit)
            d['y_min']      = float(min(self.y_values))
            d['y_max']      = float(max(self.y_values))
            d['y_len']      = int(self.descriptor.axes[-2].num_points())
        return d

    def set_done(self):
        self.send({'name': self.filter_name, 'data': [np.array([])], "msg": "done"})

    def set_quit(self):
        self.send({'name': self.filter_name, 'data': [np.array([])], "msg": "quit"})

    def update_descriptors(self):
        logger.debug("Updating Plotter %s descriptors based on input descriptor %s", self.filter_name, self.sink.descriptor)
        self.stream = self.sink.input_streams[0]
        self.descriptor = self.sink.descriptor

    def final_init(self):
        # Determine the plot dimensions
        if not self.plot_dims.value:
            if len(self.descriptor.axes) > 1:
                self.plot_dims.value = 2
            else:
                self.plot_dims.value = 1

        # Check the descriptor axes
        num_axes = len(self.descriptor.axes)
        if self.plot_dims.value > num_axes:
            logger.info("Cannot plot in more dimensions than there are data axes.")
            self.plot_dims.value = num_axes

        if self.plot_dims.value == 1:
            self.points_before_clear = self.descriptor.axes[-1].num_points()
        else:
            self.points_before_clear = self.descriptor.axes[-1].num_points() * self.descriptor.axes[-2].num_points()
        logger.debug("Plot will clear after every %d points.", self.points_before_clear)

        self.x_values = self.descriptor.axes[-1].points

        if self.plot_dims.value == 2:
            self.y_values = self.descriptor.axes[-2].points

        #I'm so sorry everyone. Send Julia
        if 'complex' in np.dtype(self.descriptor.dtype).name:
            self.plot_buffer = (np.nan*np.ones(self.points_before_clear) + 1.0j*np.nan*np.ones(self.points_before_clear)).astype(self.descriptor.dtype)
        else:
            self.plot_buffer = np.nan*np.ones(self.points_before_clear)
        self.idx = 0

    def execute_on_run(self):
        # Connect to the plot server
        if self.do_plotting:
            try:
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.DEALER)
                self.socket.identity = f"Auspex_Experiment_{self.filter_name}_{hex(id(self))}".encode()
                self.socket.connect("tcp://localhost:7762")
            except:
                logger.warning("Exception occured while contacting the plot server. Is it running?")

    def update(self):
        if self.plot_dims.value == 1:
            self.send({'name': self.filter_name, 'msg':'data', 'data': [self.x_values, self.plot_buffer.copy()]})
        elif self.plot_dims.value == 2:
            self.send({'name': self.filter_name, 'msg':'data', 'data': [self.x_values, self.y_values, self.plot_buffer.copy()]})

    def process_data(self, data):
        # If we get more than enough data, pause to update the plot if necessary
        if (self.idx + data.size) > self.points_before_clear:
            spill_over = (self.idx + data.size) % self.points_before_clear
            if spill_over == 0:
                spill_over = self.points_before_clear
            if (time.time() - self.last_full_update >= self.full_update_interval):
                # If we are getting data quickly, then we can afford to wait
                # for a full frame before pushing to plot.
                self.plot_buffer[self.idx:] = data[:(self.points_before_clear-self.idx)]
                self.update()
                self.last_full_update = time.time()
            self.plot_buffer[:] = np.nan
            self.plot_buffer[:spill_over] = data[-spill_over:]
            self.idx = spill_over
        else: # just keep trucking
            self.plot_buffer[self.idx:self.idx+data.size] = data.flatten()
            self.idx += data.size
            if (time.time() - max(self.last_full_update, self.last_update) >= self.update_interval):
                self.update()
                self.last_update = time.time()

    def on_done(self):
        if self.plot_dims.value == 1:
            self.send({'name': self.filter_name, "msg": "data", 'data': [self.x_values, self.plot_buffer.copy()], })
        elif self.plot_dims.value == 2:
            self.send({'name': self.filter_name, "msg": "data", 'data': [self.x_values, self.y_values, self.plot_buffer.copy()]})
        self._final_buffer.put(self.plot_buffer)
        if self.do_plotting:
            self.set_done()
            self.socket.close()
            self.context.term()

    def axis_label(self, index):
        unit_str = " ({})".format(self.descriptor.axes[index].unit) if self.descriptor.axes[index].unit else ''
        return self.descriptor.axes[index].name + unit_str

class MeshPlotter(Filter):
    sink = InputConnector()
    plot_mode = Parameter(allowed_values=["real", "imag", "real/imag", "amp/phase", "quad"], default="quad")

    def __init__(self, *args, name="", plot_mode=None, x_label="", y_label="", **plot_args):
        super(MeshPlotter, self).__init__(*args, name=name)
        if plot_mode:
            self.plot_mode.value = plot_mode
        self.plot_args = plot_args
        self.update_interval = 0.5
        self.last_update = time.time()
        self.x_label = x_label
        self.y_label = y_label

        self.quince_parameters = [self.plot_mode]

        # Unique id for plot server
        self.uuid = None

        # Should we actually produce plots?
        self.do_plotting = True

    def desc(self):
        d =    {'plot_type': 'mesh',
                'plot_mode': self.plot_mode.value,
                'x_label':   self.x_label,
                'y_label':   self.y_label
                }
        return d

    def send(self, message):
        if self.do_plotting:
            data = message['data']
            msg  = message['msg']
            name = message['name']

            msg_contents = [self.uuid.encode(), msg.encode(), name.encode()]

            # We might be sending multiple axes, series, etc.
            # Just add them succesively to a multipart message.
            for dat in data:
                md = dict(
                    dtype = str(dat.dtype),
                    shape = dat.shape,
                )
                msg_contents.extend([json.dumps(md).encode(), np.ascontiguousarray(dat)])
            self.socket.send_multipart(msg_contents)


    def update_descriptors(self):
        logger.info("Updating MeshPlotter %s descriptors based on input descriptor %s", self.filter_name, self.sink.descriptor)

    def execute_on_run(self):
        # Connect to the plot server
        if self.do_plotting:
            try:
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.DEALER)
                self.socket.identity = "Auspex_Experiment".encode()
                self.socket.connect("tcp://localhost:7762")
            except:
                logger.warning("Exception occured while contacting the plot server. Is it running?")

    def process_direct(self, data):
        self.send({'name': self.filter_name, "msg":"data", 'data': [self.plot_buffer.copy()]})

    def on_done(self):
        self.send({'name': self.filter_name, 'data': [np.array([])], "msg": "done"})
        if self.do_plotting:
            self.socket.close()
            self.context.term()

class ManualPlotter(object):
    """
    Establish a figure, then give the user complete control over plot creation and data. There isn't any reason to
    run this as a process, but we provide the same interface for convenience.
    """
    def __init__(self,  name="", x_label=['X'], y_label=["y"], y_lim=None, numplots = 1):
        self.x_label      = x_label if type(x_label) == list else [x_label]
        self.y_label      = y_label if type(y_label) == list else [y_label]
        self.y_lim        = y_lim
        self.filter_name  = name
        self.numplots     = numplots
        self.traces       = []

        # Calls block so this won't ever hang
        self.done = True

        # Unique id for plot server
        self.uuid = None

        # Should we actually produce plots?
        self.do_plotting = True

    def start(self):
        self.execute_on_run()

    def stop(self):
        if self.do_plotting:
            try:
                self.socket.close()
                self.context.term()
            except:
                logger.warning(f"Exception occured while closing socket and context for {self}")

    def send(self, message):
        if self.do_plotting:
            data = message['data']
            msg  = message['msg']
            name = message['name']
            msg_contents = [self.uuid.encode(), msg.encode(), name.encode()]

            # We might be sending multiple axes, series, etc.
            # Just add them succesively to a multipart message.
            for dat in data:
                md = dict(
                    dtype = str(dat.dtype),
                    shape = dat.shape,
                )
                msg_contents.extend([json.dumps(md).encode(), np.ascontiguousarray(dat)])
            self.socket.send_multipart(msg_contents)

    def execute_on_run(self):
        # Connect to the plot server
        if self.do_plotting:
            try:
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.DEALER)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.identity = f"Auspex_Experiment {str(uuid.uuid4())}".encode()
                self.socket.connect("tcp://localhost:7762")
            except:
                logger.warning("Exception occured while contacting the plot server. Is it running?")

    def add_trace(self, name, matplotlib_kwargs={}, subplot_num = 0):
        self.traces.append({'name': name, 'axis_num' : subplot_num, 'matplotlib_kwargs': matplotlib_kwargs})

    def add_fit_trace(self, name, custom_mpl_kwargs={}, subplot_num = 0):
        matplotlib_kwargs={'linestyle': '-', 'linewidth': 2}
        matplotlib_kwargs.update(custom_mpl_kwargs)
        self.add_trace(name, matplotlib_kwargs=matplotlib_kwargs, subplot_num=subplot_num)

    def add_data_trace(self, name, custom_mpl_kwargs={}, subplot_num = 0):
        matplotlib_kwargs={'linestyle': ':', 'marker': '.'}
        matplotlib_kwargs.update(custom_mpl_kwargs)
        self.add_trace(name, matplotlib_kwargs=matplotlib_kwargs, subplot_num=subplot_num)

    def desc(self):
        d =    {'plot_type': 'manual',
                'x_label':   self.x_label,
                'y_label':   self.y_label,
                'y_lim':     self.y_lim,
                'numplots':  self.numplots,
                'traces':    self.traces
                }
        return d

    def __setitem__(self, trace_name, data_tuple):
        if trace_name not in [t['name'] for t in self.traces]:
            raise KeyError("Trace {} does not exist in this plotter.".format(trace_name))
        if len(data_tuple) != 2:
            raise ValueError("__setitem__ for ManualPlotter accepts a tuple of length 2 for (xdata, ydata)")
        self.set_data(trace_name, data_tuple[0], data_tuple[1])

    def set_done(self):
        self.send({'name': self.filter_name, 'data': [np.array([])], "msg": "done"})

    def set_quit(self):
        self.send({'name': self.filter_name, 'data': [np.array([])], "msg": "quit"})

    def set_data(self, trace_name, xdata, ydata):
        self.send({"name": self.filter_name + ":" + trace_name, "msg": "data", "data": [xdata, ydata]})
