# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Plotter', 'ManualPlotter', 'XYPlotter', 'MeshPlotter']

import time
import numpy as np

from .filter import Filter
from auspex.parameter import Parameter, IntParameter
from auspex.log import logger
from auspex.stream import InputConnector, OutputConnector

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

        self.quince_parameters = [self.plot_dims, self.plot_mode]

        # This will hold the matplot server
        self.plot_server = None

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


    def update_descriptors(self):
        logger.debug("Updating Plotter %s descriptors based on input descriptor %s", self.name, self.sink.descriptor)
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

        self.plot_buffer = (np.nan*np.ones(self.points_before_clear)).astype(self.descriptor.dtype)
        self.idx = 0

    def update(self):
        if self.plot_dims.value == 1:
            self.plot_server.send(self.name, self.x_values, self.plot_buffer.copy())
        elif self.plot_dims.value == 2:
            self.plot_server.send(self.name, self.x_values, self.y_values, self.plot_buffer.copy())

    async def process_data(self, data):
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

    async def on_done(self):
        if self.plot_dims.value == 1:
            self.plot_server.send(self.name, self.x_values, self.plot_buffer)
        elif self.plot_dims.value == 2:
            self.plot_server.send(self.name, self.x_values, self.y_values, self.plot_buffer)

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

        # This will hold the matplot server
        self.plot_server = None

    def desc(self):
        d =    {'plot_type': 'mesh',
                'plot_mode': self.plot_mode.value,
                'x_label':   self.x_label,
                'y_label':   self.y_label
                }
        return d

    def update_descriptors(self):
        logger.info("Updating MeshPlotter %s descriptors based on input descriptor %s", self.name, self.sink.descriptor)

    def final_init(self):
        pass

    async def process_direct(self, data):
        self.plot_server.send(self.name, data)

    async def on_done(self):
        self.plot_server.send(self.name, np.array([]), msg="done")
        time.sleep(0.1)

class XYPlotter(Filter):
    sink_x = InputConnector()
    sink_y = InputConnector()

    def __init__(self, *args, name="", x_series=False, y_series=False, series="inner", notebook=False, webgl=False, **plot_args):
        """Theyintent is to let someone plot this vs. that from different streams."""
        super(XYPlotter, self).__init__(*args, name=name)

        self.plot_args       = plot_args
        self.update_interval = 0.5
        self.last_update     = time.time()
        self.run_in_notebook = notebook
        self.x_series        = x_series
        self.y_series        = y_series or self.x_series
        self.plot_height     = 600
        self.series          = series
        self.webgl           = webgl

        self.quince_parameters = []

    def update_descriptors(self):
        logger.debug("Updating XYPlotter %s descriptors.", self.name)
        self.stream_x = self.sink_x.input_streams[0]
        self.stream_y = self.sink_y.input_streams[0]
        self.desc_x   = self.sink_x.descriptor
        self.desc_y   = self.sink_y.descriptor

    def final_init(self):
        # Check the dimensions to ensure compatibility
        if self.desc_x.axes[-1].num_points() != self.desc_y.axes[-1].num_points():
            raise ValueError("XYPlotter x and y final axis lengths must match")
        if self.x_series and self.y_series:
            if self.desc_x.axes[-2].num_points() != self.desc_y.axes[-2].num_points():
                raise ValueError("XYPlotter x and y second-to-last axis lengths must match when plotting series.")

        if len(self.desc_x.axes) == 1 and len(self.desc_y.axes) == 1:
            series_axis = 0
            data_axis = 0
        else:
            if self.series == "inner":
                series_axis = -2
                data_axis = -1
            elif self.series == "outer":
                series_axis = 0
                data_axis = 1
            else:
                raise ValueError("series must be either inner or outer")

        # How many points before clear
        self.points_before_clear_y = self.desc_y.axes[data_axis].num_points()
        self.points_before_clear_x = self.desc_x.axes[data_axis].num_points()

        if self.x_series:
            x_data = [[] for x in range(self.desc_x.axes[series_axis].num_points())]
            self.points_before_clear_x *= self.desc_x.axes[series_axis].num_points()
        else:
            x_data = [[]]
        if self.y_series:
            y_data = [[] for y in range(self.desc_y.axes[series_axis].num_points())]
            self.points_before_clear_y *= self.desc_y.axes[series_axis].num_points()
            self.num_series = self.desc_y.axes[series_axis].num_points()
        else:
            y_data = [[]]

        x_label = "{} ({})".format(self.desc_x.data_name, self.desc_x.data_unit)
        y_label = "{} ({})".format(self.desc_y.data_name, self.desc_y.data_unit)

        self.fig = Figure(plot_width=self.plot_height, plot_height=self.plot_height, webgl=self.webgl,
                          x_axis_label=x_label, y_axis_label=y_label)

        if self.desc_y.axes[series_axis].num_points() <= 10:
            self.colors = d3['Category10'][self.desc_y.axes[series_axis].num_points()]
        elif self.desc_y.axes[series_axis].num_points() <= 20:
            self.colors = d3['Category20'][self.desc_y.axes[series_axis].num_points()]
        else:
            self.colors = Viridis256[:self.desc_y.axes[series_axis].num_points()]

        self.plot = self.fig.multi_line(x_data, y_data, name=self.name,
                                        line_width=2, color=self.colors,
                                        **self.plot_args)

        renderers = self.plot.select(dict(name=self.name))
        self.renderer = [r for r in renderers if isinstance(r, GlyphRenderer)][0]
        self.data_source = self.renderer.data_source

        self.plot_buffer_x = np.nan*np.ones(self.points_before_clear_x, dtype=self.desc_x.dtype)
        self.plot_buffer_y = np.nan*np.ones(self.points_before_clear_y, dtype=self.desc_y.dtype)
        self.idx = 0
        self.idy = 0

    async def run(self):
        while True:
            # Wait for all of the acquisition to complete, avoid asyncio.wait because of random return order...
            message_x = await self.stream_x.queue.get()
            message_y = await self.stream_y.queue.get()
            messages = [message_x, message_y]

            # Ensure we aren't getting different types of messages at the same time.
            message_types = [m['type'] for m in messages]
            try:
                if len(set(message_types)) > 1:
                    raise ValueError("Writer received concurrent messages with different message types {}".format([m['type'] for m in messages]))
            except:
                import ipdb; ipdb.set_trace()

            # Infer the type from the first message
            message_type = messages[0]['type']

            # If we receive a message
            if message_type == 'event':
                logger.debug('%s "%s" received event "%s"', self.__class__.__name__, self.name, message_type)
                if messages[0]['event_type'] == 'done':
                    break

            elif message_type == 'data':
                message_data = [message['data'] for message in messages]
                message_comp = [message['compression'] for message in messages]
                message_data = [pickle.loads(zlib.decompress(dat)) if comp == 'zlib' else dat for comp, dat in zip(message_comp, message_data)]
                message_data = [dat if hasattr(dat, 'size') and dat.size != 1 else np.array([dat]) for dat in message_data]  # Convert single values to arrays

                data_x, data_y = message_data

                # if we're going to clear then reset idy
                if self.idy + data_y.size > self.points_before_clear_y:
                    logger.debug("Clearing previous plot and restarting")
                    spill_over = (self.idy + data_y.size) % self.points_before_clear_y
                    if spill_over == 0:
                        spill_over = self.points_before_clear_y
                    self.plot_buffer_y[:] = np.nan
                    self.plot_buffer_y[:spill_over] = data_y[-spill_over:]
                    self.idy = spill_over
                else:
                    self.plot_buffer_y[self.idy:self.idy+data_y.size] = data_y.flatten()
                    self.idy += data_y.size

                # if we're going to clear then reset idy
                if self.idx + data_x.size > self.points_before_clear_x:
                    logger.debug("Clearing previous plot and restarting")
                    spill_over = (self.idx + data_x.size) % self.points_before_clear_x
                    if spill_over == 0:
                        spill_over = self.points_before_clear_x
                    self.plot_buffer_x[:] = np.nan
                    self.plot_buffer_x[:spill_over] = data_x[-spill_over:]
                    self.idx = spill_over
                else:
                    self.plot_buffer_x[self.idx:self.idx+data_x.size] = data_x.flatten()
                    self.idx += data_x.size

                # Assume that the x data is synched to the y data (they arrive at the same time...)

                if (time.time() - self.last_update >= self.update_interval):
                    if self.x_series:
                        x_data = np.reshape(self.plot_buffer_x, (self.num_series, -1)).T
                    elif self.y_series:
                        x_data = np.tile(self.plot_buffer_x, (self.num_series, 1))
                    else:
                        x_data = [self.plot_buffer_x]

                    if self.y_series:
                        y_data = np.reshape(self.plot_buffer_y, (self.num_series, -1)).T
                    else:
                        y_data = [self.plot_buffer_y]

                    # Strip NaNs
                    x_data = np.array([series[~np.isnan(series)] for series in x_data])
                    y_data = np.array([series[~np.isnan(series)] for series in y_data])

                    # Convert to lists and then push all at once...
                    self.data_source.data = dict(xs=x_data.tolist(), ys=y_data.tolist(), line_color=self.colors[0:len(y_data)])

                    self.last_update = time.time()

    # async def on_done(self):
    #     for i,j in zip(self.mapping_functions, self.data_sources):
    #         for mapping_function, data_source in zip(i,j):
    #             data_source.data["y"] = np.copy(mapping_function(self.plot_buffer))
    #     time.sleep(0.1)

class ManualPlotter(object):
    """Establish a figure, then give the user complete control over plot creation and data."""
    def __init__(self,  name="", x_label=['X'], y_label=["y"], numplots = 1):
        self.x_label      = x_label if type(x_label) == list else [x_label]
        self.y_label      = y_label if type(y_label) == list else [y_label]
        self.name         = name
        self.numplots     = numplots
        self.traces = []

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

    def set_data(self, trace_name, xdata, ydata):
        self.plot_server.send(self.name + ":" + trace_name, xdata, ydata)
