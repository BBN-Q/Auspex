# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio, concurrent
import time

import numpy as np

from bokeh.plotting import Figure
from bokeh.layouts import gridplot
from bokeh.models.renderers import GlyphRenderer

from auspex.parameter import Parameter, IntParameter
from auspex.log import logger
from auspex.filters.filter import Filter, InputConnector
import matplotlib.pyplot as plt

class Plotter(Filter):
    sink      = InputConnector()
    plot_dims = IntParameter(value_range=(0,1,2), snap=1, default=0) # 0 means auto
    plot_mode = Parameter(allowed_values=["real", "imag", "real/imag", "amp/phase", "quad"], default="real/imag")

    def __init__(self, *args, name="", plot_dims=None, plot_mode=None, notebook=False, **plot_args):
        super(Plotter, self).__init__(*args, name=name)
        if plot_dims:
            self.plot_dims.value = plot_dims
        if plot_mode:
            self.plot_mode.value = plot_mode
        self.plot_args = plot_args
        self.update_interval = 0.5
        self.last_update = time.time()
        self.run_in_notebook = notebook

        self.quince_parameters = [self.plot_dims, self.plot_mode]

    def update_descriptors(self):
        logger.info("Updating Plotter %s descriptors based on input descriptor %s", self.name, self.sink.descriptor)
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
            raise Exception("Cannot plot in more dimensions than there are data axes.")

        if self.plot_dims.value == 1:
            self.points_before_clear = self.descriptor.axes[-1].num_points()
        else:
            self.points_before_clear = self.descriptor.axes[-1].num_points() * self.descriptor.axes[-2].num_points()
        logger.info("Plot will clear after every %d points.", self.points_before_clear)

        self.x_values = self.descriptor.axes[-1].points
        xmax = max(self.x_values)
        xmin = min(self.x_values)

        # Establish how the data will be mapped to multiple subplots. Each
        # top level list element will become a row, and subelements will
        # become columns. A single plot_buffer is used to store data, and it
        # will be cast according to these functions.

        if self.plot_mode.value == "real":
            self.mapping_functions = [[np.real]]
        elif self.plot_mode.value == "imag":
            self.mapping_functions = [[np.imag]]
        elif self.plot_mode.value == "real/imag":
            self.mapping_functions = [[np.real, np.imag]]
        elif self.plot_mode.value == "amp/phase":
            self.mapping_functions = [[np.abs, lambda x: np.angle(x, deg=1)]]
        elif self.plot_mode.value =="quad":
            self.mapping_functions = [[np.abs, lambda x: np.angle(x, deg=1)],[np.real, np.imag]]

        if self.plot_dims.value == 1:
            self.figures = [[Figure(x_range=[xmin, xmax], plot_width=600, plot_height=600, webgl=False) for col in row] for row in self.mapping_functions]
            self.plots = [[fig.line(np.copy(self.x_values), np.nan*np.ones(self.points_before_clear), name=self.name) for fig in row] for row in self.figures]
        else:
            self.y_values = self.descriptor.axes[-2].points
            self.x_mesh, self.y_mesh = np.meshgrid(self.x_values, self.y_values)
            self.z_data = np.zeros_like(self.x_mesh)
            ymax = max(self.y_values)
            ymin = min(self.y_values)
            self.figures = [[Figure(x_range=[xmin, xmax], y_range=[ymin, ymax], plot_width=600, plot_height=600, webgl=False) for col in row] for row in self.mapping_functions]
            self.plots = [[fig.image(image=[self.z_data], x=[xmin], y=[ymin],
                                          dw=[xmax-xmin], dh=[ymax-ymin], name=self.name, palette="Spectral11") for fig in row] for row in self.figures]

        # Construct the master gridplot
        self.plot = gridplot(self.figures)

        self.data_sources = [[plot.data_source for plot in row] for row in self.plots]
        self.plot_buffer = np.nan*np.ones(self.points_before_clear, dtype=self.descriptor.dtype)
        self.idx = 0

    async def process_data(self, data):
        #if we're going to clear then reset idx
        if self.idx + data.size > self.points_before_clear:
            logger.debug("Clearing previous plot and restarting")
            spill_over = (self.idx + data.size) % self.points_before_clear
            if spill_over == 0:
                spill_over = self.points_before_clear
            self.plot_buffer[:] = np.nan
            self.plot_buffer[:spill_over] = data[-spill_over:]
            self.idx = spill_over
        else:
            self.plot_buffer[self.idx:self.idx+data.size] = data.flatten()
            self.idx += data.size

        if self.plot_dims.value == 1:
            if (time.time() - self.last_update >= self.update_interval):
                for i,j in zip(self.mapping_functions, self.data_sources):
                    for mapping_function, data_source in zip(i,j):
                        data_source.data["y"] = np.copy(mapping_function(self.plot_buffer))
                self.last_update = time.time()

        else:
            if (time.time() - self.last_update >= self.update_interval):
                for i,j in zip(self.mapping_functions, self.data_sources):
                    for mapping_function, data_source in zip(i,j):
                        data_source.data["image"] = [np.reshape(mapping_function(self.plot_buffer), self.z_data.shape)]
                self.last_update = time.time()

    async def on_done(self):
        if self.plot_dims.value == 1:
            for i,j in zip(self.mapping_functions, self.data_sources):
                for mapping_function, data_source in zip(i,j):
                    data_source.data["y"] = np.copy(mapping_function(self.plot_buffer))
        else:
            for i,j in zip(self.mapping_functions, self.data_sources):
                for mapping_function, data_source in zip(i,j):
                    data_source.data["image"] = [np.reshape(mapping_function(self.plot_buffer), self.z_data.shape)]
        time.sleep(1.0)

class MeshPlotter(Filter):
    sink = InputConnector()

    def __init__(self, *args, name="", plot_mode='real', notebook=False, **plot_args):
        super(MeshPlotter, self).__init__(*args, name=name)
        self.plot_mode = plot_mode
        self.plot_args = plot_args
        self.update_interval = 0.5
        self.last_update = time.time()
        self.run_in_notebook = notebook

    def update_descriptors(self):
        logger.info("Updating MeshPlotter %s descriptors based on input descriptor %s", self.name, self.sink.descriptor)
        self.stream = self.sink.input_streams[0]
        self.descriptor = self.sink.descriptor

    def final_init(self):
        # This should be a set of 2D coordinate tuples
        if hasattr(self, 'descriptor'):
            self.x_values = self.descriptor.axes[-1].points[:,0]
            self.y_values = self.descriptor.axes[-1].points[:,1]
        else:
            self.x_values = [0,10]
            self.y_values = [0,10]

        xmax = max(self.x_values)
        xmin = min(self.x_values)
        ymax = max(self.y_values)
        ymin = min(self.y_values)
        # self.figure = Figure(x_range=[xmin, xmax], plot_width=600, plot_height=600, webgl=False)
        # self.plot = self.figure.line(np.copy(np.linspace(0,10,10)), np.random.random(10), name=self.name)
        self.figure = Figure(x_range=[xmin, xmax], y_range=[ymin, ymax], plot_width=600, plot_height=600, webgl=False)
        self.plot   = self.figure.patches(xs=[[xmin, xmax, xmin],[xmin, xmax, xmax]],
                                          ys=[[ymin, ymin, ymax],[ymax, ymax, xmin]],
                                          fill_color=["#000000","#000000"],
                                          line_color=None)
        self.data_source = self.plot.data_source


    async def process_direct(self, data):
        xs, ys, vals = data
        vals = np.array(vals)
        xs = [list(el) for el in xs]
        ys = [list(el) for el in ys]
        vals   -= vals.min()
        vals   /= vals.max()
        colors = [tuple(el)[:3] for el in plt.cm.RdGy(vals)]
        colors = ["#%02x%02x%02x" % (int(255*color[0]), int(255*color[1]), int(255*color[2])) for color in colors]
        self.figure.x_range.start = np.min(xs)
        self.figure.x_range.end = np.max(xs)
        self.figure.y_range.start = np.min(ys)
        self.figure.y_range.end = np.max(ys)
        self.data_source.data = {'xs': xs, 'ys': ys, 'fill_color': colors}

    async def on_done(self):
        time.sleep(0.5)
