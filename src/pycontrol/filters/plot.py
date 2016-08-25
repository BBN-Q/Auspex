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
from bokeh.models.renderers import GlyphRenderer

from pycontrol.logging import logger
from pycontrol.filters.filter import Filter, InputConnector

class Plotter(Filter):
    data = InputConnector()

    def __init__(self, *args, name="", plot_dims=1, **plot_args):

        super(Plotter, self).__init__(*args, name=name)
        self.plot_dims = plot_dims
        self.plot_args = plot_args
        self.update_interval = 0.5
        self.last_update = time.time()

    def update_descriptors(self):
        logger.info("Updating Plotter %s descriptors based on input descriptor %s", self.name, self.data.descriptor)
        self.stream = self.data.input_streams[0]
        self.descriptor = self.data.descriptor

    def final_init(self):

        # Check the descriptor axes
        num_axes = len(self.descriptor.axes)
        if self.plot_dims > num_axes:
            raise Exception("Cannot plot in more dimensions than there are data axes.")

        if self.plot_dims == 1:
            self.points_before_clear = self.descriptor.axes[-1].num_points()
        else:
            self.points_before_clear = self.descriptor.axes[-1].num_points() * self.descriptor.axes[-2].num_points()
        logger.info("Plot will clear after every %d points.", self.points_before_clear)

        self.x_values = self.descriptor.axes[-1].points
        xmax = max(self.x_values)
        xmin = min(self.x_values)

        if self.plot_dims == 1:
            self.figure = Figure(x_range=[xmin, xmax], plot_width=600, plot_height=600, webgl=False)
            self.plot = self.figure.line(np.copy(self.x_values), np.nan*np.ones(self.points_before_clear), name=self.name, **self.plot_args)
        else:
            self.y_values = self.descriptor.axes[-2].points
            self.x_mesh, self.y_mesh = np.meshgrid(self.x_values, self.y_values)
            self.z_data = np.zeros_like(self.x_mesh)
            ymax = max(self.y_values)
            ymin = min(self.y_values)
            self.figure = Figure(x_range=[xmin, xmax], y_range=[ymin, ymax], plot_width=600, plot_height=600, webgl=False)
            self.plot = self.figure.image(image=[self.z_data], x=[xmin], y=[ymin],
                                          dw=[xmax-xmin], dh=[ymax-ymin], name=self.name, **self.plot_args)

        self.data_source = self.plot.data_source

        self.plot_buffer = np.nan*np.ones(self.points_before_clear)
        self.idx = 0

    async def process_data(self, data):
        #if we're going to clear then reset idx
        if self.idx + data.size > self.points_before_clear:
            logger.debug("Clearing previous plot and restarting")
            self.plot_buffer[:] = np.nan
            num_prev_buffer_pts = self.points_before_clear - self.idx
            data = data[num_prev_buffer_pts:]
            self.idx = 0

        self.plot_buffer[self.idx:self.idx+data.size] = data.flatten()
        self.idx += data.size

        if self.plot_dims == 1:
            if (time.time() - self.last_update >= self.update_interval):
                self.data_source.data["y"] = np.copy(self.plot_buffer)
                self.last_update = time.time()

        else:
            if (time.time() - self.last_update >= self.update_interval):
                self.data_source.data["image"] = [np.reshape(self.plot_buffer, self.z_data.shape)]
                self.last_update = time.time()

    async def on_done(self):
        if self.plot_dims == 1:
            self.data_source.data["y"] = np.copy(self.plot_buffer)
        else:
            self.data_source.data["image"] = [np.reshape(self.plot_buffer, self.z_data.shape)]
        time.sleep(1.0)
