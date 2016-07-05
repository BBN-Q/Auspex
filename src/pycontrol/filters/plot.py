import asyncio
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
        self.stream = self.data.input_streams[0]
        self.descriptor = self.data.descriptor

        logger.info("Starting descriptor update in filter %s, where the descriptor is %s",
                self.name, self.descriptor)

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
            self.figure = Figure(x_range=[xmin, xmax], plot_width=600, plot_height=600, webgl=True)
            self.plot = self.figure.line(np.copy(self.x_values), np.nan*np.ones(self.points_before_clear), name=self.name, **self.plot_args)
        else:
            self.y_values = self.descriptor.axes[-2].points
            self.x_mesh, self.y_mesh = np.meshgrid(self.x_values, self.y_values)
            self.z_data = np.zeros_like(self.x_mesh)
            ymax = max(self.y_values)
            ymin = min(self.y_values)
            self.figure = Figure(x_range=[xmin, xmax], y_range=[ymin, ymax], plot_width=600, plot_height=600, webgl=True)
            self.plot = self.figure.image(image=[self.z_data], x=[xmin], y=[ymin],
                                          dw=[xmax-xmin], dh=[ymax-ymin], name=self.name, **self.plot_args)

        renderers = self.plot.select(dict(name=self.name))
        self.renderer = [r for r in renderers if isinstance(r, GlyphRenderer)][0]
        self.data_source = self.renderer.data_source

    async def run(self):
        idx = 0
        plot_buffer = np.nan*np.ones(self.points_before_clear)

        while True:

            new_data = np.array(await self.stream.queue.get()).flatten()
            logger.debug('Plotter "%s" received %d points.', self.name, new_data.size)
            #if we're going to clear then reset idx
            if idx + new_data.size > self.points_before_clear:
                logger.debug("Clearing previous plot and restarting")
                plot_buffer[:] = np.nan
                num_prev_buffer_pts = self.points_before_clear - idx
                new_data = new_data[num_prev_buffer_pts:]
                idx = 0

            plot_buffer[idx:idx+new_data.size] = new_data
            idx += new_data.size

            if self.plot_dims == 1:
                if (time.time() - self.last_update >= self.update_interval) or self.stream.done():
                    self.data_source.data["y"] = np.copy(plot_buffer)
                    self.last_update = time.time()

            else:
                if (time.time() - self.last_update >= self.update_interval) or self.stream.done():
                    self.data_source.data["image"] = [np.reshape(plot_buffer, self.z_data.shape)]
                    self.last_update = time.time()

            if self.stream.done():
                print("No more data for plotter")
                await asyncio.sleep(1) # wait a second for plot server to do final update
                break
