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
        # self.x_axis = None # Can be string on numerical index
        # self.y_axis = None
        self.update_interval = 0.25
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
            self.plot = self.figure.line([],[], name=self.name, **self.plot_args)
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
        temp = np.empty(self.stream.num_points())

        while True:

            new_data = np.array(await self.stream.queue.get()).flatten()
            temp[idx:idx+new_data.size] = new_data
            idx += new_data.size
            logger.debug("Plotter received %d points.", new_data.size)
            
            # Clear the plots after accumulating a certain number of points
            num_traces  = int(idx/self.points_before_clear)
            extra       = idx - num_traces*self.points_before_clear

            if self.plot_dims == 1:
                if extra == 0:
                    # Plot the last full trace
                    temp[0:self.points_before_clear] = temp[(num_traces-1)*self.points_before_clear:num_traces*self.points_before_clear] 
                    idx = self.points_before_clear
                else:
                    temp[0:extra] = temp[num_traces*self.points_before_clear:num_traces*self.points_before_clear + extra]
                    idx = extra
                      
                if (time.time() - self.last_update >= self.update_interval) or self.stream.done():
                    self.data_source.data["x"] = np.copy(self.x_values[0:idx])
                    self.data_source.data["y"] = np.copy(temp[0:idx])
                    self.last_update = time.time()

            else:
                if extra == 0:
                    temp[0:self.points_before_clear] = temp[(num_traces-1)*self.points_before_clear:num_traces*self.points_before_clear] 
                    idx = self.points_before_clear
                else:
                    temp[0:extra] = temp[num_traces*self.points_before_clear:num_traces*self.points_before_clear + extra]
                    temp[extra:] = 0.0
                    idx = extra
                
                if (time.time() - self.last_update >= self.update_interval) or self.stream.done():
                    self.data_source.data["image"] = [np.reshape(temp[:self.points_before_clear], self.z_data.shape)]
                    self.last_update = time.time()

            if self.stream.done():
                print("No more data for plotter")
                break

