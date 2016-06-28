import asyncio
from bokeh.plotting import figure
from bokeh.client import push_session
from bokeh.plotting import curdoc
from bokeh.driving import cosine

import threading
import subprocess
import psutil

from pycontrol.logging import logger
from pycontrol.filters.filter import Filter, InputConnector

class Plotter(Filter):
    data = InputConnector() 

    def __init__(self, *args, **kwargs):
        super(Plotter, self).__init__(*args, **kwargs)
        self.plots_dims = 0
        
    def update_descriptors(self):
        self.stream = list(self.input_connectors.values())[0]
        self.descriptor = self.stream.descriptor
        logger.debug("Starting descriptor update in filter %s, where the descriptor is %s", 
                self.name, self.descriptor)

        # Check the descriptor axes
        
        if len(self.descriptor.axes) > 0 and len(self.descriptor.axes) < 3:
            self.plot_dims = len(self.descriptor.axes)
        if len(self.descriptor.axes) == 0:
            raise ValueError("Plotter got descriptor with zero dimensions. Thanks!")
        else:
            raise ValueError("Cannot plot in greater than 2D, give me a break!")

        # Get the plotting extents


        # Establish the proper plot variety
        if self.plot_dims == 1:
            self.figure = Figure(plot_width=400, plot_height=400, title=self.title,
                                 x_axis_label=self.xlabel, y_axis_label=self.ylabel, **self.fig_args, webgl=True)
            self.plot = self.figure.line([],[], name=title, **plot_args)
            renderers = self.plot.select(dict(name=title))
            self.renderer = [r for r in renderers if isinstance(r, GlyphRenderer)][0]
            self.data_source = self.renderer.data_source
            
        else: # 2D
            self.figure = Figure(x_range=[xmin, xmax], y_range=[ymin, ymax], plot_width=400, plot_height=400,
                                 x_axis_label=self.xlabel, y_axis_label=self.ylabel, title=self.title, webgl=True)
            self.plot = self.figure.image(image=[self.z_data], x=[xmin], y=[ymin],
                                          dw=[xmax-xmin], dh=[ymax-ymin], name=self.title, **plot_args)
            renderers = self.plot.select(dict(name=title))
            self.renderer = [r for r in renderers if isinstance(r, GlyphRenderer)][0]
            self.data_source = self.renderer.data_source

        self.x_data = stream.descriptor.axes[0].points
        self.y_data = np.full(stream.num_points(), np.nan)
        self.figure = figure(plot_width=400, plot_height=400, x_range=(self.x_data[0], self.x_data[-1]))
        

    async def run(self):
        idx = 0

        while True:
            if all([stream.done() for stream in self.input_streams]):
                print("No more data for plotter")
                break
            new_data = await self.input_streams[0].queue.get()
            print("Plotter got {} points".format(len(new_data)))
            self.y_data[idx:idx+len(new_data)] = new_data
            idx += len(new_data)
            #have to copy data to get new pointer to trigger update
            #TODO: investigate streaming
            self.plot.data_source.data["y"] = np.copy(self.y_data)

            self.x_data.append(self.x.value)
            self.y_data.append(self.y.value)
            if (time.time() - self.last_update >= self.update_interval) or force:
                self.data_source.data["x"] = np.copy(self.x_data)
                self.data_source.data["y"] = np.copy(self.y_data)
                self.last_update = time.time()


