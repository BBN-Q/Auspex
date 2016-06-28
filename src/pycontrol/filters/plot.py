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
        self.x_axis = None # Can be string on numerical index
        self.y_axis = None
        self.update_interval = 0.5
        self.last_update = time.time()

    @property
    def x_axis(self):
        return self._x_axis
    @x_axis.setter
    def x_axis(self, value):
        self._x_axis = value
        if self.data.descriptor is not None:
            self.update_descriptors() 

    @property
    def y_axis(self):
        return self._y_axis
    @y_axis.setter
    def y_axis(self, value):
        self._y_axis = value
        if self.data.descriptor is not None:
            self.update_descriptors() 

    def update_descriptors(self):
        self.stream = list(self.input_connectors.values())[0]
        self.descriptor = self.stream.descriptor
        names = [a.name for a in self.descriptor.axes]
        logger.debug("Starting descriptor update in filter %s, where the descriptor is %s", 
                self.name, self.descriptor)

        # Check the descriptor axes
        num_axes = len(self.descriptor.axes)
        remaining_axes = num_axes
        if self.x_axis is None and self.y_axis is None:
            if num_axes == 1:
                self.x_axis = 0
            elif num_axes == 2:
                self.x_axis = 0
                self.y_axis = 1
            else:
                raise Exception("Cannot infer plot axes. Too many choices.")
        elif self.x_axis is None:
            self.x_axis = 0

        # Convert named axes to an index
        if isinstance(self.x_axis, str):
            if self.x_axis not in names:
                raise ValueError("Could not find x_axis {} within the DataStreamDescriptor {}".format(self.x_axis, self.descriptor))
            self.x_axis = names.index(self.x_axis)

        remaining_axes -= 1
        self.x_values = self.descriptor.axes[self.x_axis].points
        self.xmax = max(self.x_values)
        self.xmin = min(self.x_values)

        if self.y_axis is not None:
            # Convert named axes to an index
            if isinstance(self.y_axis, str):
                if self.y_axis not in names:
                    raise ValueError("Could not find y_axis {} within the DataStreamDescriptor {}".format(self.y_axis, self.descriptor))
                self.y_axis = names.index(self.y_axis)

            remaining_axes -= 1
            self.y_values = self.descriptor.axes[self.y_axis].points
            self.ymax = max(self.y_values)
            self.ymin = min(self.y_values)

            self.x_mesh, self.y_mesh = np.meshgrid(self.x_values, self.y_values)
            self.z_data = np.zeros_like(self.x_mesh)

        if remaining_axes < 0:
            raise Exception("Trying to plot more axes than we have.")
        
        self.plot_dims = num_axes - remaining_axes

        # Establish the proper plot variety
        if self.plot_dims == 1:
            self.figure = Figure(plot_width=400, plot_height=400, **self.fig_args, webgl=True)
            self.plot = self.figure.line([],[], name=title, **plot_args)
            self.points_before_clear = self.descriptor.num_points_through_axis(self.x_axis+1)
        else: # 2D
            self.figure = Figure(x_range=[xmin, xmax], y_range=[ymin, ymax], plot_width=400, plot_height=400, webgl=True)
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
        temp = np.empty(self.stream.num_points())

        while True:

            new_data = np.array(await self.input_streams[0].queue.get()).flatten()
            temp[idx:idx+new_data.size()] = new_data
            idx += new_data.size()

            print("Plotter got {} points".format(len(new_data)))
            
            # Grab all of the partial plots
            num_partials = int(idx/self.points_before_partial_average)
            if completed_averages + num_partials > self.num_averages:
                num_partials = self.num_averages - completed_averages

            if self.plots_dims == 1:
                # self.x_data.append(self.x.value)
                # self.y_data.append(self.y.value)

                if (time.time() - self.last_update >= self.update_interval) or self.stream.done():
                    self.plot.data_source.data["x"] = np.copy(self.x_values[0:idx])
                    self.plot.data_source.data["y"] = np.copy(self.y_data)
                    
            else:


            self.last_update = time.time()
            # 
            #have to copy data to get new pointer to trigger update
            #TODO: investigate streaming
            # self.plot.data_source.data["y"] = np.copy(self.y_data)



            if self.stream.done():
                print("No more data for plotter")
                break

