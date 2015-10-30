from __future__ import print_function, division
import logging
import threading
import string
import time

import numpy as np

import bokeh.server
import bokeh.server.start
from bokeh.plotting import figure, cursession
from bokeh.models.renderers import GlyphRenderer

class BokehServerThread(threading.Thread):
    def __init__(self):
        super(BokehServerThread, self).__init__()
        self.daemon = True
        self.server = None

    def run(self):
        # Need to store some reference to this since bokeh uses global
        # variable "server" for some reason.
        try:
            bokeh.server.server = self.server
            bokeh.server.run()
        except:
            logging.info("Server could not be launched, may already be running.")
            raise Exception("Couldn't start server.")

    def join(self, timeout=None):
        bokeh.server.start.stop()
        super(BokehServerThread, self).join(timeout=timeout)

class Plotter(object):
    """Attach a plotter to the sweep."""
    def __init__(self, title, x, y, **plot_args):
        super(Plotter, self).__init__()
        self.title = title
        self.filename = string.replace(title, ' ', '_')
        self.update_interval = 0.5
        self.last_update = time.time()
        
        # Figure
        self.figure = figure(plot_width=400, plot_height=400)
        self.plot = self.figure.line([],[], name=title, **plot_args)
        renderers = self.plot.select(dict(name=title))
        self.renderer = [r for r in renderers if isinstance(r, GlyphRenderer)][0]
        for r in renderers:
            logging.info("{:s}".format(r))
        self.data_source = self.renderer.data_source

        # Data containers
        self.x_data = []
        self.y_data = []

        # These are parameters and quantities
        self.x = x
        self.y = y
        
    def update(self, force=False):
        self.x_data.append(self.x.value)
        self.y_data.append(self.y.value)
        if (time.time() - self.last_update >= self.update_interval) or force:
            self.data_source.data["x"] = self.x_data
            self.data_source.data["y"] = self.y_data
            cursession().store_objects(self.data_source)
            self.last_update = time.time()

    def clear(self):
        self.x_data = []
        self.y_data = []

class Plotter2D(object):
    """Attach a plotter to the sweep."""
    def __init__(self, title, x, y, z, **plot_args):
        super(Plotter2D, self).__init__()
        self.title = title
        self.filename = string.replace(title, ' ', '_')
        self.update_interval = 0.5
        self.last_update = time.time()

        # Figure
        xmax = max(x.values)
        ymax = max(y.values)
        xmin = min(x.values)
        ymin = min(y.values) 

        # Mesh grid of x and y values from the sweep
        self.x_mesh, self.y_mesh = np.meshgrid(x.values, y.values)
        self.z_data = np.zeros_like(self.x_mesh)

        # These are parameters and quantities
        self.x = x
        self.y = y
        self.z = z

        # Construct the plot
        self.figure = figure(x_range=[xmin, xmax], y_range=[ymin, ymax], plot_width=400, plot_height=400)
        self.plot = self.figure.image(image=[self.z_data], x=[xmin], y=[ymin], 
                                           dw=[xmax-xmin], dh=[ymax-ymin], name=title, **plot_args)
        renderers = self.plot.select(dict(name=title))
        self.renderer = [r for r in renderers if isinstance(r, GlyphRenderer)][0]
        self.data_source = self.renderer.data_source
        
    def update(self, force=False):
        # Find the coordinates and then set the array element
        new_data_loc = np.where(  np.logical_and(self.x_mesh == self.x.value, self.y_mesh == self.y.value)  )
        self.z_data[new_data_loc] = self.z.value
        if (time.time() - self.last_update >= self.update_interval) or force:
            self.data_source.data["image"] = [self.z_data]
            cursession().store_objects(self.data_source)
            self.last_update = time.time()

    def clear(self):
        self.x_data = []
        self.y_data = []
        self.z_data = []