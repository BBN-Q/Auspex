from __future__ import print_function, division
import logging
import datetime
import string
import signal
import sys

# logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

# For plotting
import threading
from collections import deque
import json
from flask import Flask, jsonify, request
from bokeh.server.crossdomain import crossdomain
import urllib2

import numpy as np
import scipy as sp
import pandas as pd
import itertools
import time
import h5py

from bokeh.plotting import figure, show, output_file, hplot, vplot
from bokeh.models.sources import AjaxDataSource

from procedure import Procedure, Parameter, Quantity

class Writer(object):
    """Data structure for the written quantities"""
    def __init__(self, dataset, quantities):
        super(Writer, self).__init__()
        self.dataset = dataset
        self.quantities = quantities

class Plotter(object):
    """Attach a plotter to the sweep."""
    def __init__(self, title, x, ys, **figure_args):
        super(Plotter, self).__init__()
        self.title = title
        self.filename = string.replace(title, ' ', '_')
        output_file(self.filename, title=self.title)
        self.figure_args = figure_args

        # These are parameters and quantities
        self.x = x
        if isinstance(ys, list):
            self.ys = ys
        else:
            self.ys = [ys]
        self.num_ys = len(self.ys)

        # FIFO data container
        self.data = deque()

    def update(self):
        data = [self.x.value]
        data.extend([y.value for y in self.ys])
        self.data.append( tuple(data) )

class SweptParameter(object):
    """Data structure for a swept Parameters, contains the Parameter
    object rather than subclassing it since we just need to keep track
    of some values"""
    def __init__(self, parameter, values):
        self.parameter = parameter
        self.values = values
        self.length = len(values)
        self.indices = range(self.length)

class FlaskThread(threading.Thread):
    def __init__(self, plotters):
        self.data_lookup = {p.filename: p.data for p in plotters}
        self.filenames = [p.filename for p in plotters]
        self.plotter_lookup = {p.filename: p for p in plotters}

        self.app = Flask(__name__)
        @self.app.route('/<filename>', methods=['GET', 'OPTIONS'])
        @crossdomain(origin="*", methods=['GET', 'POST'], headers=None)
        def fetch_func(filename):
            if filename == "shutdown":
                func = request.environ.get('werkzeug.server.shutdown')
                if func is None:
                    raise RuntimeError('Not running with the Werkzeug Server')
                func()
                return 'Server shutting down...'
            else:
                p = self.plotter_lookup[filename]

                xs = []
                ys = [[] for i in range(p.num_ys)]
                while True:
                    try:
                        data = self.data_lookup[filename].popleft()
                        xs.append(data[0])
                        for i in range(p.num_ys):
                            ys[i].append(data[i+1])
                    except:
                        break
                kwargs = { 'y{:d}'.format(i+1): ys[i] for i in range(p.num_ys) }
                kwargs['x'] = xs
                return jsonify(**kwargs)

        super(FlaskThread, self).__init__()

    def run(self):
        output_file("main.html", title="Plotting Output")
        plots = []
        sources = []

        for f in self.filenames:
            p = self.plotter_lookup[f]
            source = AjaxDataSource(data_url='http://localhost:5050/'+f,
                                    polling_interval=750, mode="append")
            
            xlabel = p.x.name + (" ("+p.x.unit+")" if p.x.unit is not None else '')
            ylabel = p.ys[0].name + (" ("+p.ys[0].unit+")" if p.ys[0].unit is not None else '')
            plot = figure(webgl=True, title=p.title,
                          x_axis_label=xlabel, y_axis_label=ylabel, 
                          tools="save,crosshair")
            plots.append(plot)
            sources.append(source)

            # plots[-1].line('x', 'y', source=sources[-1], color="firebrick", line_width=2)
            xargs = ['x' for i in range(p.num_ys)]
            yargs = ['y{:d}'.format(i+1) for i in range(p.num_ys)]
            
            if p.num_ys > 1:
                plots[-1].multi_line(xargs, yargs, source=sources[-1], **p.figure_args)
            else:
                plots[-1].line('x', 'y1', source=sources[-1], **p.figure_args)

        q = hplot(*plots)
        show(q)
        self.app.run(port=5050)

class Sweep(object):
    """For controlling sweeps over arbitrary number of arbitrary parameters. The order of sweeps\
    is defined by the order of the parameters passed to the add_parameter method. The first\
    quantity varies the slowest, the final quantity the quickest.

    """
    def __init__(self, procedure):
        super(Sweep, self).__init__()

        if isinstance(procedure, Procedure):
            self._procedure = procedure
        else:
            raise TypeError("Must pass a Procedure subclass.")

        # Container for SweptParmeters
        self._parameters =  []

        # Container for written Quantities
        self._quantities = []

        # Iterable the yields sweep values
        self._sweep_generator = None

        # Contains a list of tuples (dataset_object, [list of parameters and quantities] )
        self._filenames = []
        self._files = {}
        self._writers = []
        self._plotters = []

    def __iter__(self):
        return self

    def add_parameter(self, param, sweep_list):
        self._parameters.append(SweptParameter(param, sweep_list))
        self.generate_sweep()

    def add_writer(self, filename, sample_name, dataset_name, *quants, **kwargs):
        """Add a dataset that updates based on the supplied quantities"""

        # Loop through and check the supplied quantities
        for q in quants:
            if not isinstance(q, Quantity):
                raise TypeError("Expecting Quantity, not %s" % str(type(q)) )

        # See if we've already made the file
        if filename not in self._filenames:
            self._filenames.append(filename)
            self._files[filename] = h5py.File(filename, 'a')
        f = self._files[filename]

        # See if there is already a group matching this sample
        if sample_name not in f.keys():
            f.create_group(sample_name)
        s = f[sample_name]

        # See if there is already a group matching today's date
        date_str = datetime.date.today().strftime('%Y-%m-%d')
        if date_str not in s.keys():
            s.create_group(date_str)
        g = s[date_str]

        # See if there is already a dataset with the same name
        # increment the actual dataset name by 1 and store this
        # as dataset_name-0001 dataset_name-0002, etc. First we
        # parse any filenames already in the group, then we make
        # sure we store a new file with the name incremented by
        # 1.

        files_with_same_prefix = ["-".join(k.split("-")[:-1]) for k in g.keys() if dataset_name == "-".join(k.split("-")[:-1])]
        if dataset_name not in files_with_same_prefix:
            dataset_name = "{:s}-{:04d}".format(dataset_name, 1)
        else:
            largest_index = max([int(k.split("-")[-1]) for k in g.keys() if dataset_name in k])
            dataset_name = "{:s}-{:04d}".format(dataset_name, largest_index + 1)

        # Determine the dataset dimensions
        sweep_dims = [ p.length for p in self._parameters ]
        logging.debug("Sweep dims are %s for the list of swept parameters in the writer %s, %s." % (str(sweep_dims), filename, dataset_name) )

        data_dims = [len(quants)+len(self._parameters)]
        dataset_dimensions = tuple(sweep_dims + data_dims)

        # Get the datatype, defaulting to float
        dtype = kwargs['dtype'] if 'dtype' in kwargs else 'f'

        # Create the data set
        dset = g.create_dataset(dataset_name, dataset_dimensions, dtype=dtype)

        # Create a new instances of the data structure and store it
        self._writers.append( Writer(dset, quants) )

    def write(self):
        indices = list(next(self._index_generator))

        for w in self._writers:
            current_p_values = [p.parameter.value for p in self._parameters]

            logging.debug("Current indicies are: %s" % str(indices) )

            for i, p in enumerate(self._parameters):
                coords = tuple( indices + [i] )
                logging.debug("Coords: %s" % str(coords) )
                w.dataset[coords] = p.parameter.value
            for i, q in enumerate(w.quantities):
                coords = tuple( indices + [len(self._parameters) + i] )
                w.dataset[coords] = q.value

    def add_plotter(self, title, x, y, *args, **kwargs):
        self._plotters.append(Plotter(title, x, y, *args, **kwargs))

    def plot(self):
        for p in self._plotters:
            p.update()

    def generate_sweep(self):
        self._sweep_generator = itertools.product(*[sp.values for sp in self._parameters])
        self._index_generator = itertools.product(*[sp.indices for sp in self._parameters])

    #Python 3 compatible iterator
    #TODO if we go all in on Python 3, remove this and replace next with __next__ below
    def __next__(self):
        return self.next()

    def next(self):
        ps = next(self._sweep_generator)

        for i, p in enumerate(self._parameters):
            p.parameter.value = ps[i]

        self._procedure.run()
        self.write()
        self.plot()

    def run(self):
        """Run everything all at once..."""

        if len(self._plotters) > 0:
            t = FlaskThread(self._plotters)
            t.start()

        def shutdown():
            if len(self._plotters) > 0:
                time.sleep(0.5)
                response = urllib2.urlopen('http://localhost:5050/shutdown').read()
                t.join()

        def catch_ctrl_c(signum, frame):
            logging.info("Caught SIGINT.  Shutting down.")
            shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, catch_ctrl_c)

        for param_values in self._sweep_generator:

            # Update the paramater values
            for i, p in enumerate(self._parameters):
                p.parameter.value = param_values[i]

            # Run the procedure
            self._procedure.run()

            # Push values to file and update plots
            self.write()
            self.plot()

        shutdown()
