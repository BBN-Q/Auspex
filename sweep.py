from __future__ import print_function, division
import logging
import datetime
import string
import signal
import sys

# For plotting
from collections import deque
from FlaskPlotter import FlaskPlotter

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

    @property
    def value(self):
        return self.parameter.value

    @value.setter
    def value(self, value):
        self.parameter.value = value

    def push(self):
        for pph in self.parameter.pre_push_hooks:
            pph()
        self.parameter.push()
        for pph in self.parameter.post_push_hooks:
            pph()

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
        self._swept_parameters =  []

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
        self._swept_parameters.append(SweptParameter(param, sweep_list))
        self.generate_sweep()

        # Set the value of the parameter to the initial value of the sweep
        param.value = sweep_list[0]

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
        sweep_dims = [ p.length for p in self._swept_parameters ]
        logging.debug("Sweep dims are %s for the list of swept parameters in the writer %s, %s." % (str(sweep_dims), filename, dataset_name) )

        data_dims = [len(quants)+len(self._swept_parameters)]
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
            current_p_values = [p.parameter.value for p in self._swept_parameters]

            logging.debug("Current indicies are: %s" % str(indices) )

            for i, p in enumerate(self._swept_parameters):
                coords = tuple( indices + [i] )
                logging.debug("Coords: %s" % str(coords) )
                w.dataset[coords] = p.parameter.value
            for i, q in enumerate(w.quantities):
                coords = tuple( indices + [len(self._swept_parameters) + i] )
                w.dataset[coords] = q.value

    def add_plotter(self, title, x, y, *args, **kwargs):
        self._plotters.append(Plotter(title, x, y, *args, **kwargs))

    def plot(self):
        for p in self._plotters:
            p.update()

    def generate_sweep(self):
        self._sweep_generator = itertools.product(*[sp.values for sp in self._swept_parameters])
        self._index_generator = itertools.product(*[sp.indices for sp in self._swept_parameters])

    def run(self):
        self._procedure.init_instruments()

        if len(self._plotters) > 0:
            fp = FlaskPlotter(self._plotters)

        def shutdown():
            if len(self._plotters) > 0:
                fp.shutdown()
            self._procedure.shutdown_instruments()

        def catch_ctrl_c(signum, frame):
            logging.info("Caught SIGINT.  Shutting down.")
            shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, catch_ctrl_c)

        # Keep track of the previous values
        last_param_values = None

        for param_values in self._sweep_generator:

            # Update the parameter values. Unles set and push if there has been a change
            # in the value from the previous iteration.
            for i, sp in enumerate(self._swept_parameters):
                if last_param_values is None or param_values[i] != last_param_values[i]:
                    sp.value = param_values[i]
                    sp.push()
                    logging.info("Updated {:s} to {:g} since the value changed.".format(sp.parameter.name, sp.value))
                else:
                    logging.info("Didn't update {:s} since the value didn't change.".format(sp.parameter.name))

            # update previous values
            last_param_values = param_values

            # Run the procedure
            self._procedure.run()

            # Push values to file and update plots
            self.write()
            self.plot()

        shutdown()
