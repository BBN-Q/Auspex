from __future__ import print_function, division
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd
import itertools
import time
import h5py

from procedure import Procedure, Parameter, Quantity

class Writer(object):
    """Data structure for the written quantities"""
    def __init__(self, dataset, quantities):
        super(Writer, self).__init__()
        self.dataset = dataset
        self.quantities = quantities

class SweptParameter(object):
    """Data structure for a swept Parameters, contains the Parameter
    object rather than subclassing it since we just need to keep track
    of some values"""
    def __init__(self, parameter, values):
        self.parameter = parameter
        self.values = values
        self.length = len(values)

    def index_of(self, value):
        return self.values.index(value)

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
        self._current_index = -1

        # Container for written Quantities
        self._quantities = []

        # Iterable the yields sweep values
        self._sweep_generator = None

        # Contains a list of tuples (dataset_object, [list of parameters and quantities] )
        self._filenames = []
        self._files = {}
        self._writers = []

    def __iter__(self):
        return self

    def add_parameter(self, param, start_value, stop_value, steps=None, interval=None):
        if not isinstance(param, Parameter):
            raise TypeError("A parameter not deriving from the base class Parameter was provided to the add_parameter method.")

        if steps is None and interval is None:
            raise ValueError("Must specify either number of steps or step interval")
        elif steps is not None:
            values = np.linspace(start_value, stop_value, steps).tolist()
            self._parameters.append(SweptParameter(param, values))
        elif interval is not None:
            values = np.arange(start_value, stop_value + 0.5*interval, interval).tolist()
            self._parameters.append(SweptParameter(param, values))
        else:
            raise ValueError("Invalid specification of Parameter Sweep")

        # Generate the full set of permutations
        self.generate_sweep()

    def add_writer(self, filename, dataset_name, *quants, **kwargs):
        """Add a dataset that updates based on the supplied quantities"""

        # Loop through and check the supplied quantities
        for q in quants:
            if not isinstance(q, Quantity):
                raise TypeError("Expecting Quantity, not %s" % str(type(q)) )

        # Look before we leap
        if filename not in self._filenames:
            self._filenames.append(filename)
            self._files[filename] = h5py.File(filename, 'w')

        if dataset_name not in self._files[filename]:
            # Determine the dataset dimensions
            sweep_dims = [ p.length for p in self._parameters ]
            logging.debug("Sweep dims are %s for the list of swept parameters in the writer %s, %s." % (str(sweep_dims), filename, dataset_name) )

            data_dims = [len(quants)+len(self._parameters)]
            dataset_dimensions = tuple(sweep_dims + data_dims)

            # Get the datatype, defaulting to float
            dtype = kwargs['dtype'] if 'dtype' in kwargs else 'f'

            # Create the data set
            dset = self._files[filename].create_dataset(dataset_name, dataset_dimensions, dtype=dtype)

            # Create a new instances of the data structure and store it
            self._writers.append( Writer(dset, quants) )
        else:
            raise Exception("Cannot have the same dataset name twice in the same file.")

    def write(self):
        for w in self._writers:
            current_p_values = [p.parameter.value for p in self._parameters]
            current_p_indices = [p.index_of(pv) for p, pv in zip(self._parameters, current_p_values)]
            logging.debug("Current indicies are: %s" % str(current_p_indices) )

            for i, p in enumerate(self._parameters):
                coords = tuple(current_p_indices + [i])
                logging.debug("Coords: %s" % str(coords) )
                w.dataset[coords] = p.parameter.value
            for i, q in enumerate(w.quantities):
                coords = tuple( current_p_indices + [len(self._parameters) + i] )
                w.dataset[coords] = q.value

    def generate_sweep(self):
        self._sweep_generator = itertools.product(*[sp.values for sp in self._parameters])

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
