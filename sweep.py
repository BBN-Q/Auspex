from __future__ import print_function, division
import logging
import datetime

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
        self.indices = range(self.length)

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
            logging.info("Largest index is {:d}".format(largest_index))
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
