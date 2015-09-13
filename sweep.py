from __future__ import print_function, division
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd

import time
import h5py

from procedure import Procedure, Parameter, Quantity

def cartesian(arrays, out=None):
    """
    From http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

class Writer(object):
    """Basically a data structure"""
    def __init__(self, dataset, params_and_quants):
        super(Writer, self).__init__()
        self.dataset = dataset
        self.params_and_quants = params_and_quants

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
        self._parameters =  []
        self._values = []

        # Would be better to have this all in a dictionary
        # but just store the lengths there for now.
        self._sweep_lengths = {}

        self._current_index = -1
        self._quantities = []

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
            self._parameters.append(param)
            self._values.append( np.linspace(start_value, stop_value, steps).tolist() )
            param.swept = True
        elif interval is not None:
            self._parameters.append(param)
            self._values.append( np.arange(start_value, stop_value + 0.5*interval, interval).tolist() )
            param.swept = True
        else:
            raise ValueError("Invalid specification of Parameter Sweep")

        # Keep track of the record lengths
        self._sweep_lengths[param] = len(self._values[-1]) 

        # Generate the full set of permutations
        self.generate_sweep()

    def add_writer(self, filename, dataset_name, *params_and_quants, **kwargs):
        """Add a dataset that updates based on the supplies quantities"""
        
        params = []
        quants = []

        for pq in params_and_quants:
            if not (isinstance(pq, Parameter) or isinstance(pq, Quantity)):
                raise TypeError("Expecting Parameter or Quantity, not %s" % str(type(pq)) )
            elif isinstance(pq, Parameter):
                params.append(pq)
            elif isinstance(pq, Quantity):
                quants.append(pq)

        # Look before we leap
        if filename not in self._filenames:
            self._filenames.append(filename)
            self._files[filename] = h5py.File(filename, 'w')
        if dataset_name not in self._files[filename]:
            # Construct the dimensions (sweep1_length, sweep2_length, num_params + num_quants)
            p_dimensions = [self._sweep_lengths[p] for p in params]
            q_dimensions = [len(quants)+len(params)]
            dataset_dimensions = tuple(p_dimensions + q_dimensions)

            # Get the datatype
            dtype = kwargs['dtype'] if 'dtype' in kwargs else 'f'

            # Create the data set
            dset = self._files[filename].create_dataset(dataset_name, dataset_dimensions, dtype=dtype)

            # Create a new instances of the data structure and store it
            self._writers.append( Writer(dset, params_and_quants) )
        else:
            raise Exception("Cannot have the same dataset name twice in the same file.")

    def write(self):
        for w in self._writers:
            for i, pq in enumerate(w.params_and_quants):
                w.dataset[self._current_index, i] = pq.value

    def generate_sweep(self):
        self._sweep_values = cartesian(self._values)

    def next(self):
        logging.info("Sweep moving to step %d" % self._current_index)
        if self._current_index >= len(self._sweep_values)-1:
            for p in self._parameters:
                p.swept = False
            raise StopIteration()
        else:
            self._current_index += 1
            for i in range(len(self._parameters)):
                self._parameters[i].value = self._sweep_values[self._current_index][i]

            self._procedure.run()
            self.write()
