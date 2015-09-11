from __future__ import print_function, division
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)

import numpy as np
import scipy as sp
import pandas as pd

import time

from procedure import Procedure, Parameter

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
        self._current_index = -1
        self._quantities = []
        self._data = []

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

    def generate_sweep(self):
        self._sweep_values = cartesian(self._values)
        # print(len(self._sweep_values))

    def next(self):
        logging.info("Sweep moving to step %d" % self._current_index)
        if self._current_index >= len(self._sweep_values)-1:
            raise StopIteration()
        else:
            self._current_index += 1
            for i in range(len(self._parameters)):
                self._parameters[i].value = self._sweep_values[self._current_index][i]

            # self._data.append( [quant.measure() for quant in self._quantities] )
