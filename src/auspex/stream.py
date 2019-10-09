# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import os
import sys

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    import threading as mp
    from queue import Queue
else:
    import multiprocessing as mp
    from multiprocessing import Queue
from multiprocessing import Value, RawValue, RawArray
import ctypes
import logging
import numbers
import itertools
import time
import datetime

import numpy as np
from functools import reduce

from auspex.log import logger

def cartesian(arrays, out=None, dtype='f'):
    """http://stackoverflow.com/questions/28684492/numpy-equivalent-of-itertools-product"""

    arrays = [np.asarray(x) for x in arrays]
    # dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

class DataAxis(object):
    """An axis in a data stream"""
    def __init__(self, name, points=[], unit=None, metadata=None, dtype=np.float32):
        super(DataAxis, self).__init__()
        if isinstance(name, list):
            self.unstructured = True
            self.name = name
        else:
            self.unstructured = False
            self.name         = str(name)

        # self.points holds the CURRENT set of points. During adaptive sweeps
        # this will hold the most recently added points of the axis.
        self.points        = np.array(points)
        self.unit          = unit
        self.refine_func   = None
        self.metadata      = metadata

        # By definition data axes will be done after every experiment.run() call
        self.done         = True

        # For adaptive sweeps, etc., keep a record of the original points that we had around
        self.original_points = self.points
        self.has_been_extended = False
        self.num_new_points = 0
        self.dtype         = dtype

        if self.unstructured:
            if unit is not None and len(name) != len(unit):
                raise ValueError("DataAxis unit length {} and tuples length {} must match.".format(len(unit),len(name)))
        if self.unstructured and len(name) != len(points[0]):
            raise ValueError("DataAxis points length {} and names length {} must match.".format(len(points[0]), len(name)))

    def data_type(self, with_metadata=False):
        dtype = []
        if self.unstructured:
            name = "+".join(self.name)
            dtype.extend([(p.name, 'f') for p in self.parameter])
        else:
            name = self.name
            dtype.append((name, 'f'))

        if with_metadata and self.metadata is not None:
            dtype.append((name + "_metadata", str))
        return dtype

    def points_with_metadata(self):
        if self.metadata is not None:
            if self.unstructured:
                return [list(self.original_points[i]) + [self.metadata[i]] for i in range(len(self.original_points))]
            return [(self.original_points[i], self.metadata[i], ) for i in range(len(self.original_points))]
        if self.unstructured:
            return [tuple(self.original_points[i]) for i in range(len(self.original_points))]
        return [(self.original_points[i],) for i in range(len(self.original_points))]

    def tuple_width(self):
        if self.unstructured:
            width = len(name)
        else:
            width = 1
        if self.metadata:
            width += 1
        return width

    def num_points(self):
        if self.has_been_extended:
            return len(self.points)
        else:
            return len(self.original_points)

    def add_points(self, points):
        if self.unstructured and len(self.parameter) != len(points[0]):
            raise ValueError("Parameter value tuples must be the same length as the number of parameters.")

        if type(points) in [list, np.ndarray]:
            points = np.array(points)
        else:
            # Somebody gave one point to the "add_points" method...
            points = np.array([points])

        self.num_new_points = len(points)
        self.points = np.append(self.points, points, axis=0)
        self.has_been_extended = True

    def reset(self):
        self.points = self.original_points
        self.has_been_extended = False
        self.num_new_points = 0

    def __repr__(self):
        return "<DataAxis(name={}, start={}, stop={}, num={}, unit={})>".format(
            self.name, self.points[0], self.points[-1], len(self.points), self.unit)

    def __str__(self):
        return "<DataAxis(name={}, start={}, stop={}, num={}, unit={})>".format(
            self.name, self.points[0], self.points[-1], len(self.points), self.unit)

class SweepAxis(DataAxis):
    """ Structure for sweep axis, separate from DataAxis.
    Can be an unstructured axis, in which case 'parameter' is actually a list of parameters. """
    def __init__(self, parameter, points = [], metadata=None, refine_func=None, callback_func=None):

        self.unstructured = hasattr(parameter, '__iter__')
        self.parameter    = parameter
        if self.unstructured:
            unit = [p.unit for p in parameter]
            super(SweepAxis, self).__init__([p.name for p in parameter], points=points, unit=unit, metadata=metadata)
            self.value = points[0]
        else:
            super(SweepAxis, self).__init__(parameter.name, points, unit=parameter.unit, metadata=metadata)
            self.value     = points[0]

        # Current value of the metadata
        if self.metadata is not None:
            self.metadata_value = self.metadata[0]

        # This is run at the end of this sweep axis
        # Refine_func receives the sweep axis and the experiment as arguments
        self.refine_func = refine_func

        # This is run before each point in the sweep axis is executed
        # Callback_func receives the sweep axis and the experiment as arguments
        self.callback_func = callback_func

        self.step        = 0
        self.done        = False
        self.experiment  = None # Should be explicitly set by the experiment

        if self.unstructured and len(parameter) != len(points[0]):
            raise ValueError("Parameter value tuples must be the same length as the number of parameters.")

        logger.debug("Created {}".format(self.__repr__()))

    def update(self):
        """ Update value after each run.
        """
        if self.step < self.num_points():
            if self.callback_func:
                self.callback_func(self, self.experiment)
            self.value = self.points[self.step]
            if self.metadata is not None:
                self.metadata_value = self.metadata[self.step]
            logger.debug("Sweep Axis '{}' at step {} takes value: {}.".format(self.name,
                                                                               self.step,self.value))
            self.push()
            self.step += 1
            self.done = False

    def check_for_refinement(self, output_connectors_dict):
        """Check to see if we need to perform any refinements. If there is a refine_func
        and it returns a list of points, then we need to extend the axes. Otherwise, if the
        refine_func returns None or false, then we reset the axis to its original set of points. If
        there is no refine_func then we don't do anything at all."""

        if not self.done and self.step==self.num_points():
            logger.debug("Refining on axis {}".format(self.name))
            if self.refine_func:
                points = self.refine_func(self, self.experiment)
                if points is None or points is False:
                    # Returns false if no refinements needed, otherwise adds points to list
                    self.step = 0
                    self.done = True
                    self.reset()
                    logger.debug("Sweep Axis '{}' complete.".format(self.name))
                    # Push to ocs, which should push to processes
                    for oc in output_connectors_dict.values():
                        oc.push_event("refined", (self.name, True, self.original_points)) # axis name, reset, points
                    return False
                self.add_points(points)
                self.done = False
                for oc in output_connectors_dict.values():
                    oc.push_event("refined", (self.name, False, points)) # axis name, reset, points
                return True
            else:
                self.step = 0
                self.done = True
                logger.debug("Sweep Axis '{}' complete.".format(self.name))
                return False

    def push(self):
        """ Push parameter value(s) """
        if self.unstructured:
            for p, v in zip(self.parameter, self.value):
                p.value = v
                p.push()
        else:
            self.parameter.value = self.value
            self.parameter.push()

    def __repr__(self):
        return "<SweepAxis(name={},length={},unit={},value={},unstructured={}>".format(self.name,
                self.num_points(),self.unit,self.value,self.unstructured)

class DataStreamDescriptor(object):
    """Axes information"""
    def __init__(self, dtype=np.float32):
        super(DataStreamDescriptor, self).__init__()
        self.data_name = "Data"
        self.data_unit = "Unit"
        self.axes = []
        self.unit = None
        self.params = {} # Parameters associated with each dataset
        self.parent = None
        self._exp_src = None # Actual source code from the underlying experiment
        self.dtype = dtype
        self.metadata = {}

        # Buffer size multiplier: use this to inflate the size of the
        # shared memory buffer. This is needed for partial averages, which
        # may require more space than their descriptors would indicate
        # since they are emitted as often as possible.
        self.buffer_mult_factor = 1

        # Keep track of the parameter permutations we have actually used...
        self.visited_tuples = []

    def is_adaptive(self):
        return True in [a.refine_func is not None for a in self.axes]

    def add_axis(self, axis, position=0):
        # Check if axis is DataAxis or SweepAxis (which inherits from DataAxis)
        if isinstance(axis, DataAxis):
            logger.debug("Adding DataAxis into DataStreamDescriptor: {}".format(axis))
            self.axes.insert(position, axis)
        else:
            raise TypeError("Failed adding axis. Object is not DataAxis: {}".format(axis))

    def add_param(self, key, value):
        self.params[key] = value

    def num_dims(self):
        # Number of axes
        return len(self.axes)

    def extent(self, flip=False):
        """Convenience function for matplotlib.imshow, which expects extent=(left, right, bottom, top)."""
        if self.num_dims() == 2:
            return (self.axes[1].points[0], self.axes[1].points[-1], self.axes[0].points[0], self.axes[0].points[-1])
        else:
            raise Exception("Can't get extent for any number of axes other than two.")

    def data_dims(self):
        # Return dimension (length) of the data axes, exclude sweep axes (return 1 for each)
        dims = []

        for a in self.axes:
            if isinstance(a, SweepAxis):
                dims.append(1)
            else:
                dims.append(len(a.points))
        return dims

    def tuple_width(self):
        return sum([a.tuple_width() for a in self.axes])

    def dims(self):
        dims = []

        for a in self.axes:
                dims.append(len(a.points))
        return [a.num_points() for a in self.axes]

    def axes_done(self):
        # The axis is considered done when all of the sub-axes are done
        # This can happen mulitple times for a single axis
        doneness = [a.done for a in self.axes]
        return [np.all(doneness[i:]) for i in range(len(doneness))]

    def done(self):
        return np.all([a.done for a in self.axes])

    def num_points(self):
        if len(self.axes)>0:
            return reduce(lambda x,y: x*y, [a.num_points() for a in self.axes])
        else:
            return 0

    def expected_num_points(self):
        if len(self.axes)>0:
            return reduce(lambda x,y: x*y, [len(a.original_points) for a in self.axes])
        else:
            return 0

    def last_data_axis(self):
        # Return the outer most data axis but not sweep axis
        data_axes_idx = [i for i, a in enumerate(self.axes) if not isinstance(a,SweepAxis)]
        if len(data_axes_idx)>0:
            return data_axes_idx[0]
        else:
            logger.warning("DataStreamDescriptor has no pure DataAxis. Return None.")
            return None

    def axis_data_type(self, with_metadata=False, excluding_axis=None):
        dtype = []
        for a in self.axes:
            if a.name != excluding_axis:
                dtype.extend(a.data_type(with_metadata=with_metadata))
        return dtype

    def tuples(self, as_structured_array=True):
        """Returns a list of all tuples visited by the sweeper. Should only
        be used with adaptive sweeps."""
        if len(self.visited_tuples) == 0:
            self.visited_tuples = self.expected_tuples(with_metadata=True)

        if as_structured_array:
            # If we already have a structured array
            if type(self.visited_tuples) is np.ndarray and type(self.visited_tuples.dtype.names) is tuple:
                return self.visited_tuples
            elif type(self.visited_tuples) is np.ndarray:
                return np.rec.fromarrays(self.visited_tuples.T, dtype=self.axis_data_type(with_metadata=True))
            return np.core.records.fromrecords(self.visited_tuples, dtype=self.axis_data_type(with_metadata=True))
        return self.visited_tuples

    def expected_tuples(self, with_metadata=False, as_structured_array=True):
        """Returns a list of tuples representing the cartesian product of the axis values. Should only
        be used with non-adaptive sweeps."""
        vals           = [a.points_with_metadata() for a in self.axes]

        #
        # TODO: avoid this slow list comprehension
        simple = True
        if True in [a.unstructured for a in self.axes]:
            simple = False
        if True in [a.metadata is not None for a in self.axes]:
            simple = False
        if self.axes == []:
            simple = False

        if simple:
            # flattened_list = [tuple((val for sublist in line for val in sublist)) for line in nested_list]
            flattened_list = cartesian(vals)
        else:
            nested_list    = itertools.product(*vals)
            flattened_list = [tuple((val for sublist in line for val in sublist)) for line in nested_list]
            # flattened_list = np.array(list(nested_list)).reshape(-1, self.tuple_width())

        if as_structured_array:
            if simple:
                return np.rec.fromarrays(flattened_list.T, dtype=self.axis_data_type(with_metadata=True))
            return np.rec.fromrecords(flattened_list, dtype=self.axis_data_type(with_metadata=True))
        return flattened_list

    def axis_names(self, with_metadata=False):
        """Returns all axis names included those from unstructured axes"""
        vals = []
        for a in self.axes:
            if a.unstructured:
                for p in a.parameter:
                    vals.append(p.name)
            else:
                vals.append(a.name)
            if with_metadata and a.metadata is not None:
                if a.unstructured:
                    vals.append("+".join(a.name) + "_metadata")
                else:
                    vals.append(a.name + "_metadata")
        return vals

    def num_data_axis_points(self):
        return self.num_points_through_axis(self.last_data_axis())

    def data_axis_values(self):
        """Returns a list of point lists for each data axis, ignoring sweep axes."""
        return [a.points_with_metadata() for a in self.axes if not isinstance(a,SweepAxis) ]

    def reset(self):
        for a in self.axes:
            a.done = False
            a.reset()

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        newone.axes = self.axes[:]
        return newone

    def copy(self):
        return self.__copy__()

    def axis(self, axis_name):
        return self.axes[self.axis_num(axis_name)]

    def axis_num(self, axis_name):
        names = [a.name for a in self.axes]
        return names.index(axis_name)

    def pop_axis(self, axis_name):
        # Pop the time axis (which should be here)
        names = [a.name for a in self.axes]
        if axis_name not in names:
            raise Exception("Couldn't pop axis {} from descriptor, it probably doesn't exist.".format(axis_name))
        return self.axes.pop(names.index(axis_name))

    def num_points_through_axis(self, axis_name):
        if type(axis_name) is int:
            axis_num = axis_name
        else:
            axis_num = self.axis_num(axis_name)

        # if False in [a.refine_func is None for a in self.axes[axis_num:]]:
        #     raise Exception("Cannot call num_points_through_axis with interior adaptive sweeps.")

        if axis_num >= len(self.axes):
            return 0
        elif len(self.axes) == 1:
            return self.axes[0].num_points()
        else:
            return reduce(lambda x,y: x*y, [a.num_points() for a in self.axes[axis_num:]])

    def num_new_points_through_axis(self, axis_name):
        if type(axis_name) is int:
            axis_num = axis_name
        else:
            axis_num = self.axis_num(axis_name)

        if axis_num >= len(self.axes):
            return 0
        elif len(self.axes) == 1:
            return self.axes[0].num_new_points
        else:
            return self.axes[axis_num].num_new_points * reduce(lambda x,y: x*y, [a.num_points() for a in self.axes[axis_num+1:]])

    def __repr__(self):
        return "<DataStreamDescriptor(num_dims={}, num_points={})>".format(
            self.num_dims(), self.num_points())

    def __getitem__(self, axis_name):
        return self.axis(axis_name).points

    def _ipython_key_completions_(self):
        return [a.name for a in self.axes]

class DataStream(object):
    """A stream of data"""
    def __init__(self, name=None, unit=None):
        super(DataStream, self).__init__()
        self.queue = Queue()
        self.name = name
        self.unit = unit
        self.points_taken_lock = mp.Lock()
        self.points_taken = Value('i', 0) # Using shared memory since these are used in filter processes
        self.descriptor = None
        self.start_connector = None
        self.end_connector = None
        self.closed = False

        # Shared memory interface
        self.buffer_lock    = mp.Lock()
        # self.buffer_size    = 500000
        self.buff_idx       = Value('i', 0)

    def final_init(self):
        self.buffer_size = self.descriptor.num_points()*self.descriptor.buffer_mult_factor
        # logger.info(f"{self.start_connector.parent}:{self.start_connector} to {self.end_connector.parent}:{self.end_connector} buffer of size {self.buffer_size}")
        if self.buffer_size > 50e6:
            logger.debug(f"Limiting buffer size of {self} to 50 Million Points")
            self.buffer_size = 50e6
        self.buff_shared_re = RawArray(ctypes.c_double, int(self.buffer_size))
        self.buff_shared_im = RawArray(ctypes.c_double, int(self.buffer_size))
        self.re_np = np.frombuffer(self.buff_shared_re, dtype=np.float64)
        self.im_np = np.frombuffer(self.buff_shared_im, dtype=np.float64)

    def set_descriptor(self, descriptor):
        if isinstance(descriptor,DataStreamDescriptor):
            logger.debug("Setting descriptor on stream '%s' to '%s'", self.name, descriptor)
            self.descriptor = descriptor
        else:
            raise TypeError("Failed setting descriptor. Object is not DataStreamDescriptor: {}".format(descriptor))

    def num_points(self):
        if self.descriptor is not None:
            return self.descriptor.num_points()
        else:
            logger.warning("Stream '{}' has no descriptor. Function num_points() returns 0.".format(self.name))
            return 0

    def done(self):
        with self.points_taken_lock:
            return self.points_taken.value >= self.num_points()

    def percent_complete(self):
        if (self.descriptor is not None) and self.num_points()>0:
            with self.points_taken_lock:
                return 100.0*self.points_taken.value/self.num_points()
        else:
            return 0.0

    def reset(self):
        self.descriptor.reset()
        with self.points_taken_lock:
            self.points_taken.value = 0
        while not self.queue.empty():
            self.queue.get_nowait()
        if self.start_connector is not None:
            self.start_connector.points_taken.value = 0

    def __repr__(self):
        return "<DataStream(name={}, completion={}%, descriptor={})>".format(
            self.name, self.percent_complete(), self.descriptor)

    def push(self, data):
        if self.closed:
            raise Exception("The queue is closed and should not be receiving any more data")
        with self.points_taken_lock:
            if hasattr(data, 'size'):
                self.points_taken.value += data.size
            else:
                try:
                    self.points_taken.value += len(data)
                except:
                    try:
                        junk = data + 1.0
                        self.points_taken.value += 1
                    except:
                        raise ValueError("Got data {} that is neither an array nor a float".format(data))
        with self.buffer_lock:
            start = self.buff_idx.value
            re = np.real(np.array(data)).flatten()
            if start+re.size > self.re_np.size:
                raise ValueError(f"Stream {self} received more data than fits in the shared memory buffer. \
                    This is probably due to digitizer raw streams producing data too quickly for the pipeline.")
            self.re_np[start:start+re.size] = re
            if np.issubdtype(self.descriptor.dtype, np.complexfloating):
                im = np.imag(data).flatten()
                self.im_np[start:start+im.size] = im
            message = {"type": "data", "data": None}
            self.buff_idx.value = start + np.array(data).size
        self.queue.put(message)

    def pop(self):
        result = None
        with self.buffer_lock:
            idx = self.buff_idx.value
            if idx != 0:
                result = self.re_np[:idx]

                if np.issubdtype(self.descriptor.dtype, np.complexfloating):
                    result = result.astype(np.complex128) + 1.0j*self.im_np[:idx]
                self.buff_idx.value = 0
                result = result.copy()
        return result

    def push_event(self, event_type, data=None):
        if self.closed:
            raise Exception("The queue is closed and should not be receiving any more data")
        message = {"type": "event", "event_type": event_type, "data": data}
        self.queue.put(message)
        if event_type == "done":
            logger.debug(f"Closing out queue {self}")
            self.queue.close()
            self.closed = True

# These connectors are where we attached the DataStreams
class InputConnector(object):
    def __init__(self, name="", parent=None, datatype=None, max_input_streams=1):
        self.name = name
        self.stream = None
        self.max_input_streams = max_input_streams
        self.num_input_streams = 0
        self.input_streams = []
        self.descriptor = None
        self.parent = parent

    def add_input_stream(self, stream):
        logger.debug("Adding input stream '%s' to input connector %s.", stream, self)
        if self.num_input_streams < self.max_input_streams:
            self.input_streams.append(stream)
            self.num_input_streams += 1
            stream.end_connector = self
        else:
            raise ValueError("Reached maximum number of input connectors. Could not add another input stream to the connector.")

    def done(self):
        return all([stream.done() for stream in self.input_streams])

    def num_points(self):
        if len(self.input_streams) > 0:
            return self.input_streams[0].num_points()
        else:
            raise ValueError("Cannot get num_points since no input streams are present on this connector.")

    def update_descriptors(self):
        logger.debug("Starting descriptor update in input connector %s.", self.name)
        self.descriptor = self.input_streams[0].descriptor
        self.parent.update_descriptors()

    def __repr__(self):
        return "<InputConnector(name={})>".format(self.name)

class OutputConnector(object):
    def __init__(self, name="", data_name=None, unit=None, parent=None, dtype=np.float32):
        self.name = name
        self.output_streams = []
        self.parent = parent
        self.unit = unit
        self.points_taken_lock = mp.Lock()
        self.points_taken = Value('i', 0) # Using shared memory since these are used in filter processes

        # if data_name is not none, then it is the origin of the whole chain
        self.data_name = data_name
        self.data_unit = unit

        # Set up a default descriptor, and add access
        # to its methods for convenience.
        self.descriptor = DataStreamDescriptor(dtype=dtype)
        if self.data_name:
            self.descriptor.data_name = self.data_name
            self.descriptor.unit = self.unit
        self.add_axis   = self.descriptor.add_axis

        # Determine whether we need to deal with adaptive sweeps
        self.has_adaptive_sweeps = False

    def __len__(self):
        with self.points_taken_lock:
            return self.points_taken.value

    # We allow the connectors itself to posess
    # a descriptor, that it may pass
    def set_descriptor(self, descriptor):
        self.descriptor = descriptor

    def add_output_stream(self, stream):
        logger.debug("Adding output stream '%s' to output connector %s.", stream, self)
        self.output_streams.append(stream)
        stream.start_connector = self

    def update_descriptors(self):
        logger.debug("Starting descriptor update in output connector %s, where the descriptor is %s",
                        self.name, self.descriptor)
        for stream in self.output_streams:
            logger.debug("\tnow setting stream %s to %s", stream, self.descriptor)
            stream.set_descriptor(self.descriptor)
            logger.debug("\tnow setting stream end connector %s to %s", stream.end_connector, self.descriptor)
            stream.end_connector.update_descriptors()

    def num_points(self):
        return self.descriptor.num_points()

    def done(self):
        return all([stream.done() for stream in self.output_streams])

    def push(self, data):
        with self.points_taken_lock:
            if hasattr(data, 'size'):
                self.points_taken.value += data.size
            elif isinstance(data, numbers.Number):
                self.points_taken.value += 1
            else:
                self.points_taken.value += len(data)
        for stream in self.output_streams:
            stream.push(data)

    def push_event(self, event_type, data=None):
        for stream in self.output_streams:
            stream.push_event(event_type, data)

    def __repr__(self):
        return "<OutputConnector(name={})>".format(self.name)
