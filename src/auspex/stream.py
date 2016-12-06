# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import logging
import numbers
import itertools
import zlib
import pickle
import time
import datetime

import numpy as np
from functools import reduce

from auspex.log import logger

class DataAxis(object):
    """An axis in a data stream"""
    def __init__(self, name, points=[], unit=None, metadata=None):
        super(DataAxis, self).__init__()
        if isinstance(name, list):
            self.unstructured = True
            self.name = name
        else:
            self.unstructured = False
            self.name         = str(name)
        self.points       = np.array(points)
        self.unit         = unit
        self.metadata     = metadata

        # By definition data axes will be done after every experiment.run() call
        self.done         = True

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
        
        if with_metadata and self.metadata:
            dtype.append((name + "_metadata", 'S128'))
        return dtype

    def points_with_metadata(self):
        if self.metadata:
            if self.unstructured:
                return [list(self.points[i]) + [self.metadata[i]] for i in range(len(self.points))]
            return [(self.points[i], self.metadata[i], ) for i in range(len(self.points))]
        if self.unstructured:
            return [tuple(self.points[i]) for i in range(len(self.points))]
        return [(self.points[i],) for i in range(len(self.points))]

    def num_points(self):
        return len(self.points)

    def add_points(self, points):
        self.points = np.append(self.points, points)

    def __repr__(self):
        return "<DataAxis(name={}, points={}, unit={})>".format(
            self.name, self.points, self.unit)

class SweepAxis(DataAxis):
    """ Structure for sweep axis, separate from DataAxis.
    Can be an unstructured axis, in which case 'parameter' is actually a list of parameters. """
    def __init__(self, parameter, points = [], metadata=None, refine_func=None, callback_func=None):

        self.unstructured = hasattr(parameter, '__iter__')
        self.parameter    = parameter
        if self.unstructured:
            unit = [p.unit for p in parameter]
            super(SweepAxis, self).__init__([p.name for p in parameter], points=points, unit=unit)
            self.value = points[0]
        else:
            super(SweepAxis, self).__init__(parameter.name, points, unit=parameter.unit)
            self.value     = points[0]

        # This is run at the end of this sweep axis
        # Refine_func receives the sweep axis and the experiment as arguments
        self.refine_func = refine_func

        # This is run before each point in the sweep axis is executed
        # Callback_func receives the sweep axis and the experiment as arguments
        self.callback_func = callback_func

        self.step        = 0
        self.done        = False
        self.metadata    = metadata
        self.experiment  = None # Should be explicitly set by the experiment

        if self.unstructured and len(parameter) != len(points[0]):
            raise ValueError("Parameter value tuples must be the same length as the number of parameters.")

        logger.debug("Created {}".format(self.__repr__()))

    def add_points(self, points):
        if self.unstructured:
            self.points = np.append(self.points, points, axis=0)
        else:
            self.points = np.append(self.points, points)

    async def update(self):
        """ Update value after each run.
        If refine_func is None, loop through the list of points.
        """
        if self.step < self.num_points():
            if self.callback_func:
                self.callback_func(self, self.experiment)
            self.value = self.points[self.step]
            logger.debug("Sweep Axis '{}' at step {} takes value: {}.".format(self.name,
                                                                               self.step,self.value))
            self.push()
            self.step += 1
            self.done = False

        if not self.done and self.step==self.num_points():
            # Check to see if we need to perform any refinements
            await asyncio.sleep(0.1)
            logger.debug("Refining on axis {}".format(self.name))
            if self.refine_func is not None:
                if not await self.refine_func(self, self.experiment):
                    # Returns false if no refinements needed, otherwise adds points to list
                    self.step = 0
                    self.done = True
                    logger.debug("Sweep Axis '{}' complete.".format(self.name))
            else:
                self.step = 0
                self.done = True
                logger.debug("Sweep Axis '{}' complete.".format(self.name))

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
        self.axes = []
        self.params = {} # Parameters associated with each dataset
        self.parent = None
        self.exp_src = None # Actual source code from the underlying experiment
        self.dtype = dtype

    def add_axis(self, axis):
        # Check if axis is DataAxis or SweepAxis (which inherits from DataAxis)
        if isinstance(axis, DataAxis):
            logger.debug("Adding DataAxis into DataStreamDescriptor: {}".format(axis))
            self.axes.insert(0, axis)
        else:
            raise TypeError("Failed adding axis. Object is not DataAxis: {}".format(axis))

    def add_param(self, key, value):
        self.params[key] = value

    def num_dims(self):
        # Number of axes
        return len(self.axes)

    def data_dims(self):
        # Return dimension (length) of the data axes, exclude sweep axes (return 1 for each)
        dims = []

        for a in self.axes:
            if isinstance(a, SweepAxis):
                dims.append(1)
            else:
                dims.append(len(a.points))
        return dims

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

    def last_data_axis(self):
        # Return the outer most data axis but not sweep axis
        data_axes_idx = [i for i, a in enumerate(self.axes) if not isinstance(a,SweepAxis)]
        if len(data_axes_idx)>0:
            return data_axes_idx[0]
        else:
            logger.warning("DataStreamDescriptor has no pure DataAxis. Return None.")
            return None

    def axis_data_type(self, with_metadata=False):
        dtype = []
        for a in self.axes:
            dtype.extend(a.data_type(with_metadata=with_metadata))
        # dtype.append((self.data_name, self.dtype))
        return dtype

    def tuples(self, with_metadata=False, as_structured_array=True):
        vals = []
        for a in self.axes:
            if with_metadata:
                vals.append(a.points_with_metadata())
            else:
                vals.append(a.points)
        nested_list = list(itertools.product(*vals))
        flattened_list = [tuple((val for sublist in line for val in sublist)) for line in nested_list]
        if as_structured_array:
            return np.core.records.fromrecords(flattened_list, dtype=self.axis_data_type(with_metadata=True))
        return flattened_list

    def axis_names(self, with_metadata=False):
        # Returns all axis names included those from unstructured axes
        vals = []
        for a in self.axes:
            if a.unstructured:
                for p in a.parameter:
                    vals.append(p.name)
            else:
                vals.append(a.name)
            if with_metadata and a.metadata:
                if a.unstructured:
                    vals.append("+".join(a.name) + "_metadata")
                else:
                    vals.append(a.name + "_metadata")
        return vals

    def data_axis_points(self):
        return self.num_points_through_axis(self.last_data_axis())

    def reset(self):
        for a in self.axes:
            a.done = False

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        newone.axes = self.axes[:]
        return newone

    def copy(self):
        return self.__copy__()

    def axis(self, axis_name):
        names = [a.name for a in self.axes]
        self.axes[names.index(axis_name)]

    def pop_axis(self, axis_name):
        # Pop the time axis (which should be here)
        names = [a.name for a in self.axes]
        if axis_name not in names:
            raise Exception("Couldn't pop axis {} from descriptor, it probably doesn't exist.".format(axis_name))
        return self.axes.pop(names.index(axis_name))

    def num_points_through_axis(self, axis):
        if axis>=len(self.axes):
            return 0
        elif len(self.axes) == 1:
            return self.axes[0].num_points()
        else:
            return reduce(lambda x,y: x*y, [a.num_points() for a in self.axes[axis:]])

    def __repr__(self):
        return "<DataStreamDescriptor(num_dims={}, num_points={})>".format(
            self.num_dims(), self.num_points())

class DataStream(object):
    """A stream of data"""
    def __init__(self, name=None, unit=None, loop=None, compression="none"):
        super(DataStream, self).__init__()
        self.queue = asyncio.Queue(loop=loop)
        self.loop = loop
        self.name = name
        self.unit = unit
        self.points_taken = 0
        self.descriptor = None
        self.start_connector = None
        self.end_connector = None
        self.compression = compression

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

    def percent_complete(self):
        if (self.descriptor is not None) and self.num_points()>0:
            return 100.0*self.points_taken/self.num_points()
        else:
            return 0.0

    def reset(self):
        self.descriptor.reset()
        self.points_taken = 0
        while not self.queue.empty():
            self.queue.get_nowait()
        if self.start_connector is not None:
            self.start_connector.points_taken = 0

    def __repr__(self):
        return "<DataStream(name={}, completion={}%, descriptor={})>".format(
            self.name, self.percent_complete(), self.descriptor)

    async def push(self, data):
        if hasattr(data, 'size'):
            self.points_taken += data.size
        else:
            try:
                self.points_taken += len(data)
            except:
                try:
                    junk = data + 1.0
                    self.points_taken += 1
                except:
                    raise ValueError("Got data {} that is neither an array nor a float".format(data))
        if self.compression == 'zlib':
            message = {"type": "data", "compression": "zlib", "data": zlib.compress(pickle.dumps(data, -1))}
        else:
            message = {"type": "data", "compression": "none", "data": data}

        # This can be replaced with some other serialization method
        # and also should support sending via zmq.
        await self.queue.put(message)

    async def push_event(self, event):
        message = {"type": "event", "compression": "none", "data": event}
        await self.queue.put(message)

    async def push_direct(self, data):
        message = {"type": "data_direct", "compression": "none", "data": data}
        await self.queue.put(message)


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
    def __init__(self, name="", data_name=None, parent=None, datatype=None):
        self.name = name
        self.output_streams = []
        self.points_taken = 0
        self.parent = parent

        # if data_name is not none, then it is the origin of the whole chain
        self.data_name = data_name

        # Set up a default descriptor, and add access
        # to its methods for convenience.
        self.descriptor = DataStreamDescriptor()
        if self.data_name:
            self.descriptor.data_name = self.data_name
        self.add_axis   = self.descriptor.add_axis

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
        return np.all([stream.done for stream in self.output_streams])

    async def push(self, data):
        if hasattr(data, 'size'):
            self.points_taken += data.size
        elif isinstance(data, numbers.Number):
            self.points_taken += 1
        else:
            self.points_taken += len(data)
        for stream in self.output_streams:
            await stream.push(data)

    def __repr__(self):
        return "<OutputConnector(name={})>".format(self.name)
