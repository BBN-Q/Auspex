import asyncio
import logging
from functools import reduce
from pycontrol.logging import logger

class DataAxis(object):
    """An axes in a data stream"""
    def __init__(self, name, points, unstructured=False, coord_names=[], unit=None):
        super(DataAxis, self).__init__()
        self.name         = name
        self.points       = points
        self.unit         = unit
        self.unstructured = unstructured
        self.coord_names  = coord_names
        if unstructured and len(coord_names) != len(points[0]):
            raise ValueError("Coordinate names list must be as numerous as the coordinates themselves.")

    def num_points(self):
        return len(self.points)
    def __repr__(self):
        return "<DataAxis(name={}, points={}, unit={}, unstructured={})>".format(
            self.name, self.points, self.unit, self.unstructured)

class DataStreamDescriptor(object):
    """Axis information"""
    def __init__(self):
        super(DataStreamDescriptor, self).__init__()
        self.axes = []
        self.parent = None

    def add_axis(self, axis):
        self.axes.insert(0, axis)

    def num_dims(self):
        return len(self.axes)

    def data_dims(self):
        return [len(a.points) for a in self.axes]

    def num_points(self):
        return reduce(lambda x,y: x*y, [len(a.points) for a in self.axes])

    def num_points_through_axis(self, axis):
        if len(self.axes) == 1:
            return self.axes[0].num_points()
        else:
            return reduce(lambda x,y: x*y, [len(a.points) for a in self.axes[axis:]])

    def __repr__(self):
        return "<DataStreamDescriptor(num_dims={}, num_points={})>".format(
            self.num_dims(), self.num_points())

class DataStream(object):
    """A stream of data"""
    def __init__(self, name=None, unit=None, loop=None):
        super(DataStream, self).__init__()
        self.queue = asyncio.Queue(loop=loop)
        self.points_taken = 0
        self.descriptor = None
        self.name = name
        self.unit = unit
        self.start_connector = None
        self.end_connector = None
        self.loop = loop

    def set_descriptor(self, descriptor):
        logger.debug("Setting descriptor on stream '%s' to '%s'", self.name, descriptor)
        self.descriptor = descriptor

    def num_points(self):
        if self.descriptor is not None:
            return self.descriptor.num_points()
        else:
            return 0

    def percent_complete(self):
        if self.descriptor is not None:
            return 100.0*self.points_taken/self.num_points()
        else:
            return 0.0

    def done(self):
        return (self.points_taken >= self.num_points() - 1) and (self.num_points() > 0)

    def reset(self):
        self.points_taken = 0 
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
        await self.queue.put(data)

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
            raise ValueError("Could not add another input stream to the connector.")

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
    def __init__(self, name="", parent=None, datatype=None):
        self.name = name
        self.stream = None
        self.output_streams = []
        self.descriptor = None
        self.points_taken = 0
        self.parent = parent

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
        return (self.points_taken >= self.descriptor.num_points() - 1) and (self.descriptor.num_points() > 0)

    async def push(self, data):
        if hasattr(data, 'size'):
            self.points_taken += data.size
        else:
            self.points_taken += len(data)
        for stream in self.output_streams:
            await stream.push(data)

    def __repr__(self):
        return "<OutputConnector(name={})>".format(self.name)
