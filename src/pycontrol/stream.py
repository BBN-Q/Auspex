import asyncio
from functools import reduce

class DataAxis(object):
    """An axes in a data stream"""
    def __init__(self, name, points, unit=None):
        super(DataAxis, self).__init__()
        self.name  = name
        self.points = points
        self.unit   = unit
    def num_points(self):
        return len(self.points)
    def __repr__(self):
        return "<DataAxis(name={}, points={}, unit={})>".format(
            self.name, self.points, self.unit)

class DataStreamDescriptor(object):
    """Axis information"""
    def __init__(self):
        super(DataStreamDescriptor, self).__init__()
        self.axes = []

    def add_axis(self, axis):
        self.axes.append(axis)

    def num_dims(self):
        return len(self.axes)

    def data_dims(self, fortran=True):
        if fortran:
            return [len(a.points) for a in self.axes]
        else:
            return [len(a.points) for a in reversed(self.axes)]

    def num_points(self):
        return reduce(lambda x,y: x*y, [len(a.points) for a in self.axes])

    def num_points_through_axis(self, axis):
        return reduce(lambda x,y: x*y, [len(a.points) for a in self.axes[:axis+1]])

    def __repr__(self):
        return "<DataStreamDescriptor(num_dims={}, num_points={})>".format(
            self.num_dims(), self.num_points())

class DataStream(object):
    """A stream of data"""
    def __init__(self, name=None, unit=None):
        super(DataStream, self).__init__()
        self.queue = asyncio.Queue()
        self.points_taken = 0
        self.descriptor = None
        self.name = name
        self.unit = unit

    def set_descriptor(self, descriptor):
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

    def __repr__(self):
        return "<DataStream(name={}, completion={}%, descriptor={})>".format(
            self.name, self.percent_complete(), self.descriptor)

    async def push(self, data):
        if hasattr(data, 'size'):
            self.points_taken += data.size
        else:
            self.points_taken += len(data)
        await self.queue.put(data)

# These connectors are where we attached the DataStreams

class InputConnector(object):
    def __init__(self, name="", datatype=None, max_input_streams=1):
        self.name = name
        self.stream = None
        self.max_input_streams = max_input_streams
        self.num_input_streams = 0
        self.input_streams = []

    def add_input_stream(self, stream):
        logger.debug("Adding input stream '%s' to input connector %s.", stream, self)
        if self.num_input_streams < self.max_input_streams:
            self.input_streams.append(stream)
            self.num_input_streams += 1
        else:
            raise ValueError("Could not add another input stream to the connector.")

    def connect_to(self, output_connector):
        stream = DataStream()
        stream.name = output_connector.name
        self.add_input_stream(stream)
        output_connector.add_output_stream(stream)
        return stream

    def __repr__(self):
        return "<InputConnector(name={})>".format(self.name)

class OutputConnector(object):
    def __init__(self, name="", datatype=None):
        self.name = name
        self.stream = None
        self.output_streams = []

    def add_output_stream(self, stream):
        logger.debug("Adding output stream '%s' to output connector %s.", stream, self)
        self.output_streams.append(stream)

    def connect_to(self, input_connector):
        stream = DataStream()
        stream.name = self.name
        self.add_output_stream(stream)
        input_connector.add_input_stream(stream)
        return stream

    def __repr__(self):
        return "<OutputConnector(name={})>".format(self.name)
