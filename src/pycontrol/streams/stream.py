import asyncio
from functools import reduce

class DataAxis(object):
    """An axes in a data stream"""
    def __init__(self, label, points, unit=None):
        super(DataAxis, self).__init__()
        self.label  = label
        self.points = points
        self.unit   = unit
    def __repr__(self):
        return "<DataAxis(label={}, points={}, unit={})>".format(
            self.label, self.points, self.unit)

class DataStreamDescriptor(object):
    """Axis information"""
    def __init__(self):
        super(DataStreamDescriptor, self).__init__()
        self.axes = []

    def add_axis(self, axis):
        self.axes.append(axis)

    def num_dims(self):
        return len(self.axes)

    def num_points(self):
        return reduce(lambda x,y: x*y, [len(a.points) for a in self.axes])

    def __repr__(self):
        return "<DataStreamDescriptor(num_dims={}, num_points={})>".format(
            self.num_dims(), self.num_points())

class DataStream(object):
    """A stream of data"""
    def __init__(self, name=None):
        super(DataStream, self).__init__()
        self.queue = asyncio.Queue()
        self.points_taken = 0
        self.descriptor = None
        self.name = name

    def set_descriptor(self, descriptor):
        self.descriptor = descriptor

    def num_points(self):
        return self.descriptor.num_points()

    def percent_complete(self):
        return 100.0*self.points_taken/self.num_points()

    def done(self):
        return self.points_taken >= self.num_points() - 1

    def __repr__(self):
        return "<DataStream(completion={}%, descriptor={})>".format(
            self.percent_complete(), self.descriptor)

    async def push(self, data):
        if hasattr(data, 'size'):
            self.points_taken += data.size
        else:
            self.points_taken += len(data)
        await self.queue.put(data)

class ProcessingNode(object):
    """Any node on the graph that takes input streams with optional output streams"""
    def __init__(self, label=None):
        super(ProcessingNode, self).__init__()
        self.label = label
        self.input_streams     = []
        self.output_streams    = []
        self.max_input_streams = 1
        self.num_input_streams = 0

    def __str__(self):
        return str(self.label)

    def add_input_stream(self, stream):
        if self.num_input_streams < self.max_input_streams:
            self.input_streams.append(stream)
            self.num_input_streams += 1
        else:
            raise ValueError("Could not add another input stream to the node {}.".format(self.label))

    def add_output_stream(self, stream):
        self.output_streams.append(stream)

    def update_descriptors(self):
        for os in self.output_streams:
            os.descriptor = self.input_streams[0].descriptor
