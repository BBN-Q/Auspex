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
