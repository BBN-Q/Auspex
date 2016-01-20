import asyncio

class DataStream(object):
    """A stream of data"""
    def __init__(self, descriptor):
        super(DataStream, self).__init__()
        self.descriptor = descriptor
        self.queues = []
        self.points_taken = 0

    def num_points(self):
        return self.descriptor.num_points()

    def percent_complete(self):
        return self.points_taken/self.num_points()

    def done(self):
        return self.points_taken >= self.num_points()

    def subscribe(self):
        """Create a new data queue and return a reference"""
        self.queues.append(asyncio.Queue())
        return self.queues[-1]

    async def push_queues(self, data):
        for q in self.queues:
            await q.put(data)

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
        return sum([len(a.points) for a in self.axes])

class DataAxis(object):
    """An axes in a data stream"""
    def __init__(self, label, points, ticks=None):
        super(DataAxis, self).__init__()
        self.label = label
        self.points = points

import numpy as np

class FakeDataTaker(object):
    """docstring for FakeDataTaker"""
    def __init__(self, stream):
        super(FakeDataTaker, self).__init__()
        self.stream = stream

    async def run(self):
        while True:
            #Produce fake data every 0.1 seconds until we have 1000 points
            if self.stream.done():
                break
            await asyncio.sleep(0.1)
            print("Data recevied!")
            await self.stream.push_queues(np.random.rand(50))
            self.stream.points_taken += 50

class FakeDataCruncher(object):
    """docstring for FakeDataCruncher"""
    def __init__(self, stream):
        super(FakeDataCruncher, self).__init__()
        self.input_stream = stream
        self.q = stream.subscribe()
        self.data = np.empty(stream.num_points())

    async def process_data(self):
        idx = 0
        while True:
            if self.input_stream.done():
                break
            new_data = await self.q.get()
            self.data[idx:idx+len(new_data)] = new_data
            idx += len(new_data)

if __name__ == '__main__':
    descrip = DataStreamDescriptor()
    descrip.add_axis(DataAxis("time", 1e-9*np.arange(1000)))
    fake_stream = DataStream(descrip)
    fake_ADC = FakeDataTaker(fake_stream)
    fake_cruncher = FakeDataCruncher(fake_stream)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([fake_ADC.run(), fake_cruncher.process_data()]))
