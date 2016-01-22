import asyncio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

class FakeDataTaker(object):
    """docstring for FakeDataTaker"""
    def __init__(self, stream):
        super(FakeDataTaker, self).__init__()
        self.output_stream = stream

    async def run(self):
        while True:
            #Produce fake data every 0.1 seconds until we have 1000 points
            if self.output_stream.done():
                break
            await asyncio.sleep(0.02)
            print("Generated fake new data at {:s}!".format(str(self)))
            await self.output_stream.push_queues(np.random.rand(50))
            self.output_stream.points_taken += 50

class FakeDataCruncher(object):
    """docstring for FakeDataCruncher"""
    def __init__(self):
        super(FakeDataCruncher, self).__init__()
        self.output_stream = None

    def set_input_stream(self, stream):
        self.input_stream = stream
        self.q = stream.subscribe()
        self.data = np.empty(stream.num_points())

    async def run(self):
        idx = 0
        while True:
            if self.input_stream.done():
                break
            new_data = await self.q.get()
            print("Crunched fake new data at {:s}!".format(str(self)))
            self.data[idx:idx+len(new_data)] = new_data
            idx += len(new_data)

class Processor(object):
    """docstring for Processor"""
    def __init__(self):
        super(Processor, self).__init__()
        
        self.input_stream = None
        self.output_stream = None

    def set_input_stream(self, stream):
        self.input_stream = stream
        self.q = stream.subscribe()
        self.data = np.empty(stream.num_points())

        # By default, copy the descriptor of the input stream
        in_desc = stream.descriptor
            
        # Establish an output stream
        self.output_stream = DataStream(in_desc)

    async def run(self):
        idx = 0
        while True:
            if self.input_stream.done():
                break
            new_data = await self.q.get()
            print("Processed fake new data at {:s}!".format(str(self)))
            self.data[idx:idx+len(new_data)] = new_data
            await self.output_stream.push_queues(new_data)
            self.output_stream.points_taken += len(new_data)
            idx += len(new_data)


class FakeProcessor(Processor):
    def __init__(self, a, b):
        super(FakeProcessor, self).__init__()
        self.a = a
        self.b = b
    
    async def run(self):
        idx = 0
        while True:
            if self.input_stream.done():
                break
            new_data = await self.q.get()
            print("Processed fake new data at {:s}: {:s}!".format(str(self),str(new_data)))
            self.data[idx:idx+len(new_data)] = new_data
            await self.output_stream.push_queues(self.a*new_data + self.b)
            self.output_stream.points_taken += len(new_data)
            idx += len(new_data)

class Combiner(object):
    """docstring for Combiner"""
    def __init__(self):
        super(Combiner, self).__init__()
        
        self.input_streams = []
        self.qs = []
        self.data_containers = []
        self.output_stream = None

    def set_input_stream(self, stream):
        self.input_streams.append(stream)
        self.qs.append(stream.subscribe())
        print("Created new q {:s} for {:s}!".format(str(self.qs[-1]), str(self)))
        self.data_containers.append(np.empty(stream.num_points()))

        if self.output_stream is None:
            # By default, copy the descriptor of the most recently added input stream
            in_desc = stream.descriptor
            # Establish an output stream
            self.output_stream = DataStream(in_desc)

    async def run(self):
        # We can only push when queues are of commensurate length
        # So keep track of what's been pushed.
        last_push_idx = 0
        idxs = [0]*len(self.qs)

        while True:
            if False not in [ins.done() for ins in self.input_streams]:
                break
            
            # Accumulate new data from all the queues
            for i, (q, d) in enumerate(zip(self.qs, self.data_containers)):
                new_data = await q.get()
                d[idxs[i]:idxs[i]+len(new_data)] = new_data
                idxs[i] += len(new_data)
            
            # Once all of the queues have surpassed the last push point, write to the stream
            print("Last push: {:d}, Idxs: {:s}".format(last_push_idx, str(idxs)))

            if np.min(idxs) > last_push_idx:
                new_data_length = np.min(idxs) - last_push_idx
                print("Adding {:d} points".format(new_data_length))
                #Let's just add everything by default:
                new_data = np.sum(self.data_containers[:,last_push_idx:new_data_length], axis=0)   

                await self.output_stream.push_queues(new_data)
                self.output_stream.points_taken += len(new_data)
                last_push_idx += new_data_length

                print("Combined fake new data from q {:s} at {:s}!".format(str(q), str(self)))


def create_graph(edges):
    dag = nx.DiGraph()
    dag.add_edges_from(edges)

    # Find the input nodes
    input_nodes = [n for n in dag.nodes() if dag.in_degree(n) == 0]

    # Edge depth-first traversal of the graph starting from these input nodes
    bfs_edge_iters  = [nx.edge_dfs(dag, input_node) for input_node in input_nodes]
    processed_edges = [] # Keep track of what we've initialized

    for ei in bfs_edge_iters:
        for edge in ei:
            if edge not in processed_edges:
                src  = edge[0]
                dest = edge[1]
                dest.set_input_stream(src.output_stream) 
                processed_edges.append(edge)

    tasks = [n.run() for n in dag.nodes()]

    return dag, tasks


if __name__ == '__main__':
    descrip = DataStreamDescriptor()
    descrip.add_axis(DataAxis("time", 1e-9*np.arange(1000)))
    fake_stream = DataStream(descrip)

    fake_ADC        = FakeDataTaker(fake_stream)
    fake_processor1 = FakeProcessor(0.2, 1)
    fake_processor2 = FakeProcessor(0.1, 2)
    fake_processor3 = FakeProcessor(0.4, 2)
    fake_cruncher1  = FakeDataCruncher()
    fake_cruncher2  = FakeDataCruncher()
    fake_cruncher3  = FakeDataCruncher()
    fake_combiner   = Combiner()

    edges = [
        (fake_ADC, fake_processor1),
        (fake_ADC, fake_processor2),
        (fake_processor2, fake_processor3),
        (fake_processor1, fake_cruncher1),
        (fake_processor3, fake_cruncher2),
        (fake_processor1, fake_combiner),
        (fake_processor2, fake_combiner),
        (fake_processor3, fake_combiner),
        (fake_combiner, fake_cruncher3)
    ]

    dag, tasks = create_graph(edges)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))

    nx.draw(dag, with_labels=True)
    plt.show()





