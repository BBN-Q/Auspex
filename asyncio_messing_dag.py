import asyncio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)
class DataStream(object):
    """A stream of data"""
    def __init__(self):
        super(DataStream, self).__init__()
        self.queue = asyncio.Queue()
        self.points_taken = 0

    def set_descriptor(self, descriptor):
        self.descriptor = descriptor

    def num_points(self):
        return self.descriptor.num_points()

    def percent_complete(self):
        return self.points_taken/self.num_points()

    def done(self):
        return self.points_taken >= self.num_points()

    async def push(self, data):
        self.points_taken += len(data)
        await self.queue.put(data)

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

class DataTaker(object):
    """docstring for DataTaker"""
    def __init__(self, descriptor):
        super(DataTaker, self).__init__()
        self.input_streams  = None
        self.output_streams = []
        self.descriptor = descriptor

    def add_output_stream(self, stream):
        self.output_streams.append(stream)

    def update_descriptors(self):
        for os in self.output_streams:
            os.descriptor = self.descriptor

    async def run(self):
        print("Data taker running")
        while True:
            #Produce fake data every 0.02 seconds until we have 1000 points
            if False not in [os.done() for os in self.output_streams]:
                print("Data taker finished.")
                break
            await asyncio.sleep(0.02)
            new_data = np.random.rand(50)
            print("Data taker pushing data")
            for os in self.output_streams:
                await os.push(new_data)

class ProcessingNode(object):
    """Any node on the graph that takes input streams with optional output streams"""
    def __init__(self):
        super(ProcessingNode, self).__init__()
        self.input_streams  = []
        self.output_streams = []

    def add_input_stream(self, stream):
        self.input_streams.append(stream)

    def add_output_stream(self, stream):
        self.output_streams.append(stream)

    def update_descriptors(self):
        for os in self.output_streams:
            os.descriptor = self.input_streams[0].descriptor

class DataCruncher(ProcessingNode):
    """docstring for DataCruncher"""
    def __init__(self):
        super(DataCruncher, self).__init__()

    async def run(self):
        idx = 0
        self.data = np.empty(self.input_streams[0].num_points())
        while True:
            if self.input_streams[0].done():
                # We've stopped receiving new input, make sure we've flushed the output streams
                if len(self.output_streams) > 0:
                    if False not in [os.done() for os in self.output_streams]:
                        print("Cruncher finished crunching (clearing outputs).")
                        break
                else:
                    print("Cruncher finished crunching.")
                    break

            new_data = await self.input_streams[0].queue.get()

            self.data[idx:idx+len(new_data)] = new_data
            for output_stream in self.output_streams:
                await output_stream.push(new_data)
            idx += len(new_data)

class Combiner(ProcessingNode):
    """docstring for Combiner"""
    def __init__(self):
        super(Combiner, self).__init__()
        self.data_containers = []

    async def run(self):
        print("Combiner running")
        # We can only push when queues are of commensurate length
        # So keep track of what's been pushed.
        last_push_idx = 0
        idxs = [0]*len(self.input_streams)
        for ins in self.input_streams:
            self.data_containers.append(np.empty(ins.num_points()))

        while True:
            if all([ins.done() for ins in self.input_streams]):
                if len(self.output_streams) > 0:
                    if all([os.done() for os in self.output_streams]):
                        print("Combiner finished combining (clearing outputs).")
                        break
                else:
                    print("Combiner finished combining.")
                    break

            # Accumulate new data from all the queues
            # print(str(self.input_streams))
            for i, (container, input_stream) in enumerate(zip(self.data_containers, self.input_streams)):
                print("Combiner waiting on stream")
                new_data = await input_stream.queue.get()
                print("Combiner: new data {:s} on stream {:s}".format(str(new_data), str(input_stream)))
                # print("Last push: {:d}, Idxs: {:s}".format(last_push_idx, str(idxs)))
                container[idxs[i]:idxs[i]+len(new_data)] = new_data
                idxs[i] += len(new_data)

            # Once all of the queues have surpassed the last push point, write to the stream
            # print("Last push: {:d}, Idxs: {:s}".format(last_push_idx, str(idxs)))

            if np.min(idxs) > last_push_idx:
                new_data_length = np.min(idxs) - last_push_idx
                print("Adding {:d} points".format(new_data_length))

                #Let's just sum everything by default:
                new_data = np.zeros(new_data_length)
                for dc in self.data_containers:
                    new_data = new_data + dc[last_push_idx:last_push_idx+new_data_length]
                print("Combined fake new data into {:s}!".format(str(new_data)))

                for output_stream in self.output_streams:
                    await output_stream.push(new_data)
                last_push_idx += new_data_length

def create_graph(edges):
    dag = nx.DiGraph()
    for edge in edges:
        dag.add_edge(edge[0], edge[1], object=DataStream())

    # Find the input nodes
    input_nodes = [n for n in dag.nodes() if dag.in_degree(n) == 0]

    # # Edge depth-first traversal of the graph starting from these input nodes
    bfs_edge_iters  = [nx.edge_dfs(dag, input_node) for input_node in input_nodes]
    processed_edges = [] # Keep track of what we've initialized

    for ei in bfs_edge_iters:
        for edge in ei:
            if edge not in processed_edges:
                src    = edge[0]
                dest   = edge[1]
                stream = dag[src][dest]['object']
                print(stream)
                src.add_output_stream(stream)
                dest.add_input_stream(stream)
                processed_edges.append(edge)
                if "update_descriptors" in dir(src):
                    src.update_descriptors()

    tasks = [n.run() for n in dag.nodes()]
    return dag, tasks


if __name__ == '__main__':
    descrip = DataStreamDescriptor()
    descrip.add_axis(DataAxis("time", 1e-9*np.arange(1000)))

    ADC        = DataTaker(descrip)
    cruncher1  = DataCruncher()
    cruncher2  = DataCruncher()
    cruncher3  = DataCruncher()
    combiner   = Combiner()

    edges = [
        (ADC, cruncher1),
        (ADC, cruncher2),
        (cruncher2, cruncher3),
        (cruncher2, combiner),
        (cruncher1, combiner)
    ]

    dag, tasks = create_graph(edges)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))

    nx.draw(dag, with_labels=True)
    plt.show()
