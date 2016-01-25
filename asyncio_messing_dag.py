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
        print("Added output stream {:s} to {:s}".format(str(stream), str(self)))
        self.output_streams.append(stream)

    async def run(self):
        print("Running {:s}!".format(str(self)))
        while True:
            #Produce fake data every 0.02 seconds until we have 1000 points
            if False not in [os.done() for os in self.output_streams]:
                print("Data taker finished.")
                break
            await asyncio.sleep(0.02)
            new_data = np.random.rand(50)
            print("Generated fake new data {:s} at {:s}!".format(str(new_data[:5]),str(self)))
            for os in self.output_streams:
                await os.push(new_data)
                os.points_taken += 50

class DataCruncher(object):
    """docstring for DataCruncher"""
    def __init__(self):
        super(DataCruncher, self).__init__()
        self.input_stream   = None
        self.output_streams = []
        self.descriptor     = None

    def add_input_stream(self, stream):
        if self.input_stream is None:
            self.input_stream = stream
            self.data = np.empty(self.input_stream.num_points())
            self.descriptor = self.input_stream.descriptor
        else:
            raise Exception("DataCruncher takes only one input")

    def add_output_stream(self, stream):
        print("Added output stream to {:s}".format(str(self)))
        self.output_streams.append(stream)
        print("Streams are now {:s}!".format(str(self.output_streams)))

    async def run(self):
        print("Streams are now {:s} at beginning of lop!".format(str(self.output_streams)))
        idx = 0
        while True:
            if self.input_stream.done():
                # We'ce stopped receiving new input, make sure we've flushed the output streams
                if len(self.output_streams) > 0:
                    if False not in [os.done() for os in self.output_streams]:
                        print("Cruncher finished crunching (clearing outputs).")
                        break
                else:
                    print("Cruncher finished crunching.")
                    break

            new_data = await self.input_stream.queue.get()
            print("Crunched fake new data {:s} at {:s}!".format(str(new_data[:5]),str(self)))

            self.data[idx:idx+len(new_data)] = new_data
            for output_stream in self.output_streams:
                print("Pushing data {:s} to output stream".format(str(new_data[:10])))
                await output_stream.push(new_data)
                output_stream.points_taken += len(new_data)
            idx += len(new_data)



# class Processor(object):
#     """docstring for Processor"""
#     def __init__(self):
#         super(Processor, self).__init__()
        
#         self.input_stream = None
#         self.output_stream = None

#     def set_input_stream(self, stream):
#         self.input_stream = stream
#         self.q = stream.subscribe()
#         self.data = np.empty(stream.num_points())

#         # By default, copy the descriptor of the input stream
#         in_desc = stream.descriptor
            
#         # Establish an output stream
#         self.output_stream = DataStream(in_desc)

#     async def run(self):
#         idx = 0
#         while True:
#             if self.input_stream.done():
#                 break
#             new_data = await self.q.get()
#             print("Processed fake new data at {:s}!".format(str(self)))
#             self.data[idx:idx+len(new_data)] = new_data
#             await self.output_stream.push(new_data)
#             self.output_stream.points_taken += len(new_data)
#             idx += len(new_data)


# class FakeProcessor(Processor):
#     def __init__(self, a, b):
#         super(FakeProcessor, self).__init__()
#         self.a = a
#         self.b = b
    
#     async def run(self):
#         idx = 0
#         while True:
#             if self.input_stream.done():
#                 break
#             new_data = await self.q.get()
#             print("Processed fake new data at {:s}: {:s}!".format(str(self),str(new_data)))
#             self.data[idx:idx+len(new_data)] = new_data
#             await self.output_stream.push(self.a*new_data + self.b)
#             self.output_stream.points_taken += len(new_data)
#             idx += len(new_data)

# class Combiner(object):
#     """docstring for Combiner"""
#     def __init__(self):
#         super(Combiner, self).__init__()
        
#         self.input_streams = []
#         self.qs = []
#         self.data_containers = []
#         self.output_stream = None

#     def set_input_stream(self, stream):
#         self.input_streams.append(stream)
#         self.qs.append(stream.subscribe())
#         print("Created new q {:s} for {:s}!".format(str(self.qs[-1]), str(self)))
#         self.data_containers.append(np.empty(stream.num_points()))

#         if self.output_stream is None:
#             # By default, copy the descriptor of the most recently added input stream
#             in_desc = stream.descriptor
#             # Establish an output stream
#             self.output_stream = DataStream(in_desc)

#     async def run(self):
#         # We can only push when queues are of commensurate length
#         # So keep track of what's been pushed.
#         last_push_idx = 0
#         idxs = [0]*len(self.qs)

#         while True:
#             if False not in [ins.done() for ins in self.input_streams]:
#                 break
            
#             # Accumulate new data from all the queues
#             for i, (q, d) in enumerate(zip(self.qs, self.data_containers)):
#                 new_data = await q.get()
#                 d[idxs[i]:idxs[i]+len(new_data)] = new_data
#                 idxs[i] += len(new_data)
            
#             # Once all of the queues have surpassed the last push point, write to the stream
#             print("Last push: {:d}, Idxs: {:s}".format(last_push_idx, str(idxs)))

#             if np.min(idxs) > last_push_idx:
#                 new_data_length = np.min(idxs) - last_push_idx
#                 print("Adding {:d} points".format(new_data_length))
#                 #Let's just add everything by default:
#                 new_data = np.sum(self.data_containers[:,last_push_idx:new_data_length], axis=0)   

#                 await self.output_stream.push(new_data)
#                 self.output_stream.points_taken += len(new_data)
#                 last_push_idx += new_data_length

#                 print("Combined fake new data from q {:s} at {:s}!".format(str(q), str(self)))


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
                stream.set_descriptor(src.descriptor)
                print(stream)
                dest.add_input_stream(stream) 
                src.add_output_stream(stream)
                processed_edges.append(edge)

    tasks = [n.run() for n in dag.nodes()]
    return dag, tasks


if __name__ == '__main__':
    descrip = DataStreamDescriptor()
    descrip.add_axis(DataAxis("time", 1e-9*np.arange(1000)))

    ADC        = DataTaker(descrip)
    # fake_processor1 = FakeProcessor(0.2, 1)
    # fake_processor2 = FakeProcessor(0.1, 2)
    # fake_processor3 = FakeProcessor(0.4, 2)
    cruncher1  = DataCruncher()
    cruncher2  = DataCruncher()
    cruncher3  = DataCruncher()
    # combiner   = Combiner()

    edges = [
        (ADC, cruncher1),
        (ADC, cruncher2),
        (cruncher2, cruncher3)
    ]

    dag, tasks = create_graph(edges)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))

    nx.draw(dag, with_labels=True)
    plt.show()





