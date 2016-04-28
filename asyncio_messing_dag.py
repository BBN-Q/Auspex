import asyncio

import numpy as np
from numpy import pi
import networkx as nx
import matplotlib.pyplot as plt
import logging
import time
from bokeh.plotting import figure
from bokeh.client import push_session
from bokeh.plotting import curdoc
from bokeh.driving import cosine

from pycontrol.plotting import BokehServerThread

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
    def __init__(self, label, descriptor):
        super(DataTaker, self).__init__()
        self.label = label
        self.descriptor = descriptor
        self.input_streams  = None
        self.output_streams = []

    def __str__(self):
        return str(self.label)

    def add_output_stream(self, stream):
        self.output_streams.append(stream)

    def update_descriptors(self):
        for os in self.output_streams:
            os.descriptor = self.descriptor

    async def run(self):
        print("Data taker running")
        start_time = 0
        step = 20e-6
        while True:
            #Produce fake noisy sinusoid data every 0.02 seconds until we have 1000 points
            if False not in [os.done() for os in self.output_streams]:
                print("Data taker finished.")
                break
            await asyncio.sleep(0.5)
            timepts = np.arange(start_time, start_time+49.5*20e-6, 20e-6)
            new_data = np.sin(2*pi*1e3*timepts) + 0.1*np.random.rand(50)
            start_time += 50*step
            print("Data taker pushing data")
            for os in self.output_streams:
                await os.push(new_data)

class ProcessingNode(object):
    """Any node on the graph that takes input streams with optional output streams"""
    def __init__(self, label):
        super(ProcessingNode, self).__init__()
        self.label = label
        self.input_streams  = []
        self.output_streams = []

    def __str__(self):
        return str(self.label)

    def add_input_stream(self, stream):
        self.input_streams.append(stream)

    def add_output_stream(self, stream):
        self.output_streams.append(stream)

    def update_descriptors(self):
        for os in self.output_streams:
            os.descriptor = self.input_streams[0].descriptor

class DataCruncher(ProcessingNode):
    """docstring for DataCruncher"""
    def __init__(self, *args):
        super(DataCruncher, self).__init__(*args)

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
            print("{} got data".format(self.label))

            new_data = 2*new_data

            self.data[idx:idx+len(new_data)] = new_data
            for output_stream in self.output_streams:
                await output_stream.push(new_data)
            idx += len(new_data)

class Combiner(ProcessingNode):
    """docstring for Combiner"""
    def __init__(self, *args):
        super(Combiner, self).__init__(*args)
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
                print("Combiner: new data on stream {:s}".format(str(input_stream)))
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
                # print("Combined fake new data into {:s}!".format(str(new_data)))

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



class Plotter(ProcessingNode):
    """docstring for Plotter"""
    def __init__(self, *args):
        super(Plotter, self).__init__(*args)

    def init(self):
        ins = self.input_streams[0]
        self.x_data = ins.descriptor.axes[0].points
        self.y_data = np.full(ins.num_points(), np.nan)

        #Create the initial plot
        self.figure = figure(plot_width=400, plot_height=400, x_range=(self.x_data[0], self.x_data[-1]))
        self.plot = self.figure.line(self.x_data, self.y_data, color="navy", line_width=2)

    async def run(self):
        idx = 0

        while True:
            if all([ins.done() for ins in self.input_streams]):
                print("No more data for plotter")
                break
            new_data = await self.input_streams[0].queue.get()
            print("Plotter got {} points".format(len(new_data)))
            self.y_data[idx:idx+len(new_data)] = new_data
            idx += len(new_data)
            #have to copy data to get new pointer to trigger update
            #TODO: investigate streaming
            self.plot.data_source.data["y"] = np.copy(self.y_data)


if __name__ == '__main__':
    descrip = DataStreamDescriptor()
    descrip.add_axis(DataAxis("time", 1e-9*np.arange(1000)))

    ADC        = DataTaker("Fake ADC", descrip)
    cruncher1  = DataCruncher("Cruncher1")
    cruncher2  = DataCruncher("Cruncher2")
    cruncher3  = DataCruncher("Cruncher3")
    combiner   = Combiner("Combiner")
    plotter    = Plotter("plotter")

    edges = [
        (ADC, cruncher1),
        (ADC, cruncher2),
        (cruncher3, plotter),
        (cruncher2, cruncher3),
        (cruncher2, combiner),
        (cruncher1, combiner)
    ]

    dag, tasks = create_graph(edges)
    #initialize nodes and check for plot
    have_plots = False
    for n in dag.nodes():
        if isinstance(n, Plotter):
            have_plots = True
        if "init" in dir(n):
            n.init()

    if have_plots:
        t = BokehServerThread()
        t.start()
        #On some systems there is a possibility we try to `push_session` before the
        #the server on the BokehServerThread has started.
        time.sleep(1)
        session = push_session(curdoc())
        print(session.document)
        session.show()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))

    if have_plots:
        print("Joining bokeh server thread")
        t.join()

    nx.draw(dag, with_labels=True)
    plt.show()
