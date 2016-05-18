import asyncio
import numpy as np
from bokeh.plotting import figure
from bokeh.client import push_session
from bokeh.plotting import curdoc
from bokeh.driving import cosine
from .stream import ProcessingNode

class Printer(ProcessingNode):
    """docstring for Plotter"""
    def __init__(self, *args):
        super(Printer, self).__init__(*args)

    async def run(self):
        while True:
            if self.input_streams[0].done():
                # We've stopped receiving new input, make sure we've flushed the output streams
                if len(self.output_streams) > 0:
                    if False not in [os.done() for os in self.output_streams]:
                        print("Printer finished printing (clearing outputs).")
                        break
                else:
                    print("Printer finished printing.")
                    break

            new_data = await self.input_streams[0].queue.get()
            print("Got new data of size {}: {}".format(np.shape(new_data), new_data))

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