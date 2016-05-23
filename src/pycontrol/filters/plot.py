import asyncio
from bokeh.plotting import figure
from bokeh.client import push_session
from bokeh.plotting import curdoc
from bokeh.driving import cosine
from .filter import Filter

class Plot(Filter):
    """docstring for Plotter"""
    def __init__(self, *args, **kwargs):
        super(Plotter, self).__init__(*args, **kwargs)

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