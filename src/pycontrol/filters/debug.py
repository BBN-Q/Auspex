import asyncio
import numpy as np
from .filter import Filter
from pycontrol.stream import InputConnector

class Print(Filter):
    """docstring for Plotter"""

    data = InputConnector()

    def __init__(self, *args, **kwargs):
        super(Print, self).__init__(*args, **kwargs)

    async def run(self):
        if self.name is None:
            self.name = ""

        self.points_taken = 0 
        while True:
            if self.data.input_streams[0].done():
                print("Printer {} finished printing.".format(self.name))
                break

            print("Printer {} awaiting data".format(self.name))
            new_data = await self.data.input_streams[0].queue.get()
            self.points_taken += np.array(new_data).size
            print("Printer {} got new data of size {}: {}".format(self.name, np.shape(new_data), new_data))
            print("Printer {} now has {} of {} points.".format(self.name, self.points_taken, self.data.num_points()))

        print("Out of the fracking while loop.")
        return True
            