import asyncio
import numpy as np
from .filter import Filter, InputConnector

class Print(Filter):
    """docstring for Plotter"""

    data = InputConnector()

    def __init__(self, *args, **kwargs):
        super(Print, self).__init__(*args, **kwargs)

    async def run(self):
        while True:
            if self.data.input_streams[0].done():
                print("Printer finished printing.")
                break

            new_data = await self.data.input_streams[0].queue.get()
            print("Printer got new data of size {}: {}".format(np.shape(new_data), new_data))
            print("Printer now has {} of {} points.".format(self.data.input_streams[0].points_taken, self.data.input_streams[0].num_points()))

