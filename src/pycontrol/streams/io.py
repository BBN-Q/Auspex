import asyncio
import numpy as np

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