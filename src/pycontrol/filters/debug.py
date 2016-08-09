import asyncio, concurrent
import numpy as np
from pycontrol.filters.filter import Filter
from pycontrol.stream import InputConnector, OutputConnector
from pycontrol.logging import logger

class Print(Filter):
    """docstring for Plotter"""

    data = InputConnector()

    def __init__(self, *args, **kwargs):
        super(Print, self).__init__(*args, **kwargs)

    async def run(self):
        if self.name is None:
            self.name = ""

        self.points_taken = 0
        stream = self.data.input_streams[0]
        while True:
            done, pending = await asyncio.wait((stream.finished(), stream.queue.get()),
                                     return_when=concurrent.futures.FIRST_COMPLETED)
            for axis, d in zip(self.descriptor.axes, self.descriptor.axes_done()):
                logger.debug("Descriptor %s doneness: %s", axis, d)

            new_data = list(done)[0].result()
            if type(new_data)==bool:
                if new_data:
                    logger.debug("Printer %s finished logger.debuging.", self.name)
                    break
                else:
                    logger.debug("Printer %s awaiting data", self.name)
                    new_data = await stream.queue.get()

            self.points_taken += np.array(new_data).size
            logger.debug("Printer %s got new data of size %s: %s", self.name, np.shape(new_data), new_data)
            logger.debug("Printer %s now has %s of %s points.", self.name, self.points_taken, self.data.num_points())

        return True

class Passthrough(Filter):
    data_in  = InputConnector()
    data_out = OutputConnector()

    def __init__(self, *args, **kwargs):
        super(Passthrough, self).__init__(*args, **kwargs)

    async def run(self):
        if self.name is None:
            self.name = ""

        self.points_taken = 0
        stream = self.data.input_streams[0]
        while True:
            if stream.done:
                logger.debug("Passthrough %s finished.", self.name)
                break

            logger.debug("Passthrough %s awaiting data", self.name)
            new_data = await stream.queue.get()
            self.points_taken += np.array(new_data).size
            logger.debug("Passthrough %s got new data of size %s: %s", self.name, np.shape(new_data), new_data)
            logger.debug("Passthrough %s now has %s of %s points.", self.name, self.points_taken, self.data_in.num_points())
            await self.data_out.push(np.ravel(new_data))

        return True
