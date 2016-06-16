import asyncio
import numpy as np
from .filter import Filter
from pycontrol.stream import InputConnector, OutputConnector

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: \t%(message)s')
logger.setLevel(logging.INFO)

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
                logger.debug("Printer %s finished logger.debuging.", self.name)
                break

            logger.debug("Printer %s awaiting data", self.name)
            new_data = await self.data.input_streams[0].queue.get()
            self.points_taken += np.array(new_data).size
            logger.debug("Printer %s got new data of size %s: %s", self.name, np.shape(new_data), new_data)
            logger.debug("Printer %s now has %s of %s points.", self.name, self.points_taken, self.data.num_points())

        logger.debug("Out of the fracking while loop.")
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
        while True:
            if self.data_in.input_streams[0].done():
                logger.debug("Passthrough %s finished.", self.name)
                break

            logger.debug("Passthrough %s awaiting data", self.name)
            new_data = await self.data_in.input_streams[0].queue.get()
            self.points_taken += np.array(new_data).size
            logger.debug("Passthrough %s got new data of size %s: %s", self.name, np.shape(new_data), new_data)
            logger.debug("Passthrough %s now has %s of %s points.", self.name, self.points_taken, self.data_in.num_points())
            await self.data_out.push(np.ravel(new_data))

        return True