import asyncio, concurrent
import numpy as np
from pycontrol.filters.filter import Filter
from pycontrol.stream import InputConnector, OutputConnector
from pycontrol.logging import logger

class Print(Filter):
    """Debug printer that prints data comming through filter"""

    data = InputConnector()

    def __init__(self, *args, **kwargs):
        super(Print, self).__init__(*args, **kwargs)

    async def process_data(self, data):

        logger.debug('%s "%s" received points: %s', self.__class__.__name__, self.name, data)

class Passthrough(Filter):
    data_in  = InputConnector()
    data_out = OutputConnector()

    def __init__(self, *args, **kwargs):
        super(Passthrough, self).__init__(*args, **kwargs)

    async def process_data(self, data):
        for os in self.data_out.output_streams:
            await os.push(data)
