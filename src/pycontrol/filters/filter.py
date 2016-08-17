import asyncio
from concurrent.futures import FIRST_COMPLETED

from pycontrol.stream import DataStream, InputConnector, OutputConnector
from pycontrol.logging import logger

class MetaFilter(type):
    """Meta class to bake the input/output connectors into a Filter class description
    """
    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding connectors to %s", name)
        self._input_connectors  = []
        self._output_connectors = []
        for k,v in dct.items():
            if isinstance(v, InputConnector):
                logger.debug("Found '%s' input connector.", k)
                self._input_connectors.append(k)
            elif isinstance(v, OutputConnector):
                logger.debug("Found '%s' output connector.", k)
                self._output_connectors.append(k)

class Filter(metaclass=MetaFilter):
    """Any node on the graph that takes input streams with optional output streams"""
    def __init__(self, name=None):
        self.name = name
        self.input_connectors = {}
        self.output_connectors = {}

        for ic in self._input_connectors:
            a = InputConnector(name=ic, parent=self)
            a.parent = self
            self.input_connectors[ic] = a
            setattr(self, ic, a)
        for oc in self._output_connectors:
            a = OutputConnector(name=oc, parent=self)
            a.parent = self
            self.output_connectors[oc] = a
            setattr(self, oc, a)

    def __repr__(self):
        return "<Filter(name={})>".format(self.name)

    def update_descriptors(self):
        self.descriptor = list(self.input_connectors.values())[0].descriptor
        logger.debug("Starting descriptor update in filter %s, where the descriptor is %s",
                self.name, self.descriptor)
        for oc in self.output_connectors.values():
            oc.descriptor = self.descriptor
            oc.update_descriptors()

    async def run(self):
        """
        Generic run method which waits on a single stream and calls `process_data` on any new_data
        """
        logger.debug('Running "%s" run loop', self.name)

        input_stream = getattr(self, self._input_connectors[0]).input_streams[0]

        while True:

            #setup futures that will return either stream is done or new  data
            get_finished_task = asyncio.ensure_future(input_stream.finished())
            get_data_task = asyncio.ensure_future(input_stream.queue.get())

            done, pending = await asyncio.wait((get_finished_task, get_data_task),
                                     return_when=FIRST_COMPLETED)

            #check whether stream finished returned first in which case we break out and finish
            if get_finished_task in done and input_stream.queue.empty():
                logger.info('No more data for %s "%s"', self.__class__.__name__, self.name)
                get_data_task.cancel()
                break

            #otherwise cancel the finish check and process new data
            get_finished_task.cancel()
            new_data = get_data_task.result()
            logger.debug('%s "%s" received %d points.', self.__class__.__name__, self.name, new_data.size)

            await self.process_data(new_data)

    async def process_data(self, data):
        """Generic pass through.  """
        return data
