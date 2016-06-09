from pycontrol.stream import DataStream, InputConnector, OutputConnector

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s - %(levelname)s: \t%(asctime)s: \t%(message)s')
logger.setLevel(logging.INFO)

class MetaFilter(type):
    """Meta class to bake the instrument objects into a class description
    """
    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding controls to %s", name)
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
            self.input_connectors[ic] = a
            setattr(self, ic, a)
        for oc in self._output_connectors:
            a = OutputConnector(name=oc, parent=self)
            self.output_connectors[ic] = a
            setattr(self, oc, a)

    def __repr__(self):
        return "<Filter(name={})>".format(self.name)

    # This default update method be not work for a particular filter
    def update_descriptors(self):
        for oc in self.output_connectors.values():
            oc.reset()
            if len(self.input_connectors) > 0:
                oc.set_descriptor(list(self._input_connectors.values())[0].descriptor)
                oc.reset()
        for ic in self.input_connectors.values():
            ic.input_streams.reset()
