from pycontrol.stream import DataStream, InputConnector, OutputConnector

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: \t%(message)s')
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
