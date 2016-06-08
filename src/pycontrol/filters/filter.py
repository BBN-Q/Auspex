from pycontrol.stream import DataStream

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s - %(levelname)s: \t%(asctime)s: \t%(message)s')
logger.setLevel(logging.INFO)

class InputConnector(object):
    def __init__(self, name="", datatype=None, max_input_streams=1):
        self.name = name
        self.stream = None
        self.max_input_streams = max_input_streams
        self.num_input_streams = 0
        self.input_streams = []

    def add_input_stream(self, stream):
        logger.debug("Adding input stream '%s' to input connector %s.", stream, self)
        if self.num_input_streams < self.max_input_streams:
            self.input_streams.append(stream)
            self.num_input_streams += 1
        else:
            raise ValueError("Could not add another input stream to the connector.")

    def connect_to(self, output_connector):
        stream = DataStream()
        self.add_input_stream(stream)
        output_connector.add_output_stream(stream)
        return stream

    def __repr__(self):
        return "<InputConnector(name={})>".format(self.name)

class OutputConnector(object):
    def __init__(self, name="", datatype=None):
        self.name = name
        self.stream = None
        self.output_streams = []

    def add_output_stream(self, stream):
        logger.debug("Adding output stream '%s' to output connector %s.", stream, self)
        self.output_streams.append(stream)

    def connect_to(self, input_connector):
        stream = DataStream()
        self.add_output_stream(stream)
        input_connector.add_input_stream(stream)
        return stream

    def __repr__(self):
        return "<OutputConnector(name={})>".format(self.name)

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
            a = InputConnector(name=ic)
            self.input_connectors[ic] = a
            setattr(self, ic, a)
        for oc in self._output_connectors:
            a = OutputConnector(name=oc)
            self.output_connectors[ic] = a
            setattr(self, oc, a)

    def __repr__(self):
        return "<Filter(name={})>".format(self.name)

    # This default update method be not work for a particular filter
    def update_descriptors(self):
        for oc in self.output_connectors.values():
            for os in oc.output_streams:
                if len(self.input_streams) > 0:
                    os.descriptor = list(oc.values())[0].descriptor
                    os.reset()
        for ic in self.input_connectors:
            for iss in ic.input_streams:
                iss.reset()
