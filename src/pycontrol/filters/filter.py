class InputConnector(object):
    def __init__(self, name=None, datatype=None, max_input_streams=1):
        super(InputConnector, self).__init__()
        self.name = name
        self.stream = None
        self.max_input_streams = max_input_streams
        self.num_input_streams = 0
        self.input_streams = []

    def add_input_stream(self, stream):
        if self.num_input_streams < self.max_input_streams:
            self.input_streams.append(stream)
            self.num_input_streams += 1
        else:
            raise ValueError("Could not add another input stream to the connector.")

class OutputConnector(object):
    def __init__(self, name=None, datatype=None):
        super(OutputConnector, self).__init__()
        self.name = name
        self.stream = None
        self.output_streams = []

    def add_output_stream(self, stream):
        self.output_streams.append(stream)

class MetaFilter(type):
    """Meta class to bake the instrument objects into a class description
    """
    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        logger.debug("Adding controls to %s", name)
        self.input_connectors  = {}
        self.output_connectors = {}

        for k,v in dct.items():
            if isinstance(v, InputConnector):
                logger.debug("Found '%s' input connector.", k)
                if v.name is None:
                    v.name = k
                self.input_connectors[k] = v
            elif isinstance(v, OutputConnector):
                logger.debug("Found '%s' output connector.", k)
                if v.name is None:
                    v.name = k
                self.output_connectors[k] = v

class Filter(object):
    """Any node on the graph that takes input streams with optional output streams"""
    def __init__(self, label=None):
        super(Filter, self).__init__()
        self.label = label
        self.input_connectors  = {}
        self.output_connectors = {}

    def __str__(self):
        return str(self.label)

    # This default update method be not work for a particular filter
    def update_descriptors(self):
        for oc in self.output_connectors.values():
            for os in oc.output_streams:
                if len(self.input_streams) > 0:
                    os.descriptor = list(oc.values())[0].descriptor