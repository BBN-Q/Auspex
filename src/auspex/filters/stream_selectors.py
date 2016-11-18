# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.parameter import Parameter, IntParameter
from auspex.filters.filter import Filter, InputConnector, OutputConnector

class AlazarStreamSelector(Filter):
    """Digital demodulation and filtering to select a particular frequency multiplexed channel"""

    sink    = InputConnector()
    source  = OutputConnector()
    channel = IntParameter(value_range=(1,2), snap=1)

    def __init__(self, name=""):
        super(AlazarStreamSelector, self).__init__(name=name)
        self.channel.value = 1 # Either 1 or 2
        self.quince_parameters = [self.channel]

class X6StreamSelector(Filter):
    """Digital demodulation and filtering to select a particular frequency multiplexed channel"""

    sink   = InputConnector()
    source = OutputConnector()
    channel       = IntParameter(value_range=(1,3), snap=1)
    demod_channel = IntParameter(value_range=(1,3), snap=1)
    stream_type   = Parameter(allowed_values=["Raw", "Demodulated", "Integrated"], default='Demodulated')

    def __init__(self, name=""):
        super(X6StreamSelector, self).__init__(name=name)
        self.stream_type.value = "Raw" # One of Raw, Demodulated, Integrated
        self.quince_parameters = [self.channel, self.demod_channel, self.stream_type]

    def descriptor_map(self, input_descriptors):
        """Return a dict of the output descriptors."""
        if self.stream_type.value == "Integrated":
            out_descriptor = input_descriptors['sink'].copy()
            try:
                out_descriptor.pop_axis('time')
            except:
                self.out_of_spec = True
                out_descriptor = DataStreamDescriptor()

            return {'source': output_descriptor}
        else:
            return input_descriptors
