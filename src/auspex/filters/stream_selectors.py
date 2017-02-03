# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.log import logger
from auspex.instruments import *
from auspex.parameter import Parameter, IntParameter
from .filter import Filter
from auspex.stream import DataStreamDescriptor, DataAxis, InputConnector, OutputConnector

import numpy as np

class AlazarStreamSelector(Filter):
    """Digital demodulation and filtering to select a particular frequency multiplexed channel"""

    sink    = InputConnector()
    source  = OutputConnector()
    channel = IntParameter(value_range=(1,2), snap=1)

    def __init__(self, name=""):
        super(AlazarStreamSelector, self).__init__(name=name)
        self.channel.value = 1 # Either 1 or 2
        self.quince_parameters = [self.channel]

    def get_descriptor(self, source_instr_settings, channel_settings):
        channel = AlazarChannel(channel_settings)

        # Add the time axis
        samp_time = 1.0/source_instr_settings['sampling_rate']
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("time", samp_time*np.arange(source_instr_settings['record_length'])))
        return channel, descrip

class X6StreamSelector(Filter):
    """Digital demodulation and filtering to select a particular frequency multiplexed channel"""

    sink   = InputConnector()
    source = OutputConnector()
    phys_channel  = IntParameter(value_range=(1,3), snap=1)
    dsp_channel   = IntParameter(value_range=(0,4), snap=1)
    stream_type   = Parameter(allowed_values=["Raw", "Demodulated", "Integrated"], default='Demodulated')

    def __init__(self, name=""):
        super(X6StreamSelector, self).__init__(name=name)
        self.stream_type.value = "Raw" # One of Raw, Demodulated, Integrated
        self.quince_parameters = [self.phys_channel, self.dsp_channel, self.stream_type]

    def get_descriptor(self, source_instr_settings, channel_settings):
        # Create a channel
        channel = X6Channel(channel_settings)

        descrip = DataStreamDescriptor()
        # If it's an integrated stream, then the time axis has already been eliminated.
        # Otherswise, add the time axis.
        if channel_settings['stream_type'] == 'Raw':
            samp_time = 4.0e-9
            descrip.add_axis(DataAxis("time", samp_time*np.arange(source_instr_settings['record_length']//4)))
            descrip.dtype = np.float64
        elif channel_settings['stream_type'] == 'Demodulated':
            samp_time = 32.0e-9
            descrip.add_axis(DataAxis("time", samp_time*np.arange(source_instr_settings['record_length']//32)))
            descrip.dtype = np.complex128
        else: # Integrated
            descrip.dtype = np.complex128

        return channel, descrip

    # work in progress on "out of spec" descriptors.
    # def descriptor_map(self, input_descriptors):
    #     """Return a dict of the output descriptors."""
    #     if self.stream_type.value == "Integrated":
    #         out_descriptor = input_descriptors['sink'].copy()
    #         out_descriptor.dtype = np.complex128
    #         try:
    #             out_descriptor.pop_axis('time')
    #         except:
    #             self.out_of_spec = True
    #             out_descriptor = DataStreamDescriptor()
    #
    #         return {'source': output_descriptor}
    #     elif self.stream_type == 'Demodulated':
    #         out_descriptor = input_descriptors['sink'].copy()
    #         out_descriptor.dtype = np.complex128
    #         return {'source': output_descriptor}
    #     else:
    #         return {'source': input_descriptors['sink']}
