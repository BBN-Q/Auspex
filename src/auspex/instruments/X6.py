# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.instrument import Instrument, DigitizerChannel
from pycontrol.log import logger

class X6Channel(DigitizerChannel):
    """Channel for an X6"""
    stream_type       = None
    if_freq           = None
    demod_kernel      = None
    demod_kernel_bias = None
    raw_kernel        = None
    raw_kernel_bias   = None
    threshold         = None
    threshold_invert  = None

    phys_chan   = 0
    demod_chan  = 0
    result_chan = 0
    channel     = (0,0,0)
    
    def __init__(self, settings_dict=None):
        if settings_dict:
            self.set_all(settings_dict)

    def set_all(self, settings_dict):
        for name, value in settings_dict.items():
            if hasattr(self, name):
                setattr(self, name, value)

        if self.stream_type == "Integrated":
            self.result_chan = 1
        elif self.stream_type == "Demodulated":
            self.result_chan = 0
        else: #Raw
            self.result_chan = 0
            self.demod_chan  = 0

        self.channel = (self.phys_chan, self.demod_chan, self.result_chan)

class X6(Instrument):
    """Alazar X6 digitizer"""
    instrument_type = "Digitizer"

    def __init__(self, resource_address=None, name="Unlabeled X6"):
        # Must have one or more channels, but fewer than XX
        self.channels = []

        self.resource_address = resource_address
        self.name             = name

        # For lookup
        self._buf_to_chan = {}

    def add_channel(self, channel):
        if not isinstance(channel, X6Channel):
            raise TypeError("X6 passed {} rather than an X6Channel object.".format(str(channel)))

        if channel.stream_type not in ['Raw', 'Demodulated', 'Integrated']:
            raise ValueError("Stream type of {} not recognized by X6".format(str(channel.stream_type)))

        # todo: other checking here
        self.channels.append(channel)
        self._buf_to_chan[channel] = channel.channel

    def get_buffer_for_channel(self, channel):
        return getattr(self._lib, 'ch{:d}Buffer'.format(self._buf_to_chan[channel]))