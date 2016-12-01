# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments.instrument import Instrument, DigitizerChannel
from auspex.log import logger
from unittest.mock import MagicMock

try:
    from libx6 import X6
    fake_x6 = False
except:
    logger.warning("Could not load x6 library")
    fake_x6 = True

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
    """BBN QDSP running on the II-X6 digitizer"""
    instrument_type = "Digitizer"

    def __init__(self, resource_address=None, name="Unlabeled X6"):
        # Must have one or more channels, but fewer than XX
        self.channels = []

        self.resource_address = resource_address
        self.name             = name

    def __str__(self):
        return "<X6({}/{})>".format(self.name, self.resource_name)

    def connect(self, resource_name=None):
        if resource_name:
            self.resource_name = resource_name

        if fake_x6:
            self._lib = MagicMock()
        else:
            self._lib = X6()

        self._lib.connect(int(self.resource_name))

    def disconnect(self):
        self._lib.disconnect()

    def set_all(self, settings_dict):
        # Pop the channel settings
        settings = settings_dict.copy()
        channel_settings = settings.pop('channels')

        # Call the non-channel commands
        super(APS2, self).set_all(settings)

        for chan, ch_settings in enumerate(channel_settings):
            if chan not in self.channels[chan]:
                logger.warning("Channel {} has not been added to X6 {}".format(chan, self))
                continue
            self.channels[chan].set_all(ch_settings)
            # todo: use channel settings to call library functions like:
            # enable_stream, write_kernel, set_threshold, etc.

    def acquire(self):
        self._lib.acquire()

    def stop(self):
        self._lib.stop()

    def data_available(self):
        return self._lib.get_num_new_records() > 0

    def done(self):
        return not self._lib.get_is_running()

    def add_channel(self, channel):
        if not isinstance(channel, X6Channel):
            raise TypeError("X6 passed {} rather than an X6Channel object.".format(str(channel)))

        if channel.stream_type not in ['Raw', 'Demodulated', 'Integrated']:
            raise ValueError("Stream type of {} not recognized by X6".format(str(channel.stream_type)))

        # todo: other checking here
        self.channels.append(channel)

    def get_buffer_for_channel(self, channel):
        return self._lib.transfer_stream(*channel.channel)
