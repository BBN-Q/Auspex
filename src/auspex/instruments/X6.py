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

    def __init__(self, settings_dict=None):
        self.stream_type       = "Physical"
        self.if_freq           = 0.0
        self.kernel            = None
        self.kernel_bias       = 0.0
        self.threshold         = 0.0
        self.threshold_invert  = False

        self.phys_chan   = 1
        self.demod_chan  = 0
        self.result_chan = 0
        self.channel     = (1,0,0)

        if settings_dict:
            self.set_all(settings_dict)

    def set_all(self, settings_dict):
        for name, value in settings_dict.items():
            if hasattr(self, name):
                setattr(self, name, value)

        if self.stream_type == "Integrated":
            self.result_chan = 1
            self.demod_chan = 0
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

        if fake_x6:
            self._lib = MagicMock()
        else:
            self._lib = X6()

        # pass thru functions
        self.acquire    = self._lib.acquire
        self.stop       = self._lib.stop
        self.disconnect = self._lib.disconnect

        # pass thru properties
        self.record_length    = self._lib.record_length
        self.nbr_waveforms    = self._lib.nbr_waveforms
        self.nbr_segments     = self._lib.nbr_segments
        self.nbr_round_robins = self._lib.nbr_round_robins
        self.reference        = self._lib.reference
        self.acquire_mode     = self._lib.acquire_mode

    def __str__(self):
        return "<X6({}/{})>".format(self.name, self.resource_name)

    def connect(self, resource_name=None):
        if resource_name:
            self.resource_name = resource_name

        self._lib.connect(int(self.resource_name))

    def set_all(self, settings_dict):
        # Call the non-channel commands
        super(APS2, self).set_all(settings)

        # perform channel setup
        for chan in self.channels:
            self.channel_setup(chan)

    def channel_setup(self, channel, settings):
        a, b, c = channel.channel
        if channel.stream_type == "Physical":
            self._lib.enable_stream(a, 0, 0)
            return
        elif channel.stream_type == "Demodulated":
            self._lib.set_nco_freq(a, b, channel.if_freq)

            if channel.kernel:
                self._lib.enable_stream(a, b, 1)
                self._lib.write_kernel(a, b, 1, channel.kernel)
                self._lib.set_kernel_bias(a, b, 1, channel.kernel_bias)
            else:
                self._lib.enable_stream(a, b, 0)
        elif channel.stream_type == "Integrated":
            self._lib.enable_stream(a, b, 1)
            if not channel.kernel:
                logger.error("Integrated streams must specify a kernel")
                return
            self._lib.write_kernel(a, b, c, channel.kernel)
            self._lib.set_kernel_bias(a, b, c, channel.kernel_bias)
            self._lib.set_threshold(a, b, channel.threshold)
            self._lib.set_threshold_invert(a, b, channel.threshold_invert)
        else:
            logger.error("Unrecognized stream type %s" % channel.stream_type)

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
