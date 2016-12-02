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
    import libx6
    fake_x6 = False
except:
    logger.warning("Could not load x6 library")
    fake_x6 = True

class X6Channel(DigitizerChannel):
    """Channel for an X6"""

    def __init__(self, settings_dict=None):
        self.stream_type       = "Raw"
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

    def __init__(self, resource_name=None, name="Unlabeled X6"):
        # Must have one or more channels, but fewer than XX
        self.channels = []

        self.resource_name = resource_name
        self.name          = name

        if fake_x6:
            self._lib = MagicMock()
        else:
            self._lib = libx6.X6()

        # pass thru functions
        self.acquire    = self._lib.acquire
        self.stop       = self._lib.stop
        self.disconnect = self._lib.disconnect

    def __str__(self):
        return "<X6({}/{})>".format(self.name, self.resource_name)

    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name

        self._lib.connect(int(self.resource_name))

    def set_all(self, settings_dict):
        # Call the non-channel commands
        super(X6, self).set_all(settings_dict)

        # perform channel setup
        for chan in self.channels:
            self.channel_setup(chan)

    def channel_setup(self, channel):
        a, b, c = channel.channel
        self._lib.enable_stream(a, b, c)
        if channel.stream_type == "Raw":
            return
        elif channel.stream_type == "Demodulated":
            self._lib.set_nco_frequency(a, b, channel.if_freq)
        elif channel.stream_type == "Integrated":
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
        if self.data_available():
            return False
        if self._lib.get_is_running():
            return False
        return True

    def add_channel(self, channel):
        if not isinstance(channel, X6Channel):
            raise TypeError("X6 passed {} rather than an X6Channel object.".format(str(channel)))

        if channel.stream_type not in ['Raw', 'Demodulated', 'Integrated']:
            raise ValueError("Stream type of {} not recognized by X6".format(str(channel.stream_type)))

        # todo: other checking here
        self.channels.append(channel)

    def get_buffer_for_channel(self, channel):
        return self._lib.transfer_stream(*channel.channel)

    # pass thru properties
    @property
    def reference(self):
        return self._lib.reference
    @reference.setter
    def reference(self, value):
        self._lib.reference = value

    @property
    def acquire_mode(self):
        return self._lib.acquire_mode
    @acquire_mode.setter
    def acquire_mode(self, value):
        self._lib.acquire_mode = value

    @property
    def record_length(self):
        return self._lib.record_length
    @record_length.setter
    def record_length(self, value):
        self._lib.record_length = value

    @property
    def nbr_waveforms(self):
        return self._lib.nbr_waveforms
    @nbr_waveforms.setter
    def nbr_waveforms(self, value):
        self._lib.nbr_waveforms = value

    @property
    def nbr_segments(self):
        return self._lib.nbr_segments
    @nbr_segments.setter
    def nbr_segments(self, value):
        self._lib.nbr_segments = value

    @property
    def nbr_round_robins(self):
        return self._lib.nbr_round_robins
    @nbr_round_robins.setter
    def nbr_round_robins(self, value):
        self._lib.nbr_round_robins = value
