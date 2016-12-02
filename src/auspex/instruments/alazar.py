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
    from libalazar import ATS9870
    fake_alazar = False
except:
    logger.warning("Could not load alazar library")
    fake_alazar = True

import re

# Convert from pep8 back to camelCase labels
# http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
def camelize(word):
    word = ''.join(x.capitalize() or '_' for x in word.split('_'))
    return word[0].lower() + word[1:]

# Recursively re-label dictionary
def rec_camelize(dictionary):
    new = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v = rec_camelize(v)
        new[camelize(k)] = v
    return new

class AlazarChannel(DigitizerChannel):
    channel = None

    def __init__(self, settings_dict=None):
        if settings_dict:
            self.set_all(settings_dict)

    def set_all(self, settings_dict):
        for name, value in settings_dict.items():
            if hasattr(self, name):
                setattr(self, name, value)

class AlazarATS9870(Instrument):
    """Alazar ATS9870 digitizer"""
    instrument_type = "Digitizer"

    def __init__(self, resource_name=None, name="Unlabeled Alazar"):
        self.name = name
        self.fetch_count = 0

        # Just store the integers here...
        self.channel_numbers = []

        self.resource_name = resource_name

        # For lookup
        self._buf_to_chan = {}

        if fake_alazar:
            self._lib = MagicMock()
        else:
            self._lib = ATS9870()

    def connect(self, resource_name=None):
        if resource_name:
            self.resource_name = resource_name

        self._lib.connect("{}/{}".format(self.name, int(self.resource_name)))

    def acquire(self):
        self._lib.acquire()
        self.fetch_count = 0

    def stop(self):
        self._lib.stop()

    def data_available(self):
        return self._lib.data_available()

    def done(self):
        return self.fetch_count >= (len(self.channel_numbers) * self.number_acquisitions)

    def add_channel(self, channel):
        if not isinstance(channel, AlazarChannel):
            raise TypeError("Alazar passed {} rather than an AlazarChannel object.".format(str(channel)))

        # We can have either 1 or 2, or both.
        if len(self.channel_numbers) < 2 and channel.channel not in self.channel_numbers:
            self.channel_numbers.append(channel.channel)
            self._buf_to_chan[channel] = channel.channel

    def get_buffer_for_channel(self, channel):
        self.fetch_count += 1
        return getattr(self._lib, 'ch{}Buffer'.format(self._buf_to_chan[channel]))

    def set_all(self, settings_dict):
        # Flatten the dict and then pass to super
        settings_dict_flat = {}

        def flatten(dictionary):
            for k, v in dictionary.items():
                if isinstance(v, dict):
                    flatten(v)
                else:
                    settings_dict_flat[k] = v
        flatten(rec_camelize(settings_dict))

        allowed_keywords = [
            'acquireMode',
            'bandwidth',
            'clockType',
            'delay',
            'enabled',
            'label',
            'recordLength',
            'nbrSegments',
            'nbrWaveforms',
            'nbrRoundRobins',
            'samplingRate',
            'triggerCoupling',
            'triggerLevel',
            'triggerSlope',
            'triggerSource',
            'verticalCoupling',
            'verticalOffset',
            'verticalScale',
        ]

        finicky_dict = {k: v for k, v in settings_dict_flat.items() if k in allowed_keywords}

        self._lib.setAll(finicky_dict)
        self.number_acquisitions     = self._lib.numberAcquisitions
        self.samples_per_acquisition = self._lib.samplesPerAcquisition
        self.ch1_buffer              = self._lib.ch1Buffer
        self.ch2_buffer              = self._lib.ch2Buffer

    def disconnect(self):
        self._lib.disconnect()

    def __str__(self):
        return "<AlazarATS9870({}/{})>".format(self.name, self.resource_name)
