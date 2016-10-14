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
    from libAlazar import LibAlazar
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

class ATS9870(Instrument):
    """Alazar ATS9870 digitizer"""
    instrument_type = "Digitizer"

    def __init__(self, resource_name, name="Unlabeled Alazar"):
        self.name = name

        # Just store the integers here...
        self.channel_numbers = []

        # For lookup
        self._buf_to_chan = {}

        self.resource_name = int(resource_name)
        self.fake = fake_alazar
        if self.fake:
            self._lib = MagicMock()
        else:
            self._lib = LibAlazar()

        commands = ['acquire', 'stop', 'wait_for_acquisition']
        for c in commands:
            setattr(self, c, getattr(self._lib, c))

    def connect(self, resource_name=None):
        if resource_name:
            self.resource_name = resource_name

        self._lib.connectBoard(self.resource_name, "")

    def add_channel(self, channel):
        if not isinstance(channel, AlazarChannel):
            raise TypeError("X6 passed {} rather than an X6Channel object.".format(str(channel)))

        # We can have either 1 or 2, or both.
        if len(self.channel_numbers) < 2 and channel.channel not in self.channel_numbers:
            self.channel_numbers.append(channel.channel)
            self._buf_to_chan[channel] = channel.channel

    def get_buffer_for_channel(self, channel):
        return getattr(self._lib, 'ch{:d}Buffer'.format(self._buf_to_chan[channel]))

    def acquire_all(self, channel=2):
        ch1 = np.array([], dtype=np.float32)
        ch2 = np.array([], dtype=np.float32)

        for _ in range(self.numberAcquisitions):
            while not self.wait_for_acquisition():
                time.sleep(0.0001)
            ch1 = np.append(ch1, self._lib.ch1Buffer)
            ch2 = np.append(ch2, self._lib.ch2Buffer)

        if channel==1:
            return ch1
        else:
            return ch2

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
            'bufferSize',
        ]

        finicky_dict = {k: v for k, v in settings_dict_flat.items() if k in allowed_keywords}
        # TODO: don't do this
        finicky_dict['bufferSize'] = 4096000
        self._lib.setAll(finicky_dict)
        self.number_acquisitions     = self._lib.numberAcquisitions
        self.samples_per_acquisition = self._lib.samplesPerAcquisition
        self.ch1_buffer              = self._lib.ch1Buffer
        self.ch2_buffer              = self._lib.ch2Buffer

    def __del__(self):
        self._lib.disconnect()

    def __repr__(self):
        return "I'm an alazar!"
