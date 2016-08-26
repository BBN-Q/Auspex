# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.instrument import Instrument
from libAlazar import LibAlazar
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

class ATS9870(Instrument):
    """Alazar ATS9870 digitizer"""
    instrument_type = "Digitizer"
    _lib = LibAlazar()

    def __init__(self, resource_name, *args, **kwargs):
        #If we only have an IP address then tack on the raw socket port to the VISA resource string
        #super(ATS9870, self).__init__(resource_name, *args, **kwargs)
        pass
        self.name = "ATS9870"
        self.resource_name = int(resource_name)
        self._freeze()

        self._lib.connectBoard(self.resource_name, "")

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
        self._lib.setAll(finicky_dict)

    def __del__(self):
        self._lib.disconnect()

    def __repr__(self):
        return "I'm an alazar!"