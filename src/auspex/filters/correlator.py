# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import asyncio, concurrent
import itertools
import h5py
import pickle
import zlib
import numpy as np
import os.path
import time

from auspex.parameter import Parameter, FilenameParameter
from auspex.stream import DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger
from .elementwise import ElementwiseFilter

class Correlator(ElementwiseFilter):
    sink   = InputConnector()
    source = OutputConnector()

    def operation(self):
        return np.multiply

    def update_descriptors(self):
        logger.debug('Updating correlator "%s" descriptors based on input descriptor: %s.', self.name, self.sink.descriptor)

        # Sometimes not all of the input descriptors have been updated... pause here until they are:
        if None in [ss.descriptor for ss in self.sink.input_streams]:
            logger.debug('Correlator "%s" waiting for all input streams to be updated.', self.name)
            return

        descriptor = self.sink.descriptor.copy()
        descriptor.data_name = "Correlator"
        if descriptor.unit:
            descriptor.unit = descriptor.unit + "^{}".format(len(self.sink.input_streams))
        self.source.descriptor = descriptor
        self.source.update_descriptors()
