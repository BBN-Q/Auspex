# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['SingleShotMeasurement']

import numpy as np

from .filter import Filter
from auspex.parameter import Parameter, FloatParameter, IntParameter, BoolParameter
from auspex.stream import DataStreamDescriptor, InputConnector, OutputConnector
from auspex.log import logger

class SingleShotMeasurement(Filter):

    save_kernel = BoolParameter()
    optimal_integration_time = BoolParameter()
    logistic_regression = BoolParameter()

    def __init__(self, **kwargs):
        super(SingleShotMeasurement, self).__init__(**kwargs)
        if len(kwargs) > 0:
            self.save_kernel.value = kwargs['save_kernel']
            self.optimal_integration_time.value = kwargs['optimal_integration_time']
            self.logistic_regression.value = False #To be implemented

    def update_descriptors(self):
        pass

    async def process_data(self, data):
        pass

    async def on_done(self, data):
        pass

    
