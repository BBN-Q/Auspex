# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Correlator']

import numpy as np

from auspex.stream import InputConnector, OutputConnector
from auspex.log import logger
from .elementwise import ElementwiseFilter

class Correlator(ElementwiseFilter):
    sink        = InputConnector()
    source      = OutputConnector()
    filter_name = "Correlator"

    def operation(self):
        return np.multiply

    def unit(self, base_unit):
        return base_unit + "^{}".format(len(self.sink.input_streams))
