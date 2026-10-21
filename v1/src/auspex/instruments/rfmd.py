# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['RFMDAttenuator']

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from auspex.log import logger

class RFMDAttenuator(object):
    """Simple wrapper for using the RFMD voltage controller attenuator.
    Remember that the calibration values for attenuation will be referenced
    to a certain point in the circuit."""

    supply_level = 3.0

    def __init__(self, calibration_file):
        super(RFMDAttenuator, self).__init__()
        self.name = "RFMD VC Attenuator"
        self.df = pd.read_csv(calibration_file, sep=",")
        attenuator_interp = interp1d(self.df["Attenuation"], self.df["Control Voltage"])
        self.attenuator_lookup = lambda x : float(attenuator_interp(x))
        # if self.voltage_control_method = voltage_control_method
        # voltage_supply_method(3.0)

    def set_supply_method(self, func):
        self.voltage_supply_method = func

    def set_control_method(self, func):
        self.voltage_control_method = func

    def minimum_atten(self):
        return (1+1e-3)*np.amin(np.absolute(self.df["Attenuation"]))

    def maximum_atten(self):
        return (1-1e-3)*np.amax(np.absolute(self.df["Attenuation"]))

    # Add a property setter only
    def set_attenuation(self, value):
        value = -abs(value)
        self.voltage_supply_method(self.supply_level)
        self.voltage_control_method(self.attenuator_lookup(value))

    attenuation = property(None, set_attenuation)

    def __repr__(self):
        return self.name
