# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

class HallProbe(object):
    """Simple wrapper for converting Hall probe voltage measurements to
    actual fields values."""
    def __init__(self, calibration_file, supply_voltage_method, readout_voltage_method):
        super(HallProbe, self).__init__()
        self.name = "Lakeshore Hall Probe"
        with open(calibration_file) as cf:
            lines = [l for l in cf.readlines() if l[0] != '#']
            if len(lines) != 2:
                raise Exception("Invalid Hall probe calibration file, must contain two lines.")
            try:
                self.output_voltage = float(lines[0])
            except:
                raise TypeError("Could not convert output voltage to floating point value.")
            try:
                poly_coeffs = np.array(lines[1].split(), dtype=np.float)
                self.field_vs_voltage = np.poly1d(poly_coeffs)
            except:
                raise TypeError("Could not convert calibration coefficients into list of floats")
        self.getter = readout_voltage_method
        self.setter = supply_voltage_method

        self.setter(self.output_voltage)

    @property
    def field(self):
        return self.get_field()

    def get_field(self):
        return self.field_vs_voltage( self.getter() )

    def __repr__(self):
        name = "Mystery Instrument" if self.name == "" else self.name
        return "{} @ {}".format(name, self.resource_name)
