# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Electromagnet']

import numpy as np
import time
from auspex.log import logger

class Electromagnet(object):
    """Wrapper for electromagnet """
    def __init__(self, calibration_file, field_getter, current_setter, current_getter, field_averages=5):
        super(Electromagnet, self).__init__()
        self.name = "Composite Magnet Instrument"
        with open(calibration_file) as cf:
            lines = [l for l in cf.readlines() if l[0] != '#']
            if len(lines) != 1:
                raise Exception("Invalid magnet control calibration file, must contain one line.")
            try:
                # Construct the fit
                poly_coeffs = np.array(lines[0].split(), dtype=np.float)
                self.current_vs_field = np.poly1d(poly_coeffs)
            except:
                raise TypeError("Could not convert calibration coefficients into list of floats")

        self.field_getter = field_getter
        self.current_setter = current_setter
        self.current_getter = current_getter

        self.field_averages = field_averages
        self.calibrated_slope = poly_coeffs[0]

    @property
    def field(self):
        return np.mean( [self.field_getter() for i in range(self.field_averages)] )
    @field.setter
    def field(self, target_field):
        # logging.debug("Appropriate current is: %f" % self.current_vs_field(target_field))
        self.current_setter( self.current_vs_field(target_field) )
        time.sleep(0.6)
        # logging.debug("Arrived at: %f" % self.field)
        field_offset = self.field - target_field
        # logging.debug("Revising: Field offset is %f" % field_offset)
        revised_field = target_field - field_offset
        # logging.debug("Revising: Revised target field is %f" % revised_field)
        self.current_setter( self.current_vs_field(revised_field) )
        # logging.debug("Arrived at: %f, repeat measurement %f" % (self.field, self.field) )

    # hackathon
    def set_field(self, value):
        self.field = value

    # hackathon continues
    def get_field(self):
        return self.field

    def __repr__(self):
        name = "Mystery Instrument" if self.name == "" else self.name
        return "{} @ {}".format(name, self.resource_name)
