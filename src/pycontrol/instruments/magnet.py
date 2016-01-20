import numpy as np
import time
import logging

class Electromagnet(object):
    """Wrapper for electromagnet """
    def __init__(self, calibration_file, field_getter, current_setter, current_getter, field_averages=5):
        super(Electromagnet, self).__init__()
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
        # logging.info("Appropriate current is: %f" % self.current_vs_field(target_field))
        self.current_setter( self.current_vs_field(target_field) )
        time.sleep(0.6)
        # logging.info("Arrived at: %f" % self.field)
        field_offset = self.field - target_field
        # logging.info("Revising: Field offset is %f" % field_offset)
        revised_field = target_field - field_offset
        # logging.info("Revising: Revised target field is %f" % revised_field)
        self.current_setter( self.current_vs_field(revised_field) )
        # logging.info("Arrived at: %f, repeat measurement %f" % (self.field, self.field) )

    # hackathon
    def set_field(self, value):
        self.field = value

    # hackathon continues
    def get_field(self):
        return self.field
