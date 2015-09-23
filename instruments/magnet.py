import numpy as np

class Electromagnet(object):
    """Wrapper for electromagnet """
    def __init__(self, calibration_file, field_getter, current_setter, current_getter, field_averages=5):
        super(Electromagnet, self).__init__()
        with open(calibration_file) as cf:
            lines = [l for l in cf.readlines() if l[0] != '#']
            if len(lines) != 1:
                raise Exception("Invalid magnet control calibration file, must contain one line.")
            try:
                poly_coeffs = np.array( map( float, lines[0].split() ) )
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
    def field(self, set_field):
        careful = True

        if careful:
            # Go most of the way there
            initial_field = self.field
            initial_current = self.current_getter()
            print("Initial current %f" % initial_current )
            next_field = initial_field + 0.95*(set_field - initial_field)
            print("Next field %f" % next_field )

            next_current = self.current_vs_field(next_field) 
            print("Corresponding current %f" % next_current )
            self.current_setter( next_current )

            # Calculate the actual Delta_field/Delta_current
            actual_field = self.field
            print("Actually went to field %f" % actual_field )
            new_slope = (next_current - initial_current) / (actual_field - initial_field)
            print("New slope %f" % new_slope )
            
            # Go the rest of the way
            next_current = initial_current + new_slope*(set_field - initial_field)
            print("Next current %f" % next_current )
            self.current_setter( next_current )

            print("Final field %f" % self.field )

        else:
            self.current_setter( self.current_vs_field(set_field) )
