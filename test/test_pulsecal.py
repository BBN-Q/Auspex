# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest
import numpy as np

import auspex.pulsecal.phase_estimation as pe


class PhaseEstimateTestCase(unittest.TestCase):

    def test_simulated_measurement(self):
        
        numPulses = 9
        amp = .55
        direction = 'X'
        target = np.pi
        
        # Using the same simulated data as matlab
        data, vardata =  pe.simulate_measurement(amp, target, numPulses)   
        
        # Verify output matches what was previously seen by matlab
        phase, sigma = pe.phase_estimation(data, vardata, verbose=True)
        self.assertAlmostEqual(phase,-1.2012,places=4)
        self.assertAlmostEqual(sigma,0.0245,places=4)


if __name__ == '__main__':
    unittest.main()
