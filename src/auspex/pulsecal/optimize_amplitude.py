# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from auspex.pulsecal.phase_estimation import *

def optimize_amplitude( amp_init, direction, target, update_data, numShots=1):

    """Attempts to optimize the pulse amplitude for a pi/2 or pi pulse about X or Y. """

    done = False
    ct = 1
    amp = amp_init
    while done == False:
        
        # NOTE: wait_for_data is a place holder to simulate an AWG generating
        # a sequence and a digitizer receiving the sequence.  This function
        # is passed into the optimize_amplitude routine to be able to update
        # the amplitude as part of the optimization loop.
        data, vardata = update_data(amp)
        
        [phase, sigma] = phase_estimation(data, vardata/numShots)
        print("Phase: %.4f Sigma: %.4f"%(phase,sigma))
        # correct for some errors related to 2pi uncertainties
        if np.sign(phase) != np.sign(amp):
            phase += np.sign(amp)*2*np.pi
        angleError = phase - target;
        print('Angle error: %.4f'%angleError);

        ampTarget = target/phase * amp
        ampError = amp - ampTarget
        print('Amplitude error: %.4f\n'%ampError)

        amp = ampTarget
        ct += 1

        # check for stopping condition
        phaseError = phase - target
        if np.abs(phaseError) < 1e-2 or np.abs(phaseError/sigma) < 1 or ct > 5:
            if np.abs(phaseError) < 1e-2:
                print('Reached target rotation angle accuracy');
            elif abs(phaseError/sigma) < 1:
                print('Reached phase uncertainty limit');
            else:
                print('Hit max iteration count');
            done = True;
    print('Amp',amp)
    
    return amp