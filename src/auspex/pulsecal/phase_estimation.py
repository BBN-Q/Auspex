# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt

def restrict(phase):
    out = np.mod( phase + np.pi, 2*np.pi, ) - np.pi
    return out


def phase_estimation( data_in, vardata_in, verbose=False):
    """Estimates pulse rotation angle from a sequence of P^k experiments, where
    k is of the form 2^n. Uses the modified phase estimation algorithm from
    Kimmel et al, quant-ph/1502.02677 (2015). Every experiment i doubled.
    vardata should be the variance of the mean"""     

    #average together pairs of data points
    avgdata = (data_in[0::2] + data_in[1::2])/2
    
    # normalize data using the first two pulses to calibrate the "meter"
    data = 1 + np.divide( 2*(avgdata[2:] - avgdata[0]), (avgdata[0] - avgdata[1]))
    zdata = data[0::2]
    xdata = data[1::2]

    # similar scaling with variances
    vardata = (vardata_in[0::2] + vardata_in[1::2])/2
    vardata = np.divide( vardata[2:] * 2, abs(avgdata[0] - avgdata[1])**2)
    zvar = vardata[0::2]
    xvar = vardata[1::2]

    phases = np.arctan2(xdata, zdata)
    distances = np.sqrt(xdata**2 + zdata**2);

    curGuess = phases[0]
    phase = curGuess
    sigma = np.pi
    
    if verbose == True:
        print('Current Guess: %f'%(curGuess))
        
    for k in np.arange(1,len(phases)):
        
        if verbose == True:
            print('k: %d'%(k))
            
        # Each step of phase estimation needs to assign the measured phase to
        # the correct half circle. We will conservatively require that the
        # (x,z) tuple is long enough that we can assign it to the correct
        # quadrant of the circle with 2σ confidence
        if distances[k] < 2*np.sqrt(xvar[k] + zvar[k]):
            print('Phase estimation terminated at %dth pulse because the (x,z) vector is too short'%(k));
            break
        
        lowerBound = restrict(curGuess - np.pi/2**(k));
        upperBound = restrict(curGuess + np.pi/2**(k));
        possiblesTest = [ restrict((phases[k] + 2*n*np.pi)/2**(k)) for n in np.arange(0,2**(k)+1)]
        
        if verbose == True:
            print('Lower Bound: %f'%lowerBound)
            print('Upper Bound: %f'%upperBound)
        
        possibles=[]
        for p in possiblesTest:
            # NOTE: previous code did not handle upperbound == lowerBound
            if lowerBound >= upperBound:
                satisfiesLB = p > lowerBound or p < 0.
                satisfiesUP = p < upperBound or p > 0.
            else:
                satisfiesLB = p > lowerBound 
                satisfiesUP = p < upperBound 
            
            if satisfiesLB == True and satisfiesUP == True:
                possibles.append(p)
                    
        curGuess = possibles[0];
        if verbose == True:
            print('Current Guess: %f'%(curGuess))
    phase = curGuess;
    sigma = np.maximum(np.abs(restrict(curGuess - lowerBound)), np.abs(restrict(curGuess - upperBound)));
    
    return phase,sigma


def simulate_measurement(amp, target, numPulses):
    
    idealAmp = 0.34
    noiseScale = 0.05
    polarization = 0.99 # residual polarization after each pulse

    # data representing over/under rotation of pi/2 pulse
    # theta = pi/2 * (amp/idealAmp);
    theta = target * (amp/idealAmp)
    ks = [ 2**k for k in np.arange(0,numPulses+1)]
    
    xdata = [ polarization**x * np.sin(x*theta) for x in ks];
    xdata = np.insert(xdata,0,-1.0)
    zdata = [ polarization**x * np.cos(x*theta) for x in ks];
    zdata = np.insert(zdata,0,1.0)
    data = np.array([zdata,xdata]).flatten('F')
    data = np.tile(data,(2,1)).flatten('F')
    
    # add noise
    #data += noiseScale * np.random.randn(len(data));
    vardata = noiseScale**2 * np.ones((len(data,)));
    
    return data, vardata
    