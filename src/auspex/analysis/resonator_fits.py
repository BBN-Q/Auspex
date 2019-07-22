# Copyright 2019 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy
import numpy as np
import scipy
import scipy.stats
from matplotlib import pyplot
from scipy.optimize import newton
from numpy.linalg import det

from .fits import AuspexFit

class ResonatorCircleFit(AuspexFit):

    def __init__(self, data, freqs, initial_Qc=None, make_plots=False):
        assert len(freqs) == len(data), "Length of X and Y points must match!"
        self.data = data 
        self.freqs = freqs 
        self.make_plots 
        self.initial_Qc = initial_Qc
        self._do_fit()

    @staticmethod
    def _model(x, *p):
        scaling = p[1] * np.exp(1j*p[2]) * np.exp(-2j*np.pi * x * p[0])
        A = (p[5]/np.abs(p[6])) * np.exp(-1j * p[4])
        B = 1 + 2j*p[5]*(x / p[3] - 1)
        return scaling*(1.0-A/B)

    def _do_fit(self):

        result = resonator_circle_fit(self.freqs, self.data, makePlots=self.make_plots, 
                                                            manual_qc=self.initial_Qc) 

        popt = result[:-1]
        self.fit_params = {"tau": popt[0],
                           "a": popt[1],
                           "alpha": popt[2],
                           "fr": popt[3],
                           "phi0": popt[4],
                           "Ql": popt[5],
                           "Qc": popt[6],
                           "Qi": popt[7]}

        self.fit_errors = result[-1]
        self.fit_function = lambda x: self._model(x, *popt)

        fit = np.array([self.fit_function(f) for f in self.freqs])
        self.sq_error = np.sum(np.abs(fit - self.data)**2)
        dof = len(self.freqs) - len(popt)
        #See AuspexFit class for explanation of Nsigma
        self.Nsigma = self.sq_error/np.sqrt(2*dof) - dof/np.sqrt(2*dof) 

    def __str__(self):
        return "Resonator Circle Fit"


def circle_fit(data, freqs):
    '''
    Fits a trace of data to a circle.
    Parameters:
    data (numpy array of complex): S21 data
    freqs (numpy array of reals): frequency of each data point index
    
    Returns:
    (r,xc,yc): tuple of fit parameters
    '''
    # Apply the cable transformation and generate the matrix of moments
    M = numpy.zeros((4,4))
    M[3][3] = len(data)
    
    for point in data:       
        # Add next term in moment summations
        M[0][0] += numpy.absolute(point)**4                     # Mzz
        M[0][1] += numpy.real(point) * numpy.absolute(point)**2 # Mxz
        M[0][2] += numpy.imag(point) * numpy.absolute(point)**2 # Myz
        M[0][3] += numpy.absolute(point)**2                     # Mz
        M[1][0] += numpy.real(point) * numpy.absolute(point)**2 # Mxz
        M[1][1] += numpy.real(point)**2                         # Mxx
        M[1][2] += numpy.real(point) * numpy.imag(point)        # Mxy
        M[1][3] += numpy.real(point)                            # Mx
        M[2][0] += numpy.imag(point) * numpy.absolute(point)**2 # Myz
        M[2][1] += numpy.real(point) * numpy.imag(point)        # Mxy
        M[2][2] += numpy.imag(point)**2                         # Myy
        M[2][3] += numpy.imag(point)                            # My
        M[3][0] += numpy.absolute(point)**2                     # Mz
        M[3][1] += numpy.real(point)                            # Mx
        M[3][2] += numpy.imag(point)                            # My
        
    # Create the constraint matrix
    bmat = numpy.asarray([[0,0,0,-2],
                          [0,1,0,0],
                          [0,0,1,0],
                          [-2,0,0,0]])
    
    # Define some lambda functions for the characteristic polynomial and its derivative
    characteristic_poly = lambda n: det(M - n*bmat)
    
    minor = lambda mat,i,j: det(numpy.asarray([numpy.concatenate([mat[row][:j],mat[row][j+1:]]) for row in range(mat.shape[0]) if row != i]))
    cofactor = lambda mat: numpy.asarray([[(-1)**(i+j)*minor(mat,i,j) for j in range(len(mat[i]))] for i in range(mat.shape[0])])
    characteristic_poly_prime = lambda n: numpy.trace(-numpy.matmul(numpy.transpose(cofactor(M - n*bmat)), bmat))
    
    # Now, use Newton's method starting at 0 to find n such that det(M-nB) = 0
    #nmin = newton(characteristic_poly, 0, fprime=characteristic_poly_prime, tol=1e-15, maxiter=200000)
    nmin = newton(characteristic_poly, 0, fprime=characteristic_poly_prime, maxiter=20000)
        
    # Finally, solve the matrix equation (M - nB)A = 0
    A = scipy.linalg.null_space(M - nmin*bmat).flatten()
    
    return [1.0/(2.0*numpy.absolute(A[0]))*numpy.sqrt(A[1]**2 + A[2]**2 - 4*A[0]*A[3]), -A[1]/(2.0*A[0]), -A[2]/(2.0*A[0])]

def _circle_residuals(tau, data, freqs):
    '''
    Computes the error residuals from the circle fit for a given tau
    Parameters:
    tau (numpy real) : cable delay in ns
    data (numpy complex array): s21 data
    freqs (numpy real array)  : test frequencies
    
    Returns:
    Vector of error residuals resulting from fitting trace to a circle
    '''
    transformed_data = data * numpy.exp(2*numpy.pi*1j*freqs*1e-9*tau)
    [r,xc,yc] = circle_fit(transformed_data, freqs)
    return numpy.sum(r**2 - ((numpy.real(transformed_data)-xc)**2 + (numpy.imag(transformed_data)-yc)**2))

def resonator_circle_fit(data, freqs, makePlots=False, manual_qc=None):
    '''
    Fits a data trace to a resonance circle by finding the corresponding environmental parameters and 
    resonator properties.
    Parameters:
    data (numpy array of complex): S21 data
    freqs (numpy array of reals): frequency of each data point index
    makePlots (boolean, optional): if True, generates line plots of fits. intended for use in Jupyter notebooks; call the following two lines before executing this function with makePlots=True:
        %matplotlib inline
        import matplotlib.pyplot as pyplot
    manual_qc (numpy complex, optional): provide a manual value for the coupling Q rather than computing it from the data
    
    Returns:
    [tau, a, alpha, fr, phi0, Ql, Qc, Qi, sigma_Qi]
    tau: cable delay
    a: environmental gain factor
    alpha: environmental rotation
    fr: resonant frequency
    phi0: impedance mismatch parameter
    Ql: loaded quality factor
    Qc: coupling quality factor
    Qi: internal quality factor
    sigma_Qi: error in Qi 
    '''
    axes = None
    
    if(makePlots):
        # Set up the plots
        fig, axes = pyplot.subplots(2,2)
        pyplot.autoscale(tight=True)
        pyplot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # First plot the original data
        axes[0,0].scatter(data.real, data.imag, s=0.5)
        axes[0,0].autoscale_view(tight=True)
        axes[0,0].figure.set_size_inches(10,10)
        axes[0,0].set_xlabel('Re[S21]')
        axes[0,0].set_ylabel('Im[S21]')
        axes[0,0].set_title('Raw S21 Data')
    
    # First, do a least-squares optimization to determine the cable delay
    # Determine the bound for "prying open" the resonance circle
    # The points on the circle all take the form exp(2 pi i f t)
    # The phase of this is 2 pi f t
    # Hence, taking a linear regression against f and finding the slope can give you
    # an idea of t
    phases = numpy.unwrap(numpy.angle(data))
    m,b,r,p,err = scipy.stats.linregress(freqs*1e-9,phases)
    bound = 2*numpy.absolute(m)/(2*numpy.pi)
    result = scipy.optimize.minimize_scalar(_circle_residuals, 0, method='Bounded', args=(data,freqs), bounds=(0, bound))
    tau = result.x
    
    '''
    taus = numpy.linspace(0, 1000, 1000)
    res = [_circleResiduals(t, data, freqs) for t in taus]
    pyplot.autoscale(tight=True)
    ax = pyplot.subplot(1,1,1)
    ax.plot(taus,res)
    '''
    
    #print("Bound: " + str(bound))
    #print("Tau: " + str(tau))
    #print("Cost: " + str(result.fun))
        
    # Take the data, revert the cable delay, and translate the circle to the origin
    delay_corrected_data = numpy.multiply(data, numpy.exp(2 * numpy.pi * 1j * freqs * 1e-9 * tau))
    
    # Get the circle parameters from the optimization
    [r,xc,yc] = circle_fit(delay_corrected_data, freqs)
    translated_data = delay_corrected_data - xc - 1j*yc
    
    if(makePlots):
        axes[0,1].scatter(delay_corrected_data.real, delay_corrected_data.imag, s=0.5)
        axes[0,1].set_xlabel('Re[S21]')
        axes[0,1].set_ylabel('Im[S21]')
        axes[0,1].set_title('Phase-Corrected S21 Data')
        axes[0,1].add_patch(pyplot.Circle((xc,yc), r, fill=False,color='r'))
        axes[0,1].autoscale_view(tight=True)
    
    # Fit a phase-vs-frequency curve
    phases = numpy.unwrap(numpy.angle(translated_data))
    slope = 1 if numpy.mean(phases[0:10]) > numpy.mean(phases[-10:-1]) else -1  
    phase_model = lambda f,fr,Ql,theta0: theta0 + 2*slope*numpy.arctan(2*Ql*(1-(f/fr)))
    
    # Some parameter guesses
    # theta0 guess given in paper
    phi0 = 0 # -numpy.arcsin(yc/r)
    theta0_guess = numpy.mod(phi0 + numpy.pi, numpy.pi)
    
    # fr guess, find minimum transmission
    min_index = numpy.argmin(numpy.absolute(translated_data))
    fr_guess = freqs[min_index]
    
    # Q guess
    # Find average value of s21, (roughly) find where curve takes this value on either side of the minimum
    avg_s21 = numpy.mean(numpy.absolute(translated_data))
    avg_s21_index = numpy.argmin(numpy.absolute(numpy.absolute(translated_data) - avg_s21))
    Ql_guess = fr_guess / numpy.absolute(2.0*(freqs[avg_s21_index] - freqs[min_index]))
    
    # Perform the fit and extract some parameters
    [phase_result, phase_cov] = scipy.optimize.curve_fit(phase_model, freqs, phases, p0=[fr_guess, Ql_guess, theta0_guess], maxfev=2000)
    fr = phase_result[0]
    Ql = phase_result[1]
    theta0 = phase_result[2]
    
    if(makePlots):
        phase_fit = theta0 + 2*slope*numpy.arctan(2*Ql*(1-(freqs/fr)))
        axes[1,0].plot(freqs, phases)
        axes[1,0].plot(freqs, phase_fit)
        #axes[1,0].scatter(fit[3], fit[7])
        axes[1,0].set_xlabel('Freq (Hz)')
        axes[1,0].set_ylabel('Phase (rad)')
        axes[1,0].set_title('Phase vs. Frequency of Translated Circle')
    
    # Now that we've found the phase offset of the resonant point, find the point on the other side
    # of the circle - this is the off-resonant point
    Pr = (xc + 1j*yc) + r*numpy.exp(1j * theta0)
    Pprime = (xc + 1j*yc) + r*numpy.exp(1j * (theta0 + numpy.pi))
    
    if(makePlots):
        axes[0,1].scatter(Pr.real, Pr.imag, c='g')
        axes[0,1].scatter(Pprime.real, Pprime.imag, c='m')
        axes[0,1].autoscale_view(tight=True)
    
    # Find the remaining environmental parameters
    a = numpy.absolute(Pprime)
    alpha = numpy.angle(Pprime)
    
    # Correct for the environmental factors and redo the fit
    transformed_data = delay_corrected_data * numpy.exp(-1j * alpha) / a
    [r_corrected, x_corrected, y_corrected] = circle_fit(transformed_data, freqs)
    
    if(makePlots):
        transformed_Pr = Pr * numpy.exp(-1j * alpha) / a
        transformed_Pprime = Pprime * numpy.exp(-1j * alpha) / a
        axes[1,1].scatter(transformed_data.real, transformed_data.imag, s=0.5)
        axes[1,1].autoscale_view(tight=True)
        axes[1,1].set_xlabel('Re[S21]')
        axes[1,1].set_ylabel('Im[S21]')
        axes[1,1].set_title('Transformed Canonical S21 Data')
        axes[1,1].add_patch(pyplot.Circle((x_corrected,y_corrected), r_corrected, fill=False,color='r'))
        axes[1,1].scatter(transformed_Pr.real, transformed_Pr.imag, c='g')
        axes[1,1].scatter(transformed_Pprime.real, transformed_Pprime.imag, c='m')
        axes[1,1].autoscale_view(tight=True)
    
    # Find phi0
    phi0 = -numpy.arcsin(y_corrected/r_corrected)
    
    # Redo the phase fit to find the resonant frequency and loaded Q
    phases = numpy.unwrap(numpy.angle(transformed_data - x_corrected - (1j*y_corrected)))
    slope = 1 if numpy.mean(phases[0:10]) > numpy.mean(phases[-10:-1]) else -1  
    phase_model = lambda f,fr,Ql,theta0: theta0 + 2*slope*numpy.arctan(2*Ql*(1-(f/fr)))
    
    [corrected_phase_result, corrected_phase_cov] = scipy.optimize.curve_fit(phase_model, freqs, phases, p0=[fr,Ql,theta0],maxfev=2000)
    
    fr = corrected_phase_result[0]
    Ql = corrected_phase_result[1]
    theta0 = corrected_phase_result[2]
    
    # Compute the remaining resonator properties
    Qc = Ql/(2*r_corrected*numpy.exp(-1j * phi0)) if not manual_qc else manual_qc
    Qi = 1.0/((1.0/Ql) - numpy.real(1.0/Qc))
    
    # Get an error bar
    sigma_f = numpy.sqrt(corrected_phase_cov[0][0])
    sigma_Ql = numpy.sqrt(corrected_phase_cov[1][1])
    sigma_r = numpy.sqrt(
        (1/(len(transformed_data)-1))*
        numpy.sum(
            (r_corrected - numpy.sqrt((numpy.real(transformed_data)-x_corrected)**2 + (numpy.imag(transformed_data)-y_corrected)**2))**2
        )
    )
    
    # 1/Qc = (2*r_corrected*numpy.exp(-1j * phi0))/Ql
    # Neglecting error in phi0:
    sigma_ReOneOverQc = numpy.real(1/Qc)*numpy.sqrt((2*sigma_r/r_corrected)**2 + (sigma_Ql/Ql)**2) if not manual_qc else 0
    sigma_OneOverQl = sigma_Ql
    sigma_Qi = numpy.sqrt((sigma_ReOneOverQc*Qi**2)**2 + (sigma_OneOverQl*(Qi/Ql)**2)**2)

    errors = {"fr": sigma_f, "Ql": sigma_Ql, "R": sigma_r, "Qc": sigma_ReOneOverQc, "Qi": sigma_Qi}
    
    return [tau, a, alpha, fr, phi0, Ql, Qc, Qi, errors]