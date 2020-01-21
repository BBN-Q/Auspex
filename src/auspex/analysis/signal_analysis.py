# Copyright 2019 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from auspex.log import logger
from numpy.fft import fft
from scipy.linalg import svd, eig, inv, pinv

"""
Produces initial guess for damped exponential using SVD method described in:
Van Huffel, S. (1993). Enhanced resolution based on minimum variance estimation and exponential data modeling Signal Processing, 33(3), 333-355. doi:10.1016/0165-1684(93)90130-3
"""

def hilbert(signal):
    """ Construct the Hilbert transform of the signal via the Fast Fourier Transform.
        This sets the negative frequency components to zero and multiplies positive frequencies by 2.
        This is neccessary since the fitted model refers only to positive frequency components e^(jwt)
    """
    spectrum = np.fft.fft(signal)
    n = len(signal)
    midpoint = int(np.ceil(n/2))
    kernel = np.zeros(n)
    kernel[0] = 1
    if n%2 == 0:
        kernel[midpoint] = 1
    kernel[1:midpoint] = 2
    return np.fft.ifft(kernel * spectrum)

def hankel(signal,M):

    # Create the Hankel matrix
    N = len(signal)
    L = N-M+1
    H = np.zeros((L, M), dtype=np.complex128)
    for ct in range(M):
        H[:,ct] = signal[ct:ct+L]

    return H

def cleandata(signal,M,K):
    """ Clean data iteratively using minimum variance (MV) estimation described in:
        Van Huffel, S. (1993). Enhanced resolution based on minimum variance estimation and exponential data modeling Signal Processing, 33(3), 333-355. doi:10.1016/0165-1684(93)90130-3
    """

    L = len(signal)-M+1
    H = hankel(signal,M)

    #Try and seperate the signal and noise subspace via the svd
    #Note: Numpy returns matrix = UxSxV, not matrix = UxSXV*
    U,S,V = svd(H, False)

    #Reconstruct the approximate Hankel matrix with the first K singular values
    #Here we can iterate and modify the singular values
    S_k = np.diag(S[:K])

    #Estimate the variance from the rest of the singular values
    varEst = (1/((M-K)*L)) * np.sum(S[K:]**2)
    Sfilt = np.matmul(S_k**2 - L*varEst*np.eye(K), inv(S_k))
    HK = np.matmul(np.matmul(U[:,:K], Sfilt), V[:K,:])

    #Reconstruct the data from the averaged anti-diagonals
    cleanedData = np.zeros(len(signal), dtype=np.complex128)
    tmpMat = np.flip(HK,1)
    idx = -L+1
    for ct in range(len(signal)-1,-1,-1):
        cleanedData[ct] = np.mean(np.diag(tmpMat,idx))
        idx += 1

    #Iterate until variance of noise is less than signal eigenvalues, which should be the case for high SNR data
    if L*varEst > min(np.diagonal(S_k))**2:
        logger.warning("Noise variance greater than signal amplitudes. Consider taking more averages. Iterating cleanup ...")
        cleanedData = cleandata(hilbert(cleanedData),M,K)

    return cleanedData



def KT_estimation(data, times, order):
    """KT estimation of periodic signal components."""

    K = order
    N = len(data)
    time_step = times[1]-times[0]

    #clean data with M = K+1 so that L>>M is guaranteed as per MV estimation
    cleanedData = cleandata(hilbert(data),order+1,order)

    #Create a cleaned Hankel matrix, here the matrix is constructed so that L~M
    cleanedAnalyticSig = hilbert(cleanedData)
    cleanedH = hankel(cleanedAnalyticSig,(N//2) - 1)

    #Compute Q with total least squares (TLS)
    #Bjõrck, Ake (1996) Numerical Methods for Least Squares Problems

    #UK1*Q = UK2
    U = svd(cleanedH, False)[0]
    UK = U[:,0:K]

    tmpMat = np.hstack((UK[:-1,:],UK[1:,:]))
    V = svd(tmpMat, False)[2].T.conj()
    n = np.size(UK,1)
    V_AB = V[:n,n:]
    V_BB = V[n:,n:]
    Q = -1*np.matmul(V_AB,inv(V_BB))

    #Now poles are eigenvalues of Q
    poles, _ = eig(Q)

    #Take the log and return the decay constant and frequency
    freqs = np.zeros(K)
    Tcs   = np.zeros(K)
    for ct in range(K):
        sk = np.log(poles[ct])
        freqs[ct] = np.imag(sk)/(2*np.pi*time_step)
        Tcs[ct] = -1.0/np.real(sk)*time_step

    #Refit the data to get the amplitude
    A = np.zeros((N, K), dtype=np.complex128)
    for ct in range(K):
        A[:,ct] = np.power(poles[ct], range(0,N))

    amps = np.linalg.lstsq(A, cleanedData, rcond=-1)[0]

    return freqs, Tcs, amps
