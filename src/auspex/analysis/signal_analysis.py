# Copyright 2019 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from numpy.fft import fft
from scipy.linalg import svd, eig, inv, pinv


def hilbert(signal):
    """Construct the Hilbert transform of the signal via the Fast Fourier Transform.
        In essense, we just want to set negative frequency components to zero
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

def KT_estimation(data, times, order):
    """KT estimation of periodic signal components."""

    # Find the hilbert transform
    analytic_signal = hilbert(data)
    time_step = times[1]-times[0]

    # Create the Hankel matrix
    N = len(analytic_signal)
    K = order
    M = (N//2)-1
    L = N-M+1
    H = np.zeros((L, M), dtype=np.complex128)
    for ct in range(M):
        H[:,ct] = analytic_signal[ct:ct+L]

    #Try and seperate the signal and noise subspace via the svd
    U,S,V = svd(H, False) # V is not transposed/conjugated in numpy svd

    #Reconstruct the approximate Hankel matrix with the first K singular values
    #Here we can iterate and modify the singular values
    S_k = np.diag(S[:K])

    #Estimate the variance from the rest of the singular values
    varEst = (1/((M-K)*L)) * np.sum(S[K:]**2)
    Sfilt = np.matmul(S_k**2 - L*varEst*np.eye(K), inv(S_k))
    Hbar = np.matmul(np.matmul(U[:,:K], Sfilt), V[:K,:])

    #Reconstruct the data from the averaged anti-diagonals
    cleanedData = np.zeros(N, dtype=np.complex128)
    tmpMat = np.flip(Hbar,1)
    idx = -L+1
    for ct in range(N-1,-1,-1):
        cleanedData[ct] = np.mean(np.diag(tmpMat,idx))
        idx += 1

    #Create a cleaned Hankel matrix
    cleanedH = np.empty_like(H)
    cleanedAnalyticSig = hilbert(cleanedData)
    for ct in range(M):
        cleanedH[:,ct] = cleanedAnalyticSig[ct:ct+L]

    #Compute Q with total least squares
    #U_K1*Q = U_K2
    U = svd(cleanedH, False)[0]
    U_K = U[:,0:K]
    tmpMat = np.hstack((U_K[:-1,:],U_K[1:,:]))
    V = svd(tmpMat, False)[2].T.conj()
    n = np.size(U_K,1)
    V_AB = V[:n,n:]
    V_BB = V[n:,n:]
    Q = np.linalg.lstsq(V_BB.conj().T, -V_AB.conj().T, rcond=None)[0].conj().T

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

    amps = np.linalg.lstsq(A, cleanedData, rcond=None)[0]

    return freqs, Tcs, amps