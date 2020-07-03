#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:31:02 2019

@author: pm
"""

import numpy as np


def convf(u, v, nf, dt):
    """
    Convolution conducted in the frequency domain.

    Parameters
    ----------
    u : np.array
        Array 1.
    v : np.array
        Array 2.
    nf : INTEGER
        Array length in frequency domain (use next power of 2).
    dt : FLOAT
        Sampling Interval [s].

    Returns
    -------
    c : np.array
        Convolution of u with v.

    """

    U = np.fft.fft(u, n=nf)
    V = np.fft.fft(v, n=nf)
    C = U*V*dt
    c = np.real(np.fft.ifft(C, n=nf))
    return c


def corrf(u, v, nf):
    """
    Cross-correlation in frequency domain,
    calculates the x=crosscorr(u,v). Hence, v is the flipped vector.

    Parameters
    ----------
    u : np.array
        flipped vector.
    v : np.array
        Vector that the correlation is measured to.
    nf : INTEGER
        Array length in frequency domain (use next power of 2).

    Returns
    -------
    x : np.array
        Correlation vector.

    """

    V = np.conj(np.fft.fft(v, n=nf))
    U = np.fft.fft(u, n=nf)
    X = U*V
    x = np.real(np.fft.ifft(X, n=nf))
    return x


def gaussian(N, dt, width):
    """
    Create a zero-phase Gaussian function. In particular meant to be
    convolved with the impulse
    response that is the output of an iterative deconvolution

    Parameters
    ----------
    N : INTEGER
        Length of the desired array.
    dt : FLOAT
        Sampling interval [s].
    width : FLOAT
        Width parameter of the Gaussian function.

    Returns
    -------
    G : np.array
        Gaussian window.

    """

    df = 1/(N*dt)  # frequency step
    f = np.arange(0, round(0.5*N), 1, dtype=float)*df  # frequency array
    w = 2*np.pi*f  # angular frequency

    G = np.array([0]*N, dtype=float)
    G[0:round(N/2)] = np.exp(-w**2/(4*width**2))/dt
    G_lr = np.flip(G)
    G[round(N/2)+1:] = G_lr[-len(G[round(N/2)+1:]):]
    return G


def filter(s, F, dt, nf):
    """
    Convolves a filter with a signal (given in time domain).

    Parameters
    ----------
    s : np.array
        Signal given in time domain.
    F : np.array
        Filter's amplitude response.
    dt : FLOAT
        Sampling interval [s].
    nf : INTEGER
        Array length in frequency domain (use next power of 2).

    Returns
    -------
    s_f : np.array
        Filtered signal.

    """
    """ 
    INPUT
    s: signal in time domain
    F: Filter's amplitude response
    dt: sampling interval
    nf: length of the signal/filter in f-domain"""
    S = np.fft.fft(s, n=nf)
    S_f = np.multiply(S, F)*dt
    s_f = np.real(np.fft.ifft(S_f, n=nf))
    return s_f


def ricker(sigma, N2, dt):
    """ create a zero-phase Ricker / Mexican hat wavelet
INPUT:
    sigma - standard deviation
    dt - sampling Interval
    N2 - number of samples/2 (halflength). Full length = 2*N2+1"""
    rick_tt = np.arange(-N2, N2+1, 1, dtype=float)*dt  # time vector
    rick = 2/(np.sqrt(3*sigma)*np.pi**(1/4))*(1-np.power(rick_tt, 2)/sigma**2)\
        * np.exp(-rick_tt**2/(2*sigma**2))
    return rick_tt, rick


def noise(N, A):
    """ create random noise
    INPUT:
        N: signal length
        A: maximal amplitude"""
    noise = np.random.rand(N)*A
    return noise


def sshift(s, N2, dt, shift):
    """ shift a signal by a given time-shift in the frequency domain
    INPUT:
        s: signal
        N2: length of the signal in f-domain (usually equals the next pof 2)
        dt: sampling interval
        shift: the time shift in seconds"""
    S = np.fft.fft(s, n=N2)

    k = round(shift/dt)  # discrete shift
    # p = 2*np.pi*np.arange(0, N2, 1, dtype=float)*k/N2  # phase shift
    p = 2*np.pi*np.arange(1, N2+1, 1, dtype=float)*k/N2  # phase shift
    S = S*(np.cos(p) - 1j*np.sin(p))

    s_out = np.real(np.fft.ifft(S, N2))/np.cos(2*np.pi*k/N2)  # correct scaling
    return s_out
