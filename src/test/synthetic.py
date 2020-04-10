#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:37:49 2019
Create synthetic impulse resonses.
Used to test the deconvolve module.
@author: pm
"""
import numpy as np

from src.utils import signalproc as sptb


def synthetic(N2,dt,R,SNR,stdv):
    """
Function to create synthetic seismograms (given Impulse response convolved with a Ricker wavelet)
INPUT:
    N2: half-length of the Input traces, (full length N= N2*2+1)
    dt: sampling rate
    R: Impulse response / reflectivity series with length N
    SNR: ratio of maximal amplitude of the source wavelet to max noise amplitude
    stdv : standard deviation of the source pulse (controls width)
Output:
    v_noise: the theorethical source wavelet with noise independent from the noise on the horizontal compoennt
            (equal to the hoizontal component of a seismogram in RF)
    h_noise: The horizontal component of the seismogram with noise
"""
    
    # Full-length
    N = N2*2+1
    if len(R) != N:
        print("Error: The given impulse response has the wrong length. Give an array with length=",N)
        quit()

    
    # create source wavelet
    [time,s] = sptb.ricker(stdv,N2,dt)
    
    # create horizomtal component
    h = np.convolve(R,s)[0:N]

    
    
    # create noise
    if SNR:
        np.random.seed(7) #create consistently the same random numbers
        A_noise = max(s)/SNR
        noise = sptb.noise(N,A_noise)
    else:
        noise = 0
    
    # add noise to source-wavelet
    v_noise = s+noise
    
    # add independent noise to horizontal component
    if SNR:
        noise = sptb.noise(N,A_noise)
    h_noise = h+noise

    return v_noise, h_noise, time


def create_R(N2, M):
    """Program to create random reflectivity series
  INPUT
  N2: half-length of the signal (N=2*N2+1)
  M: number of seismic discontinuities"""
    N = N2*2+1

    # create reflectivity series
    R = np.zeros(N)

    np.random.seed(42)  # create consistently the same random array
    for ii in range(M-1):
        R[np.random.randint(0, N2)] = 2*np.random.rand()-1
    return R




















