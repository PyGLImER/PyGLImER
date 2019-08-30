#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:26:50 2019

@author: pm
"""
import numpy as np

# WATER LEVEL DECONVOLUTION    
def deconvolve(stream,delta):  # my first testrun gave me good results with delta = 1e-12, but how does the original code do that?
    # identify R and Z component
    st = {
        stream[0].stats.channel[2]: stream[0],
        stream[1].stats.channel[2]: stream[1],
        stream[2].stats.channel[2]: stream[2]}
    # df = stream[0].stats.sampling_rate 
    # dt = 1/df
    length = stream[0].stats.npts
   # n2 = np.ceil(2**(np.log(length)/np.log(2))) #round towards the next power of two
    
    
    # cast the R and Z component in the Fourier-domain - I might want to change the length here (longer) to improve computation time
    R_ft = np.fft.fft(st["R"])
    Z_ft = np.fft.fft(st["Z"])
    
    denom = np.multiply(Z_ft, np.conj(Z_ft))
    
    for j in range(length): # Check if there are amplitudes under the waterlevel threshold
        if denom[j] < delta:
            denom[j] = delta
            
    RF_ft = np.divide(np.multiply(R_ft, np.conj(Z_ft)), denom)
    
    RF = np.real(np.fft.ifft(RF_ft)) #[0:length]
    
    return RF
