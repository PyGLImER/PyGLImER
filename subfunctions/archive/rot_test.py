#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:59:37 2020

@author: pm
"""

import numpy as np
from obspy.core import *


# INPUT PARAMETERS
NEZ_file = "rot.tr"
RTZ_file = "rtz.tr"  # comes in order ZRT
PSS_file = "PSS.tr"
N = 600  # number of samples
dt = 0.05
avp = 6540
avs = 3710
rayp = 3e-5



def main(NEZ_file, RTZ_file, PSS_file, N):
    NEZ, RTZ, PSS = read_ray_in(NEZ_file, RTZ_file, PSS_file, N)
    PSS_mine, rotmat = rot(avp, avs, rayp, RTZ)
    return PSS_mine, PSS, RTZ, rotmat

def read_ray_in(NEZ_file, RTZ_file, PSS_file, N):
    """Read synthetic waveforms
    created with help of RAYSUM by Andrew Fredriksen."""
    NEZ_f = open(NEZ_file)
    RTZ_f = open(RTZ_file)
    PSS_f = open(PSS_file)
    files = [NEZ_f, RTZ_f, PSS_f]

    # The f five lines are header
    for f in files:
        for ii in range(5):
            f.readline()
        out = [[],[],[]]
        for line in range(N):
            x = f.readline()
            xl = x.split()
            for kk, value in enumerate(xl):
                for c in out:
                    out[kk].append(float(value))
        if f == NEZ_f:
            NEZ = np.array(out)
        elif f == RTZ_f:
            RTZ = np.array(out)
        elif f == PSS_f:
            PSS = np.array(out)
    return NEZ, RTZ, PSS

def rot(avp, avs, rayp, RTZ):
    qa = np.sqrt(1/avp**2 - rayp**2)
    qb = np.sqrt(1/avs**2 - rayp**2)
    Delta = 1 - 4*rayp**2*avs**2 + 4*rayp**4*avs**4 + 4*avs**4*rayp**2*qa*qb
    rotmat = np.array([[-2*avp*qa*(1 - 2*rayp**2*avs**2)/Delta,
                        4*avp*avs**2*rayp*qa*qb/Delta, 0],
                        [4*avs**3*rayp*qa*qb/Delta,
                        2*avs*qb*(1-2*rayp**2*avs**2)/Delta, 0],
                        [0, 0, .5]])
    
    # Do the rotation as in Rondenay (2009)
    # qa = np.sqrt(1/avp**2 - rayp**2)
    # qb = np.sqrt(1/avs**2 - rayp**2)
    # a = avs**2*rayp**2 - .5
    # rotmat = np.array([[a/(avp*qa), rayp*avs**2/avp, 0],
    #                     [rayp*avs, -a/(avs*qb), 0],
    #                     [0, 0, 0.5]])
    A_in = np.array([RTZ[2],
                RTZ[0],
                RTZ[1]])
    PSS = np.dot(rotmat, A_in)
    return PSS,rotmat

PSS_mine, PSS, RTZ, rotmat = main(NEZ_file, RTZ_file, PSS_file, N)

        