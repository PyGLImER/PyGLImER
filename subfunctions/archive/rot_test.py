#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:59:37 2020

@author: pm
"""

import numpy as np
from obspy.core import *
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import matplotlib as mpl
from subfunctions.createRF import stackRF, moveout
from obspy.core import trace, stream
wdir = '/home/pm/Documents/Masters/GLimer to Obspy/'

# INPUT PARAMETERS
NEZ_file = "rot.tr"
RTZ_file = "rtz.tr"  # comes in order ZRT
PSS_file = "0.tr"
N = 3600  # number of samples
dt = 0.05
avp = 6540
avs = 3710
rayp = 3e-5



def main(NEZ_file, RTZ_file, PSS_file, N):
    NEZ, RTZ, PSS = read_ray_in(NEZ_file, RTZ_file, PSS_file, N)
    PSS_mine, rotmat = rot(avp, avs, rayp, RTZ)
    return PSS_mine, PSS, RTZ, rotmat, NEZ

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
                out[kk].append(float(value))
        if f == NEZ_f:
            NEZ = np.array(out)
        elif f == RTZ_f:
            RTZ = np.array(out)
        elif f == PSS_f:
            PSS = [np.array(out)]
            for i in range(36):
                for jj in range(3):
                    f.readline()
                out = [[],[],[]]
                for line in range(N):
                    x = f.readline()
                    xl = x.split()
                    for kk, value in enumerate(xl):
                        out[kk].append(float(value))
                PSS.append(np.array(out))
    return NEZ, RTZ, PSS

def rot(avp, avs, rayp, RTZ):
    # somehow the '99 rotation still doesn't work
    # qa = np.sqrt(1/avp**2 - rayp**2)
    # qb = np.sqrt(1/avs**2 - rayp**2)
    # Delta = 1 - 4*rayp**2*avs**2 + 4*rayp**4*avs**4 + 4*avs**4*rayp**2*qa*qb
    # rotmat = np.array([[-2*avp*qa*(1 - 2*rayp**2*avs**2)/-Delta,
    #                     4*avp*avs**2*rayp*qa*qb/Delta, 0],
    #                     [4*avs**3*rayp*qa*qb/-Delta,
    #                     2*avs*qb*(1-2*rayp**2*avs**2)/Delta, 0],
    #                     [0, 0, .5]])
    
    # Do the rotation as in Rondenay (2009)
    qa = np.sqrt(1/avp**2 - rayp**2)
    qb = np.sqrt(1/avs**2 - rayp**2)
    a = avs**2*rayp**2 - .5
    rotmat = np.array([[-a/(avp*qa), rayp*avs**2/avp, 0],
                        [-rayp*avs, -a/(avs*qb), 0],
                        [0, 0, 0.5]])
    A_in = np.array([RTZ[2],
                RTZ[0],
                RTZ[1]])
    PSS = np.dot(rotmat, A_in)
    return PSS,rotmat

# PSS_mine, PSS, RTZ, rotmat, NEZ = main(NEZ_file, RTZ_file, PSS_file, N)
NEZ, RTZ, PSS = read_ray_in(NEZ_file, RTZ_file, PSS_file, N)


plt.style.use('data/PaperDoubleFig.mplstyle')
# Make some style choices for plotting
colourWheel = ['#329932', '#ff6961', 'b', '#6a3d9a', '#fb9a99', '#e31a1c',
                '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99',
                '#b15928', '#67001f', '#b2182b', '#d6604d', '#f4a582',
                '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3',
                '#2166ac', '#053061']
dashesStyles = [[3, 1], [1000, 1], [2, 1, 10, 1], [4, 1, 1, 1, 1, 1]]


# moveout correction test
p = np.arange(0,3.6e-5,.1e-5)*1e3
traces = []
for ii, tr in enumerate(PSS):
    el = Trace(tr[2])
    el.stats["distance"] = p[ii-1]
    traces.append(el)
st = Stream(traces=traces)
# st.plot(type='section')
thick = np.array([40,100])
z = np.zeros(3)
z[1:] = np.cumsum(thick)
vp = np.array([6.54, 6.54, 7.7])
vs = np.array([3.71, 3.710, 4.200])
z_all =[]

htab = np.arange(0, 100, .1)  # hypothetical conversion dept
dt = np.zeros(np.shape(htab))  # delay times
# vertical slownesses
for p in p:
    q_a = np.sqrt(vp**-2 - p**2)
    q_b = np.sqrt(vs**-2 - p**2)
    # if phase.upper() == "P":
    for kk, h in enumerate(htab):
        ii = np.where(z >= h)[0][0]
        dz = np.diff(np.append(z[:ii], h))
        dt[kk] = np.sum((q_b[:len(dz)]-q_a[:len(dz)])*dz)
    tas = 0  # theoretical arrival sample
# Nq = st.npts - tas
# htab, dt = dt_table(rayp)
# queried times
    tq = np.arange(0, len(el.data))*5e-2
    zq = np.interp(tq, dt, htab)  # Depth array
    z_all.append(zq)
# if phase.upper() == "S":
#     data = np.flip(data)
#     data = -data
# RF = data[tas:tas+len(z)]
# zq = np.arange(0, max(z), 0.3)
# # interpolate RF
# RF = np.interp(zq, z, RF)
# z2 = np.arange(0, 800, .3)
# RF2 = np.zeros(z2.shape)
# RF2[:len(RF)] = RF

# y = NEZ[2]
# # create time vector
# plt.close('all')
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax.set_xlabel('time in s')

# ax.plot(y, color="k",linewidth=2)
# #x.set_xlim(-20, 50)
# #x.set_title(ch[ii])RTY
# #ax.yaxis.set_major_formatter(ScalarFormatter())
# #ax.yaxis.major.formatter._useMathText = True
# ax.xaxis.set_major_formatter(ScalarFormatter())
# ax.xaxis.major.formatter._useMathText = True
# #x.yaxis.set_minor_locator(AutoMinorLocator(5))
# ax.xaxis.set_minor_locator(AutoMinorLocator(5))
# # x.yaxis.set_label_coords(0.63, 1.01)
# #x.yaxis.tick_right()
# ax.yaxis.remove()
    #ax.grid(color='c', which='both', linewidth=.25)

        