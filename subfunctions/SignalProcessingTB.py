#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:31:02 2019

@author: pm
"""

import numpy as np
import subprocess
from subfunctions import config


def convf(u, v, nf, dt):
    """ convolution in frequency domain
    INPUT:
    u,v: arrays - to be convolved
    dt: sampling interval
    nf: array length in frequency-domain (usually next Power of 2)"""
    U = np.fft.fft(u, n=nf)
    V = np.fft.fft(v, n=nf)
    C = U*V*dt
    c = np.real(np.fft.ifft(C, n=nf))
    return c


def corrf(u, v, nf):
    """cross-correlation in frequency domain,
    calculates the x=crosscorr(u,v). Hence, v is the flipped vector.
    INPUT:
    v: flipped vector
    u: vector, to be correlated to
    nf: array length in frequency-domain (usually next Power of 2)"""
    V = np.conj(np.fft.fft(v, n=nf))
    U = np.fft.fft(u, n=nf)
    X = U*V
    x = np.real(np.fft.ifft(X, n=nf))
    return x


def gaussian(N, dt, width):
    """ Create a zero-phase Gaussian function. In particular meant to be
        convolved with the impulse
        response that is the output of an iterative deconvolution.
        INPUT:
        N = length of the desired array
        width = width parameter of the Gaussian function
        dt = sampling interval"""
    df = 1/(N*dt)  # frequency step
    f = np.arange(0, round(0.5*N), 1, dtype=float)*df  # frequency array
    w = 2*np.pi*f  # angular frequency

    G = np.array([0]*N, dtype=float)
    G[0:round(N/2)] = np.exp(-w**2/(4*width**2))/dt
    G_lr = np.flip(G)
    G[round(N/2)+1:] = G_lr[-len(G[round(N/2)+1:]):]
    return G


def filter(s, F, dt, nf):
    """ Convolve a filter with a signal (given in time domain)
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
        *np.exp(-rick_tt**2/(2*sigma**2))
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

    p = 2*np.pi*np.arange(1, N2+1, 1, dtype=float)*k/N2  # phase shift
    S = S*(np.cos(p) - 1j*np.sin(p))

    s_out = np.real(np.fft.ifft(S, N2))/np.cos(2*np.pi*k/N2)  # correct scaling

    return s_out


def rotate_PSV(statlat, statlon, rayp, st):
    """Finds the incidence angle of an incoming ray with the weighted average
    of the lithosphere's P velocity
    INPUT:
        statlat: station latitude
        statlon: station longitude
        rayp: rayparameter in s/m (i.e. horizontal slowness)
        st: stream in RTZ"""
    # Filter
    if config.phase == "S":
        st.filter('bandpass', freqmin=.03, freqmax=.33, zerophase=True)
    x = subprocess.Popen([config.lith1, "-p", str(statlat),
                          str(statlon)], stdout=subprocess.PIPE)
    ls = str(x.stdout.read()).split("\\n")  # save the output
    for ii, item in enumerate(ls):
        ls[ii] = item.split()
    # clean list
    del ls[-1]
    del ls[0][0]
    # reorder items
    depth = []
    vp = []
    vs = []
    name = []
    for item in ls:
        depth.append(float(item[0]))  # in m
        vp.append(float(item[2]))  # m/s
        vs.append(float(item[3]))  # m/
        name.append(item[-1])  # name of the boundary

    # build weighted average for upper 15km -10
    maxd = 15e3
    for ii, item in enumerate(depth):
        if item <= maxd:
            break
    avp = np.multiply(vp[ii+1:], -np.diff(depth)[ii:])
    avp = sum(avp) + vp[ii]*(-np.diff(depth)[ii-1] + maxd - depth[ii-1])
    avp = avp/maxd
    avs = np.multiply(vs[ii+1:], -np.diff(depth)[ii:])
    avs = sum(avs) + vs[ii]*(-np.diff(depth)[ii-1] + maxd - depth[ii-1])
    avs = avs/maxd

    # Do the rotation as in Rondenay (2009)
    # Note that in contrast to the paper the z-components have to be
    # negated as it uses a definition, where depth is positive
    qa = np.sqrt(1/avp**2 - rayp**2)
    qb = np.sqrt(1/avs**2 - rayp**2)
    a = avs**2*rayp**2 - .5
    rotmat = np.array([[-a/(avp*qa), rayp*avs**2/avp, 0],
                        [-rayp*avs, -a/(avs*qb), 0],
                        [0, 0, 0.5]])


    # Bostock (1999)
    # That doesn't work
    # qa = np.sqrt(1/avp**2 - rayp**2)
    # qb = np.sqrt(1/avs**2 - rayp**2)
    # Delta = 1 - 4*rayp**2*avs**2 + 4*rayp**4*avs**4 + 4*avs**4*rayp**2*qa*qb
    # rotmat = np.array([[-2*avp*qa*(1 - 2*rayp**2*avs**2)/Delta,
    #                     4*avp*avs**2*rayp*qa*qb/Delta, 0],
    #                    [4*avs**3*rayp*qa*qb/Delta,
    #                     2*avs*qb*(1-2*rayp**2*avs**2)/Delta, 0],
    #                    [0, 0, .5]])
    # 1. Find which component is which one and put them in a dict
    stream = {
        st[0].stats.channel[2]: st[0].data,
        st[1].stats.channel[2]: st[1].data,
        st[2].stats.channel[2]: st[2].data}
    PSvSh = st.copy()
    del st
    # create input matrix, Z component is sometimes called 3
    if "Z" in stream:
        A_in = np.array([stream["Z"],
                        stream["R"],
                        stream["T"]])
    elif "3" in stream:
        A_in = np.array([stream["3"],
                stream["R"],
                stream["T"]])
    PSS = np.dot(rotmat, A_in)

    # 3. save P, Sv and Sh trace and change the label of the stream
    for tr in PSvSh:
        if tr.stats.channel[2] == "R":
            tr.stats.channel = tr.stats.channel[0:2] + "V"
            tr.data = PSS[1]
        elif tr.stats.channel[2] == "Z" or tr.stats.channel[2] == "3":
            tr.stats.channel = tr.stats.channel[0:2] + "P"
            tr.data = PSS[0]
        elif tr.stats.channel[2] == "T":
            tr.stats.channel = tr.stats.channel[0:2] + "H"
            tr.data = PSS[2]
    PSvSh.normalize()
    return avp, avs, PSvSh


def rotate_LQT(st):
    """rotates a stream given in RTZ to LQT using Singular Value Decomposition
    INPUT:
        st: stream given in RTZ, normalized
    RETURNS:
        LQT: stream in LQT"""
    dt = st[0].stats.delta  # sampling interval
    LQT = st.copy()
    LQT.normalize()
    # Filter - is now down before
    # if config.phase == "S":
    #     LQT.filter('bandpass', freqmin=.03, freqmax=.33, zerophase=True)
    del st

    # only check around phase-arrival
    # window
    if config.phase == "S":
        ws = round((config.tz-3)/dt)
        we = round((config.tz+5)/dt)
    elif config.phase == "P":
        ws = round((config.tz-5)/dt)
        we = round((config.tz+5)/dt)
    # 1. Find which component is which one and put them in a dict
    stream = {
        LQT[0].stats.channel[2]: LQT[0].data,
        LQT[1].stats.channel[2]: LQT[1].data,
        LQT[2].stats.channel[2]: LQT[2].data}
    # create input matrix, Z component is sometimes called 3
    if "Z" in stream:
        A_in = np.array([stream["Z"][ws:we], stream["R"][ws:we]])
        ZR = np.array([stream["Z"], stream["R"]])
    elif "3" in stream:
        A_in = np.array([stream["3"][ws:we], stream["R"][ws:we]])
        ZR = np.array([stream["3"], stream["R"]])
    u, s, vh = np.linalg.svd(A_in, full_matrices=False)

    # 2. Now, find out which is L and which Q by finding out which one has
    # the maximum energy around theoretical S-wave arrival - that one would
    # be Q. Or for Ps: maximum energy around P-wave arrival on L
    # Note: the energy window has to be different otherwise the normalized
    # traces sometimes have the energy 1 over the whole time, should work like
    # this:
    # if config.phase == "S":
    #     ews = 1
    #     ewe = 5
    # elif config.phase == "P":
    #     ews = 1
    #     ewe = 8
    # a = np.sum(np.square(vh[0])  # , round(ews/dt):round(ewe/dt)]))
    # b = np.sum(np.square(vh[1]  # , round(ews/dt):round(ewe/dt)]))
    A_rot = np.linalg.inv(np.dot(u, np.diag(s)))
    LQ = np.dot(A_rot, ZR)
    # point of primary arrival
    pp1 = round((config.tz-2)/dt)
    pp2 = round((config.tz+20)/dt)
    npp = pp2 - pp1
    # point where converted Sp arrive
    pc1 = round((config.tz-40)/dt)
    pc2 = round((config.tz-15)/dt)
    npc = pc2 - pc1
    a = np.sum(np.square(LQ[0][pp1:pp2])/npp) /\
        np.sum(np.square(LQ[0][pc1:pc2])/npc)
    b = np.sum(np.square(LQ[1][pp1:pp2])/npp) /\
        np.sum(np.square(LQ[1][pc1:pc2])/npc)
    if a > b:# and config.phase == "S" or a < b and config.phase == "P":
        Q = LQ[0]
        L = LQ[1]
    elif a < b:# and config.phase == "S" or a > b and config.phase == "P":
        Q = LQ[1]
        L = LQ[0]

    # 3. save L and Q trace and change the label of the stream
    for tr in LQT:
        if tr.stats.channel[2] == "R":
            tr.stats.channel = tr.stats.channel[0:2] + "Q"
            tr.data = Q
        elif tr.stats.channel[2] == "Z" or tr.stats.channel[2] == "3":
            tr.stats.channel = tr.stats.channel[0:2] + "L"
            tr.data = L
    # LQT.normalize()
    return LQT, a/b


def rotate_LQT_min(st, phase=config.phase):
    """Rotates stream to LQT by minimising the energy of the S-wave primary
    arrival on the L component (SRF) or maximising the primary arrival
    energy on L (PRF)."""
    dt = st[0].stats.delta
    # point of primary arrival
    pp1 = round((config.tz-2)/dt)
    pp2 = round((config.tz+10)/dt)
    npp = pp2 - pp1
    LQT = st.copy()
    LQT.normalize()
    del st
    # identify components
    stream = {
        LQT[0].stats.channel[2]: LQT[0].data,
        LQT[1].stats.channel[2]: LQT[1].data,
        LQT[2].stats.channel[2]: LQT[2].data}
    if "Z" in stream:
        ZR = np.array([stream["Z"], stream["R"]])
    elif "3" in stream:
        ZR = np.array([stream["3"], stream["R"]])
    # calculate energy of the two components around zero
    # E_ZR = np.sum(np.square(ZR[:, pp1:pp2]), axis=1)/npp
    ia = np.linspace(0, np.pi/2, num=360)  # incidence angle
    E_L = []
    for ii in ia:
        A_rot = np.array([[np.cos(ii), np.sin(ii)],
                          [-np.sin(ii), np.cos(ii)]])
        LQ = np.dot(A_rot, ZR[:, pp1:pp2])
        # E_LQ = np.dot(A_rot, E_ZR)
        E_LQ = np.sum(np.square(LQ), axis=1)
        E_L.append(E_LQ[0]/E_LQ[1])
    if phase == "S":
        ii = E_L.index(min(E_L))
    elif phase == "P":
        ii = E_L.index(max(E_L))
    ia = ia[ii]
    A_rot = np.array([[np.cos(ia), np.sin(ia)],
                      [-np.sin(ia), np.cos(ia)]])
    LQ = np.dot(A_rot, ZR)
    # 3. save L and Q trace and change the label of the stream
    for tr in LQT:
        if tr.stats.channel[2] == "R":
            tr.stats.channel = tr.stats.channel[0:2] + "Q"
            tr.data = LQ[1]
        elif tr.stats.channel[2] == "Z" or tr.stats.channel[2] == "3":
            tr.stats.channel = tr.stats.channel[0:2] + "L"
            tr.data = LQ[0]
    return LQT, ia*180/np.pi
