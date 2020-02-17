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

def rotate_PSV(statlat,statlon,rayp,st):
    """Finds the incidence angle of an incoming ray with the weighted average
    of the lithosphere's P velocity
    INPUT:
        statlat: station latitude
        statlon: station longitude
        rayp: rayparameter in s/m
        st: stream in RTZ"""
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

    # build weighted average for upper 15km
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
    qa = np.sqrt(1/avp**2 - rayp**2)
    qb = np.sqrt(1/avs**2 - rayp**2)
    a = avs**2*rayp**2 - .5
    rotmat = np.array([[a/(avp*qa), rayp*avs**2/avp, 0],
                       [rayp*avs, -a/(avs*qb), 0],
                       [0, 0, 0.5]])
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
            tr.stats.channel = tr.stats.channel[0:2] + "Sv"
            tr.data = PSS[1]
        elif tr.stats.channel[2] == "Z" or tr.stats.channel[2] == "3":
            tr.stats.channel = tr.stats.channel[0:2] + "P"
            tr.data = PSS[0]
        elif tr.stats.channel[2] == "T":
            tr.stats.channel = tr.stats.channel[0:2] + "Sh"
            tr.data = PSS[2]
    return avp, avs, PSvSh


def rotate_LQT(st):
    """rotates a stream given in RTZ to LQT using Singular Value Decomposition
    INPUT:
        st: stream given in RTZ
    RETURNS:
        LQT: stream in LQT"""
    dt = st[0].stats.delta  # sampling interval
    LQT = st.copy()
    del st

    # 1. Find which component is which one and put them in a dict
    stream = {
        LQT[0].stats.channel[2]: LQT[0].data,
        LQT[1].stats.channel[2]: LQT[1].data,
        LQT[2].stats.channel[2]: LQT[2].data}
    # create input matrix, Z component is sometimes called 3
    if "Z" in stream:
        A_in = np.array([stream["Z"], stream["R"]])
    elif "3" in stream:
        A_in = np.array([stream["3"], stream["R"]])
    u, s, vh = np.linalg.svd(A_in, full_matrices=False)

    # 2. Now Find out which is L and which Q by finding out which one has
    # the maximum energy around theoretical S-wave arrival - that one would
    # be Q
    tas = config.tz/dt  # theoretical arrival sample
    # enery windows - 5 seconds before and after (is that enough?)
    ws = round(tas - 0.5/dt)
    we = round(tas + 15/dt)
    a = np.sum(np.square(vh[0, ws:we]))/np.sum(np.square(vh[0]))
    b = np.sum(np.square(vh[1, ws:we]))/np.sum(np.square(vh[0]))

    # maybe I should have a factor of 1.5 here? See how it works out
    # 13.02: ok, that doesn't work should think of a different method
    # It seems like the iasp91 arrival is not precise enough
    # Maybe it is also the data. The testdata was downloaded for different
    # epicentral distance, so perhaps I'm seeing different phases
    if a > b:
        Q = vh[0]
        L = vh[1]
    elif a < b:
        Q = vh[1]
        L = vh[0]

    # 3. save L and Q trace and change the label of the stream
    for tr in LQT:
        if tr.stats.channel[2] == "R":
            tr.stats.channel = tr.stats.channel[0:2] + "Q"
            tr.data = Q
        elif tr.stats.channel[2] == "Z" or tr.stats.channel[2] == "3":
            tr.stats.channel = tr.stats.channel[0:2] + "L"
            tr.data = L
    return LQT
