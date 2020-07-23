#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:26:57 2020

@author: pm
Contains functions to rotate a stream into different domains
"""
import numpy as np

from ..utils.createvmodel import load_avvmodel


def rotate_PSV(statlat, statlon, rayp, st, phase):
    """
    Finds the incidence angle of an incoming ray with the weighted average
    of the lithosphere's P velocity with a velocity model compiled from 
    Litho1.0.

    Parameters
    ----------
    statlat : FLOAT
        station latitude.
    statlon : FLOAT
        station longitude.
    rayp : FLOAT
        ray parameter / slownesss in s/m.
    st : obspy.Stream
        Input stream given in RTZ.
    phase : str
        Primary phase, either P or S.

    Returns
    -------
    avp : FLOAT
        Average P-wave velocity.
    avs : FLOAT
        Average S-wave velocity.
    PSvSh : obspy.Stream
        Stream in P-Sv-Sh.

    """

    model = load_avvmodel()
    avp, avs = model.query(statlat, statlon, phase)

    # Do the rotation as in Rondenay (2009)
    # Note that in contrast to the paper the z-components have to be
    # negated as it uses a definition, where depth is positive
    qa = np.sqrt(1/avp**2 - rayp**2)
    qb = np.sqrt(1/avs**2 - rayp**2)
    a = avs**2*rayp**2 - .5
    rotmat = np.array([[-a/(avp*qa), rayp*avs**2/avp, 0],
                       [-rayp*avs, -a/(avs*qb), 0],
                       [0, 0, 0.5]])

    # 1. Find which component is which one and put them in a dict
    stream = {
        st[0].stats.channel[2]: st[0].data,
        st[1].stats.channel[2]: st[1].data,
        st[2].stats.channel[2]: st[2].data}

    # deep copy
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


def rotate_LQT(st, phase):
    """
    Rotates a stream given in RTZ to LQT using Singular Value Decomposition

    Parameters
    ----------
    st : obspy.Stream
        Input Stream in RTZ.
    phase : STRING, optional
        "S" for Sp or "P" for Ps.

    Returns
    -------
    LQT : obspy.Stream
        Output Stream in LQT.
    QC : FLOAT
        Quality control criterion. Energy balance at first arrival for L and Q.
        Should deviate from 1 as far as possible.

    """
    if phase[-1] == 'P':
        onset = 30
    elif phase[-1] == 'S':
        onset = 120

    dt = st[0].stats.delta  # sampling interval

    # Deep copy stream & Normalise
    LQT = st.copy()
    LQT.normalize()
    del st

    # only check around phase-arrival
    # window
    if phase[-1] == "S":
        ws = round((onset-3)/dt)
        we = round((onset+5)/dt)
    elif phase[-1] == "P":
        ws = round((onset-5)/dt)
        we = round((onset+5)/dt)

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

    # Work with energies instead of amplitudes
    A_in = np.square(A_in)

    # Conduct svd
    u, s, vh = np.linalg.svd(A_in, full_matrices=False)

    # 2. Now, find out which is L and which Q by finding out which one has
    # the maximum energy around theoretical S-wave arrival - that one would
    # be Q. Or for Ps: maximum energy around P-wave arrival on L

    A_rot = np.linalg.inv(np.dot(u, np.diag(s)))

    # When working with energies A_rot is ambiguous for 180 deg, so I will
    # have to test here: element (2,1) is always negative, all others positive.
    if A_rot[0, 0] < 0:
        A_rot[0, :] = -A_rot[0, :]
    if A_rot[1, 0] > 0:
        A_rot[1, :] = -A_rot[1, :]

    LQ = np.dot(A_rot, ZR)
    # point of primary arrival
    pp1 = round((onset-2)/dt)
    pp2 = round((onset+10)/dt)
    npp = pp2 - pp1
    # point where converted Sp arrive
    if phase[-1] == "S":
        pc1 = round((onset-40)/dt)
        pc2 = round((onset-15)/dt)
    elif phase[-1] == "P":
        # Point where PS arrives
        pc1 = round((onset+20)/dt)
        pc2 = round((onset+40)/dt)

    npc = pc2 - pc1
    a = np.sum(np.square(LQ[0][pp1:pp2])/npp) /\
        np.sum(np.square(LQ[0][pc1:pc2])/npc)
    b = np.sum(np.square(LQ[1][pp1:pp2])/npp) /\
        np.sum(np.square(LQ[1][pc1:pc2])/npc)
    if a > b and phase[-1] == "S" or a < b and phase[-1] == "P":
        Q = LQ[0]
        L = LQ[1]
    elif a < b and phase[-1] == "S" or a > b and phase[-1] == "P":
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

    QC = a/b
    return LQT, QC


def rotate_LQT_min(st, phase):
    """
    Rotates stream to LQT by minimising the energy of the S-wave primary
    arrival on the L component (SRF) or maximising the primary arrival
    energy on L (PRF).

    Parameters
    ----------
    st : obspy.Stream
        Input stream in RTZ.
    phase : STRING, optional
        "P" for Ps or "S" for Sp.

    Returns
    -------
    LQT : obspy.Stream
        Output stream in LQT.
    ia : float
        Computed incidence angle in degree. Can serve as QC criterion.

    """
    phase = phase[-1]
    if phase == 'P':
        onset = 30
    elif phase == 'S':
        onset = 120

    dt = st[0].stats.delta

    # point of primary arrival
    pp1 = round((onset-2)/dt)
    pp2 = round((onset+10)/dt)
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
    ia = np.linspace(0, np.pi/2, num=90)  # incidence angle
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
    ia = ia*180/np.pi
    return LQT, ia
