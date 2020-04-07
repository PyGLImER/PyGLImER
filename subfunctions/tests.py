#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:51:46 2020
Assorted functions that serve as a tests (mostly with synthetic data) of other
GLImER functions.
@author: pm
"""
import numpy as np
from subfunctions.moveout_stack import stackRF, moveout, load_model
from subfunctions.deconvolve import it, multitaper, spectraldivision
from obspy.core import Stream, Trace, Stats
from obspy import UTCDateTime, read
from obspy.signal.filter import lowpass
import matplotlib.pyplot as plt
import config
from subfunctions.preprocess import QC_S, QC_P
import os


tr_folder = "data/raysum_traces/"


def read_raysum(NEZ_file=None, RTZ_file=None, PSS_file=None):
    """Reads the output of the raysum program (by Andrew Frederiksen).
    INPUT:
        NEZ_f, RTZ_f, PSS_f: location, filename of the respectively rotated
            traces, can be None if the respective rotation does not exists.
        N: number of data points"""
    if NEZ_file:
        NEZ_f = open(tr_folder + NEZ_file)
    else:
        NEZ_f = None
    if RTZ_file:
        RTZ_f = open(tr_folder + RTZ_file)
    else:
        RTZ_f = None
    if PSS_file:
        PSS_f = open(tr_folder + PSS_file)
    else:
        PSS_f = None
    files = []
    for f in [NEZ_f, RTZ_f, PSS_f]:
        if f:
            files.append(f)
    if len(files) == 0:
        raise Exception("Choose at least one input file.")

    # The first five lines are header
    for f in files:
        f.readline()

        # read header information, M = number of traces, N = number of samples
        # per trace, dt = sampling interval, align = 1 if aligned to P-arrival,
        # shift = time shift
        x = f.readline()
        M, N, dt, align, shift = x.split()
        M = int(M)
        N = int(N)
        shift = float(shift)
        # Has to be aligned
        if not align:
            raise Exception("""The traces have to be aligned.""")
        if PSS_f:
            PSS = np.zeros([M, 3, N])
        if RTZ_f:
            RTZ = np.zeros([M, 3, N])
        if NEZ_f:
            NEZ = np.zeros([M, 3, N])

        # Loop over each stream
        for k in range(M):
            for i in range(3):  # header
                f.readline()
            out = [[], [], []]
            for l in range(N):
                x = f.readline()
                xl = x.split()
                for kk, value in enumerate(xl):
                    out[kk].append(float(value))
            if f == NEZ_f:
                NEZ[k, :, :] = np.array(out)
            elif f == RTZ_f:
                RTZ[k, :, :] = np.array(out)
            elif f == PSS_f:
                PSS[k, :, :] = np.array(out)

    # Program output
    dt = float(dt)
    if NEZ_f and RTZ_f and PSS_f:
        return NEZ, RTZ, PSS, dt, M, N, shift
    elif len(files) == 2:
        if NEZ_f:
            if RTZ_f:
                return NEZ, RTZ, dt, M, N, shift
            elif PSS_f:
                return NEZ, PSS, dt, M, N, shift
        elif PSS_f and RTZ_f:
            return RTZ, PSS, dt, M, N, shift
    elif len(files) == 1:
        if NEZ_f:
            return NEZ, dt, M, N, shift
        elif RTZ_f:
            return RTZ, dt, M, N, shift
        elif PSS_f:
            return PSS, dt, M, N, shift


def moveout_test(PSS_file, q, phase):
    """Creates synthetic PRFs and stacks them after moveout correction.
    INPUT: PSS_file: location + filename of PSS file
    q: slowness in s/m
    model: has to be in the glimer folder data/raysum.dat"""
    rayp = q*1.111949e5
    PSS, dt, M, N, shift = read_raysum(PSS_file=PSS_file)

    # Create receiver functions
    RF = []
    RF_mo = []
    stats = Stats()
    stats.npts = N
    stats.delta = dt
    stats.starttime = UTCDateTime(0)
    IRs = []
    for i in range(M):
        if phase == "P":
            data, _, IR = it(PSS[i, 0, :], PSS[i, 1, :], dt, shift=shift,
                             width=4)
        elif phase == "S":
            data, _, _ = it(PSS[i, 1, :], PSS[i, 0, :], dt, shift=shift,
                            width=4)
        RF.append(data)
        z, rfc = moveout(data, stats, UTCDateTime(shift), rayp[i], phase,
                         fname="raysum.dat")
        # z, IRc = moveout(IR, stats, UTCDateTime(0), rayp[i], phase,
        #                  fname="raysum.dat")
        # IRs.append(IRc)
        RF_mo.append(rfc)
    stack = np.average(RF_mo, axis=0)
    plt.close('all')
    plt.figure()
    for mo in RF_mo:
        plt.plot(z, mo)
    return z, stack, RF_mo, RF, dt, PSS  # ,   IRs


def decon_test(PSS_file, phase, method):
    PSS, dt, M, N, shift = read_raysum(PSS_file=PSS_file)

    # Create receiver functions
    RF = []
    for i in range(M):
        if phase == "P":
            u = PSS[i, 0, :]
            v = PSS[i, 1, :]
        elif phase == "S":
            u = PSS[i, 1, :]
            v = PSS[i, 0, :]
        if method == "it":
            data, _, _ = it(u, v, dt, shift=shift)
        elif method == "fqd" or method == "wat" or method == "con":
            data, _ = spectraldivision(v, u, dt, shift, regul=method)
        elif method == "multit_fqd":
            data, _, _, _ = multitaper(u, v, dt, shift, 'fqd')
            data = lowpass(data, 4.99, 1/dt, zerophase=True)
        elif method == "multit_con":
            data, _, _, _ = multitaper(u, v, dt, shift, 'con')
            data = lowpass(data, 4.99, 1/dt, zerophase=True)
        RF.append(data)
    return RF, dt


def test_SNR(network, station, phase=config.phase):
    """Test the automatic QC scripts for a certain station and writes ratings
    in the rating file."""
    noisematls = []
    critls = []
    loc = config.outputloc[:-1] + phase + '/by_station/' + \
        network + '/' + station
    for file in os.listdir(loc):
        try:
            st = read(loc + '/' + file)
        except IsADirectoryError:
            continue
        dt = st[0].stats.delta
        sampling_rate = st[0].stats.sampling_rate
        if phase == "S":
            _, crit, _, noisemat = QC_S(st, dt, sampling_rate)
        elif phase == "P":
            _, crit, _, noisemat = QC_P(st, dt, sampling_rate)
        noisematls.append(noisemat)
        critls.append(crit)
    return noisematls, critls
