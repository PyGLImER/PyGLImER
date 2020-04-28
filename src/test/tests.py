#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:51:46 2020
Assorted functions that serve as a tests (mostly with synthetic data) of other
GLImER functions.
@author: pm
"""
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from joblib import Parallel, delayed

from obspy import UTCDateTime, read, Trace
from obspy.core import Stats
from obspy.signal.filter import lowpass

import config
from ..rf import RFTrace, RFStream
from ..rf.deconvolve import it, multitaper, spectraldivision
from ..rf.moveout import moveout, DEG2KM
from ..waveform.qc import qcs, qcp

tr_folder = "data/raysum_traces/"


def read_raysum(NEZ_file=None, RTZ_file=None, PSS_file=None):
    """
    Reads the output of the raysum program (by Andrew Frederiksen).

    Parameters
    ----------
    NEZ_file : str, optional
        Filename of file in NEZ coordinate system. The default is None.
    RTZ_file : str, optional
        Filename of file in RTZ coordinate system. The default is None.
    PSS_file : str, optional
        Filename of file in P-Sv-Sh coordinate system. The default is None.

    Raises
    ------
    Exception
        For wron inputs.

    Returns
    -------
    stream : np.array
        One numpy array for each provided input file.
    dt : float
        The sampling interval [s].
    M : int
        Number of traces per file.
    N : int
        Number of samples per trace
    """

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
            for j in range(N):
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
    """
    Creates synthetic PRFs and stacks them after depth migration.

    Parameters
    ----------
    PSS_file : str
        Filename of raysum file containing P-Sv-Sh traces.
    q : float
        Slowness [s/m].
    phase : str
        Either "P" for Ps or "S" for Sp.

    Returns
    -------
    z : np.array
        Depth vector.
    stack : np.array
        Receiver function stack.
    RF_mo : np.array
        Matrix containing all depth migrated RFs.
    RF : np.array
        Matrix containing all RFs.
    dt : float
        Sampling interval.
    PSS : np.array
        Matrix containing all traces in P-Sv-Sh.

    """
    rayp = q*1.111949e5
    PSS, dt, M, N, shift = read_raysum(PSS_file=PSS_file)

    # Create receiver functions
    RF = []
    RF_mo = []
    stats = Stats()
    stats.npts = N
    stats.delta = dt
    stats.starttime = UTCDateTime(0)

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
        RF_mo.append(rfc)
    stack = np.average(RF_mo, axis=0)
    plt.close('all')
    plt.figure()
    for mo in RF_mo:
        plt.plot(z, mo)
    return z, stack, RF_mo, RF, dt, PSS


def decon_test(PSS_file, phase, method):
    """
    Function to test a given deconvolution method with synthetic data created
    with raysum.

    Parameters
    ----------
    PSS_file : str
        Filename of raysum file containing P-Sv-Sh traces.
    phase : str
        "S" or "P".
    method : str
        Deconvolution method: use 1. "fqd", "wat", or "con for
        frequency-dependent damped, waterlevel damped, constantly damped
        spectraldivision. 2. "it" for iterative deconvolution, and 3.
        "multit_con" or "multitap_fqd" for constantly or frequency-dependent
        damped multitaper deconvolution.

    Returns
    -------
    RF : np.array
        Matrix containing all receiver functions.
    dt : float
        Sampling interval.

    """
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
            data, _, _, data2 = multitaper(u, v, dt, shift, 'con')
            data = lowpass(data, 4.99, 1/dt, zerophase=True)
        else:
            raise NameError
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
            _, crit, _, noisemat = qcs(st, dt, sampling_rate)
        elif phase == "P":
            _, crit, _, noisemat = qcp(st, dt, sampling_rate)
        noisematls.append(noisemat)
        critls.append(crit)
    return noisematls, critls


def read_rfs_mat(filename, path='output/ccps',
                 outdir='output/waveforms/RF/P/matlab', outfile='matlab'):
    """
    Reads in Matlab files constructed by the old GLImER workflow.
    It takes a long time to decode h5 files with that and they cannot
    be pickled either, so no multicore :-/

    Parameters
    ----------
    filename : str
        filename.
    path : str, optional
        Input directory. The default is 'output/ccps'.
    outdir : str, optional
        Output directory. The default is 'output/waveform/RF/matlab'.
    outfile : str, optional
        Output file. The default is 'matlab'.

    Yields
    ------
    rf : RFTrace
        RFTracec object, one per receiver function.
    i : int
        position in .mat.

    """
    inp = os.path.join(path, filename)
    out = os.path.join(outdir, outfile)

    mat = h5py.File(inp, 'r')

    # Create ouput dir
    if not Path(outdir).is_dir():
        subprocess.call(['mkdir', '-p', outdir])

    # Create RFTrace objects and save them for each of the RF

    for i, _ in enumerate(mat['cbaz'][0]):
        rf = RFTrace(data=mat['crfs'][:, i])
        rf.stats.delta = mat['dt'][0]
        rf.stats.station_latitude = mat['cslat'][0, i]
        rf.stats.station_longitude = mat['cslon'][0, i]
        rf.stats.station_elevation = mat['celev'][0, i]
        rf.stats.back_azimuth = mat['cbaz'][0, i]
        rf.stats.slowness = float(mat['crayp'][0, i])*DEG2KM
        rf.stats.type = 'time'
        rf.stats.phase = 'P'
        rf.stats.npts = len(mat['crfs'][:, i])
        rf.stats.starttime = UTCDateTime(0)
        rf.stats.onset = UTCDateTime(30)
        rf.write(out+str(i)+'.sac', 'SAC')
