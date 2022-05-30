#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Apr  1 13:51:46 2020
Assorted functions that serve as a tests (mostly with synthetic data) of other
GLImER functions.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 1st April 2020 01:51:30 pm
Last Modified: Monday, 30th May 2022 03:22:50 pm
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import h5py

from obspy import UTCDateTime, read, Trace, Stream
from obspy.core import Stats
from obspy.signal.filter import lowpass
from geographiclib.geodesic import Geodesic

from pyglimer.data import finddir
from pyglimer.ccp import CCPStack
from pyglimer.rf import RFTrace, RFStream
from pyglimer.rf.create import createRF, read_by_station
from pyglimer.rf.deconvolve import it, multitaper, spectraldivision  # , gen_it
from pyglimer.rf.moveout import moveout, DEG2KM
from pyglimer.waveform.qc import qcs, qcp

tr_folder = os.path.join(finddir(), 'raysum_traces')


def read_raysum(phase, NEZ_file=None, RTZ_file=None, PSS_file=None):
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
        NEZ_f = open(os.path.join(tr_folder, phase, NEZ_file))
    else:
        NEZ_f = None
    if RTZ_file:
        RTZ_f = open(os.path.join(tr_folder, phase, RTZ_file))
    else:
        RTZ_f = None
    if PSS_file:
        PSS_f = open(os.path.join(tr_folder, phase, PSS_file))
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


def read_geom(geom_file, phase):
    """
    Reads in the geometry file of the raysum program to determine parameters.

    Parameters
    ----------
    geom_file : str
        Filename of the geometry file.

    Returns
    -------
    baz : np.ndarray(1D)
        Array containing the backazimuths of each trace [deg].
    q : np.ndarray(1D)
        Array containing the slownesses of each trace [s/m].
    dN : np.ndarray(1D)
        Array containing the north shift of each trace [m].
    dE : np.ndarray(1D)
        Array containing the east shift of each trace [m].

    """
    geom = open(os.path.join(tr_folder, phase, geom_file))
    x = geom.readlines()

    baz = []  # back-azimuths [deg]
    q = []  # slownesses [s/m]
    dN = []  # North shift [m]
    dE = []  # East shift [m]
    for line in x:
        # Ignore comments
        if line[0] == '#':
            continue
        # Create list
        ls = line.split()

        # append each element
        baz.append(ls[0])
        q.append(ls[1])
        dN.append(ls[2])
        dE.append(ls[3])

    # Convert to float
    baz = np.array(baz, dtype=float)
    q = np.array(q, dtype=float)
    dN = np.array(dN, dtype=float)
    dE = np.array(dE, dtype=float)

    return baz, q, dN, dE


def rf_test(
        phase, dip, rfloc='output/waveforms/RF', geom_file='3D.geom',
        decon_meth='it'):
    """
    Creates synthetic PRFs from Raysum data.

    Parameters
    ----------
    phase : string
        "P" or "S".
    dip : int
        Dip of the LAB in deg, determines, which files to use
    rfloc : The parental directory, in which the RFs are saved.
    geom_file : str, optional
        Filename of the geometry file

    Returns
    -------
    rfs: list
        List of RFTrace objects. Will in addition be saved in SAC format.

    """
    # Determine filenames
    PSS_file = []
    for i in range(16):
        PSS_file.append('3D_' + str(dip) + '_' + str(i) + '.tr')

    # Read geometry
    baz, q, dN, dE = read_geom(geom_file, phase)

    # statlat = dN/(DEG2KM*1000)
    d = np.sqrt(np.square(dN) + np.square(dE))
    az = np.rad2deg(np.arccos(dN/d))
    i = np.where(dE < 0)
    az[i] = az[i]+180
    statlat = []
    statlon = []
    for azimuth, delta in zip(az, d):
        if delta == 0:
            statlat.append(0)
            statlon.append(0)
            continue
        coords = Geodesic.WGS84.Direct(0, 0, azimuth, delta)
        statlat.append(coords["lat2"])
        statlon.append(coords["lon2"])
    #         for n, longitude in enumerate(lon):
#             y, _, _ = gps2dist_azimuth(latitude, 0, latitude, longitude)
    # statlon = dE/(DEG2KM*1000)
    rayp = q*DEG2KM*1000

    # Read traces
    stream = []

    for f in PSS_file:
        PSS, dt, _, N, shift = read_raysum(phase, PSS_file=f)
        stream.append(PSS)

    streams = np.vstack(stream)
    del stream

    M = len(baz)

    if M != streams.shape[0]:
        raise ValueError(
            ["Number of traces", streams.shape[0], """does not
             equal the number of backazimuths in the geom file""", M])

    rfs = []
    odir = os.path.join(rfloc, phase, 'raysum', str(dip))
    ch = ['BHP', 'BHV', 'BHH']  # Channel names

    os.makedirs(odir, exist_ok=True)

    # Create RF objects
    for i, st in enumerate(streams):
        s = Stream()
        for j, tr in enumerate(st):
            stats = Stats()
            stats.npts = N
            stats.delta = dt
            stats.st    # if old:
            stats.channel = ch[j]
            stats.network = 'RS'
            stats.station = str(dip)
            s.append(Trace(data=tr, header=stats))

        # Create info dictionary for rf creation
        info = {
            'onset': [UTCDateTime(0)+shift], 'starttime': [UTCDateTime(0)],
            'statlat': statlat[i], 'statlon': statlon[i],
            'statel': 0, 'rayp_s_deg': [rayp[i]], 'rbaz': [baz[i]],
            'rdelta': [np.nan], 'ot_ret': [0], 'magnitude': [np.nan],
            'evt_depth': [np.nan], 'evtlon': [np.nan], 'evtlat': [np.nan]}

        rf = createRF(s, phase=phase, method=decon_meth, info=info)

        # Write RF
        rf.write(os.path.join(odir, str(i)+'.sac'), format='SAC')
        rfs.append(rf)

    return rfs, statlat, statlon

# def multicore_RF(st, i, N, dt, ch, dip, shift, statlat, statlon, rayp, baz):
#     s = Stream()
#     for j, tr in enumerate(st):
#         stats = Stats()
#         stats.npts = N
#         stats.delta = dt
#         stats.starttime = UTCDateTime(0)
#         stats.channel = ch[j]
#         stats.network = 'RS'
#         stats.station = str(dip)
#         s.append(Trace(data=tr, header=stats))

#     # Create info dictionary for rf creation
#     info = {
#         'onset': [UTCDateTime(0)+shift], 'starttime': [UTCDateTime(0)],
#         'statlat': statlat[i], 'statlon': statlon[i],
#         'statel': 0, 'rayp_s_deg': [rayp[i]], 'rbaz': [baz[i]],
#         'rdelta': [np.nan], 'ot_ret': [0], 'magnitude': [np.nan],
#         'evt_depth': [np.nan], 'evtlon': [np.nan], 'evtlat': [np.nan]}

#     rf = createRF(s, phase='P', method='it', info=info)

#     # The rotation in Raysum does not work properly
#     # if abs(rf.data.max()) > abs(rf.data.min()):
#     #     rf.data = -rf.data

#     # Write RF
#     rf.write(os.path.join(odir, str(i)+'.sac'), format='SAC')
#     rfs.append(rf)


def ccp_test(
        phase, dip, geom_file='3D.geom', multiple=False, use_old_rfs=True):
    if not use_old_rfs:
        _, statlat, statlon = rf_test(phase, dip, geom_file=geom_file)
        print("RFs created")

    else:
        rfs = read_by_station('raysum', str(dip), phase, 'output/waveforms/RF')
        statlat = []
        statlon = []
        for rf in rfs:
            statlat.append(rf.stats.station_latitude)
            statlon.append(rf.stats.station_longitude)

    coords = np.unique(np.column_stack((statlat, statlon)), axis=0)

    # Create empty CCP object
    ccp = CCPStack(coords[:, 0], coords[:, 1], 0.1, phase='P')
    ccp.bingrid.phase = phase  # I'm doing it this way because computing a
    # bingrid for phase S takes way longer and I don't really need the extra
    # area
    print("CCP object created")

    # Create stack
    ccp.compute_stack(
        'raysum.dat', network='raysum', station=str(dip), save=False,
        multiple=multiple)
    print('ccp stack concluded')
    ccp.conclude_ccp(r=0, keep_water=True)
    ccp.write('raysum'+phase+str(dip))
    return ccp


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
    PSS, dt, M, N, shift = read_raysum(phase, PSS_file=PSS_file)

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
    PSS, dt, M, N, shift = read_raysum(phase, PSS_file=PSS_file)

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
            lrf = None
        # elif method == "gen_it":
        #     data, IR, iters, rej = gen_it(u, v, dt, phase=phase, shift=shift)
        #     lrf = None
        elif method == "fqd" or method == "wat" or method == "con":
            data, lrf = spectraldivision(
                v, u, dt, shift, phase=phase, regul=method, test=True)
        elif method == "multit_fqd":
            data, lrf, _, _ = multitaper(u, v, dt, shift, 'fqd')
            data = lowpass(data, 4.99, 1/dt, zerophase=True)
        elif method == "multit_con":
            data, lrf, _, data2 = multitaper(u, v, dt, shift, 'con')
            data = lowpass(data, 4.99, 1/dt, zerophase=True)
        else:
            raise NameError
        # if lrf is not None:
        #     # Normalisation for spectral division and multitaper
        #     # In order to do that, we have to find the factor that is
        #       necessary to
        #     # bring the zero-time pulse to 1
        #     fact = abs(lrf).max() #[round(shift/dt)]
        #     data = data/fact
        RF.append(RFTrace(data))
        RF[-1].stats.delta = dt
        RF[-1].stats.starttime = UTCDateTime(0)
        RF[-1].stats.onset = UTCDateTime(0) + shift
        RF[-1].stats.type = 'time'
        RF[-1].stats.phase = phase
        RF[-1].stats.channel = phase + 'RF'
        RF[-1].stats.network = 'RS'
    RF = RFStream(RF)
    return RF, dt


def test_SNR(
        network, station, phase, preproloc='ouput/waveforms/preprocessed'):
    """Test the automatic QC scripts for a certain station and writes ratings
    in the rating file."""
    noisematls = []
    critls = []
    loc = os.path.join(preproloc, phase, '/by_station/', network, station)
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
        RFTrace object, one per receiver function.
    i : int
        position in .mat.

    """
    inp = os.path.join(path, filename)
    out = os.path.join(outdir, outfile)

    mat = h5py.File(inp, 'r')

    # Create ouput dir
    os.makedirs(outdir, exist_ok=True)

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


# def test_rot(NEZ_file:str, phase:str, rot:str, baz:float, rayp:float):
#     NEZ, dt, M, N, shift = read_raysum(phase, NEZ_file=NEZ_file)
#     if M>1:
#         NEZ = NEZ[0]
#     stream = []
#     for tr in NEZ:
#         stats = Stats()
#         stats.npts = N
#         stats.delta = dt
#         stats.starttime = UTCDateTime(0)
#         stats.channel = ch[j]
#         stats.network = 'RS'
#         stats.station = 'rot'
#         stream.append(Trace(data=tr, header=stats))
#     stream = Stream(stream)
#     RTZ = stream.rotate
#     if rot=='PSS'

# def rot_PSS(avp, avs, rayp, RTZ):
#     # Do the rotation as in Rondenay (2009)
#     qa = np.sqrt(1 / avp ** 2 - rayp ** 2)
#     qb = np.sqrt(1 / avs ** 2 - rayp ** 2)
#     a = avs ** 2 * rayp ** 2 - .5
#     rotmat = np.array([[-a / (avp * qa), rayp * avs ** 2 / avp, 0],
#                        [-rayp * avs, -a / (avs * qb), 0],
#                        [0, 0, 0.5]])
#     A_in = np.array([RTZ[2],
#                      RTZ[0],
#                      RTZ[1]])
#     PSS = np.dot(rotmat, A_in)

#     return PSS, rotmat
