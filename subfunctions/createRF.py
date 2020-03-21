#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:10 2020

@author: pm

toolset to create RFs and RF stacks
"""

import numpy as np
from subfunctions import config
from subfunctions.deconvolve import it, damped, waterlevel
import subfunctions.config
from obspy import read
import shelve
import os
from obspy.taup import velocity_model, velocity_layer
from math import floor
from pkg_resources import resource_filename
import obspy.core
from scipy import interpolate
from scipy.signal.windows import hann

_MODEL_CACHE = {}
DEG2KM = 111.2


def createRF(st, dt, phase=config.phase, shift=config.tz):
    RF = st.copy()
    # delete the old traces
    while RF.count() > 1:
        del RF[1]
    RF[0].stats.channel = phase + "RF"
    stream = {}
    for tr in st:
        stream[tr.stats.channel[2]] = tr.data
    # define denominator v and enumerator u
    if phase == "P" and "R" in stream:
        if "Z" in stream:
            v = stream["Z"]
        elif "3" in stream:
            v = stream["3"]
        u = stream["R"]
    elif phase == "P" and "Q" in stream:
        v = stream["L"]
        u = stream["Q"]
    elif phase == "P" and "V" in stream:
        v = stream["P"]
        u = stream["V"]
    elif phase == "S" and "R" in stream:
        if "Z" in stream:
            u = stream["Z"]
        elif "3" in stream:
            u = stream["3"]
        v = stream["R"]
    elif phase == "S" and "Q" in stream:
        u = stream["L"]
        v = stream["Q"]
    elif phase == "S" and "V" in stream:
        u = stream["P"]
        v = stream["V"]
    if config.decon_meth == "it":
        RF[0].data = it(v, u, dt, shift=shift, width=1.25)[0]
    elif config.decon_meth == "dampedf":
        RF[0].data = damped(v, u)
    elif config.decon_meth == "waterlevel":
        RF[0].data = waterlevel(v, u)
    RF.normalize()
    return RF


def stackRF(network, station, phase=config.phase):
    # extract data
    ioloc = config.RF[:-1] + phase + '/' + network + '/' + station
    data = []
    st = []  # stats
    RF_mo = []  # moveout corrected RF
    # z_all = []
    for file in os.listdir(ioloc):
        if file[:4] == "info":
            continue
        try:
            stream = read(ioloc + '/' + file)
        except IsADirectoryError:
            continue
        stream.normalize()  # make sure traces are normalised
        data.append(stream[0].data)
        st.append(stream[0].stats)
    with shelve.open(ioloc + '/info') as info:
        rayp = info['rayp_s_deg']
        onset = info["onset"]
        starttime = info["starttime"]
    for ii, tr in enumerate(data):
        jj = starttime.index(st[ii].starttime)
        z, RF = moveout(tr, st[ii], onset[jj], rayp[jj], phase)
        RF_mo.append(RF)
        # z_all.append(z)
    # Empty array for stack
    # zl = []
    # for z in z_all:
    #     zl.append(len(z))
    # ii = min(zl)
    stack = np.average(RF_mo, axis=0)
    return z, stack, RF_mo, data


def moveout(data, st, onset, rayp, phase):
    """Corrects the for the moveout of a converted phase. Flips time axis
    and polarity of SRF.
    INPUT:
        data: data from the trace object
        st: stats from the trace object
        onset: the onset time (UTCDATETIME)
        rayp: ray parameter in sec/deg
        phase: P or S for PRF or SRF, respectively"""
    tas = round((onset - st.starttime)/st.delta)  # theoretical arrival sample
    # Nq = st.npts - tas
    htab, dt = dt_table(rayp)
    # queried times
    tq = np.arange(0, round(max(dt), 1), st.delta)
    z = np.interp(tq, dt, htab)  # Depth array
    # tck = interpolate.splrep(dt, htab)
    # z = interpolate.splev(tq, tck)
    if phase.upper() == "S":
        data = np.flip(data)
        data = -data
    RF = data[tas:tas+len(z)]
    zq = np.arange(0, max(z), 0.3)
    # interpolate RF
    # RF = np.interp(zq, z, RF)
    tck = interpolate.splrep(z, RF)
    RF = interpolate.splev(zq, tck)
    # Taper the last 3.5 seconds of the RF to avoid discontinuities in stack
    tap = hann(round(7/st.delta))
    tap = tap[round(len(tap)/2):]
    taper = np.ones(len(RF))
    taper[-len(tap):] = tap
    RF = np.multiply(taper, RF)
    z2 = np.arange(0, 800, .3)
    RF2 = np.zeros(z2.shape)
    RF2[:len(RF)] = RF
    return z2, RF2


def dt_table(rayp):
    """Creates a phase delay table for a specific ray parameter given in
    s/deg. Using the equation as in Rondenay 2009."""
    p = rayp/DEG2KM  # convert to s/km
    maxz = 800  # maximum depth
    z, zf, vp, vs = iasp_flatearth(maxz)
    z = np.append(z, maxz)
    # dz = np.diff(z)  # thicknesses
    htab = np.arange(0, maxz, 1)  # hypothetical conversion depth, spherical
    htab_f = -6371*np.log((6371-htab)/6371)  # flat earth depth
    dt = np.zeros(np.shape(htab))  # delay times
    # vertical slownesses
    q_a = np.sqrt(vp**-2 - p**2)
    q_b = np.sqrt(vs**-2 - p**2)
    # if phase.upper() == "P":
    for kk, h in enumerate(htab_f):
        ii = np.where(zf >= h)[0][0]
        dz = np.diff(np.append(zf[:ii], h))
        dt[kk] = np.sum((q_b[:len(dz)]-q_a[:len(dz)])*dz)
    try:
        index = np.nonzero(np.isnan(dt))[0][0] - 1
    except IndexError:
        index = len(dt)
    return htab[:index], dt[:index]


def iasp91():
    """Loads velocity model in data subfolder"""
    # values = np.loadtxt('data/iasp91.dat', unpack=True)
    # z, vp, vs, n = values
    # n = n.astype(int)
    model = load_model()
    z = model.z
    vp = model.vp
    vs = model.vs
    # n = model.n
    return z, vp, vs  # , n


def iasp_flatearth(maxz):
    """Creates a flat-earth approximated velocity model down to 800 km as in
    Peter M. Shearer"""
    z, vp, vs = iasp91()
    r = 6371  # earth radius
    ii = np.where(z > maxz)[0][0]+1
    vpf = (r/(r-z[:ii]))*vp[:ii]
    vsf = (r/(r-z[:ii]))*vs[:ii]
    zf = -r*np.log((r-z[:ii])/r)
    z = z[:ii]
    # n = n[:ii]
    return z, zf, vpf, vsf  # , n


# def Moveout(station, network, phasen=config.phase):
#     """Moveout corretion for pS and sP phases. A modified version of the
#     original Fortran program written by Xiaohui Yuan
#     INPUT:
#         station
#         network (make sure data is in RF folder)
#         phase: P for Ps moveout, S for Sp moveout"""
#     # change for compatibility with the RF module
#     if phasen == "S":
#         phase = "Sp"
#     if phasen == "P":
#         phase = "Ps"
#     ioloc = config.RF[:-1] + phasen + '/' + network + '/' + station

#     RF = []
#     starttime = []
#     for file in os.listdir(ioloc):
#         if file[:4] == "info":
#             continue
#         try:
#             st = read(ioloc + '/' + file)
#         except IsADirectoryError:
#             continue
#         st.normalize()  # make sure traces are normalised
#         st[0].data = np.flip(st[0].data)  # double-check if that should be done
#         RF.append(st.copy())
#     # dt = st[0].stats.delta
#     # read info file
#     with shelve.open(ioloc + '/info') as info:
#         rayp = info['rayp_s_deg']
#         onset = info["onset"]
#         starttime = info["starttime"]
#     rayp = np.array(rayp)  # create numpy array
#     # rayp_km = rayp*1000  # in s/km
#     model = load_model()
#     for st in RF:
#         jj = starttime.index(st[0].stats.starttime)
#         st = model.moveout(st, onset[jj], rayp[jj], phase=phase)
#     return RF


# """ Everything from here on is from the RF model by Tom Eulenfeld, modified
# by me"""


def load_model(fname='iasp91.dat'):
    """
    Load model from file.

    :param fname: path to model file or 'iasp91'
    :return: `SimpleModel` instance

    The model file should have 4 columns with depth, vp, vs, n.
    The model file for iasp91 starts like this::

        #IASP91 velocity model
        #depth  vp    vs   n
          0.00  5.800 3.360 0
          0.00  5.800 3.360 0
        10.00  5.800 3.360 4
    """
    try:
        return _MODEL_CACHE[fname]
    except KeyError:
        pass
    values = np.loadtxt('data/' + fname, unpack=True)
    try:
        z, vp, vs, n = values
        n = n.astype(int)
    except ValueError:
        n = None
        z, vp, vs = values
    _MODEL_CACHE[fname] = model = SimpleModel(z, vp, vs, n)
    return model


def _interpolate_n(val, n):
    vals = [np.linspace(val[i], val[i + 1], n[i + 1] + 1, endpoint=False)
            for i in range(len(val) - 1)]
    return np.hstack(vals + [np.array([val[-1]])])


class SimpleModel(object):

    """
    Simple 1D velocity model for move out and piercing point calculation.

    :param z: depths in km
    :param vp: P wave velocities at provided depths in km/s
    :param vs: S wave velocities at provided depths in km/s
    :param n: number of support points between provided depths

    All arguments can be of type numpy.ndarray or list.
    taken from the RF module based on obspy by Tom Eulenfeld
    """

    def __init__(self, z, vp, vs, n=None):
        assert len(z) == len(vp) == len(vs)
        if n is not None:
            z = _interpolate_n(z, n)
            vp = _interpolate_n(vp, n)
            vs = _interpolate_n(vs, n)
        self.z = z[:-1]
        self.dz = np.diff(z)
        self.vp = vp[:-1]
        self.vs = vs[:-1]
        self.t_ref = {}


#     def calculate_vertical_slowness(self, slowness, phase='PS'):
#         """
#         Calculate vertical slowness of P and S wave.

#         :param slowness: slowness in s/deg
#         :param phase: Whether to calculate only P, only S or both vertical
#             slownesses
#         :return: vertical slowness of P wave, vertical slowness of S wave
#             at different depths (z attribute of model instance)
#         """
#         phase = phase.upper()
#         hslow = slowness / DEG2KM  # convert to horizontal slowness (s/km)
#         qp, qs = 0, 0
#         # catch warnings because of negative root
#         # these values will be nan
#         with np.errstate(invalid='ignore'):
#             if 'P' in phase:
#                 qp = np.sqrt(self.vp ** (-2) - hslow ** 2)
#             if 'S' in phase:
#                 qs = np.sqrt(self.vs ** (-2) - hslow ** 2)
#         return qp, qs


#     def calculate_delay_times(self, slowness, phase='PS'):
#         """
#         Calculate delay times between direct wave and converted phase.

#         :param slowness: ray parameter in s/deg
#         :param phase: Converted phase or multiple (e.g. Ps, Pppp)
#         :return: delay times at different depths
#         """
#         phase = phase.upper()
#         qp, qs = self.calculate_vertical_slowness(slowness, phase=phase)
#         # Well that technically enables to calculate delay for multiples
#         dt = (qp * phase.count('P') + qs * phase.count('S') -
#               2 * (qp if phase[0] == 'P' else qs)) * self.dz
#         return np.cumsum(dt)


#     def stretch_delay_times(self, slowness, phase='Ps', ref=6.4):
#         """
#         Stretch delay times of provided slowness to reference slowness.

#         First, calculate delay times (time between the direct wave and
#         the converted phase or multiples at different depths) for the provided
#         slowness and reference slowness.
#         Secondly, stretch the the delay times of provided slowness to reference
#         slowness.

#         :param slowness: ray parameter in s/deg
#         :param phase: 'Ps', 'Sp' or multiples
#         :param ref: reference ray parameter in s/deg
#         :return: original delay times, delay times stretched to reference
#             slowness
#         """
#         if len(phase) % 2 == 1:
#             msg = 'Length of phase (%s) should be divisible by two'
#             raise ValueError(msg % phase)
#         phase = phase.upper()
#         try:
#             t_ref = self.t_ref[phase]
#         except KeyError:
#             self.t_ref[phase] = t_ref = self.calculate_delay_times(ref, phase)
#         t = self.calculate_delay_times(slowness, phase)
#         if phase[0] == 'S':
#             t_ref = -t_ref
#             t = -t
#         try:
#             index = np.nonzero(np.isnan(t))[0][0] - 1
#         except IndexError:
#             index = len(t)
#         t = t[:index]
#         t_ref = t_ref[:index]
#         return (np.hstack((-t[1:10][::-1], t)),
#                 np.hstack((-t_ref[1:10][::-1], t_ref)))


#     def moveout(self, stream, onset, rayp, phase='Ps', ref=6.4):
#         """
#         In-place moveout correction to reference slowness.

#         :param stream: stream with stats attributes onset and slowness.
#         :param phase: 'Ps', 'Sp', 'Ppss' or other multiples
#         :param ref: reference slowness (ray parameter) in s/deg
#         onset: onset time of primary arrival
#         rayp: in s/deg
#         """
#         for tr in stream:
#             st = tr.stats
#             if not (st.starttime <= onset <= st.endtime):
#                 msg = 'onset time is not between starttime and endtime of data'
#                 raise ValueError(msg)
#             index0 = int(floor((onset - st.starttime) * st.sampling_rate))
#             t0, t1 = self.stretch_delay_times(rayp, phase=phase,
#                                               ref=ref)
#             S_multiple = phase[0].upper() == 'S' and len(phase) > 3
#             if S_multiple:
#                 time0 = st.starttime - st.onset + index0 * st.delta
#                 old_data = tr.data[:index0][::-1]
#                 t = -time0 - np.arange(index0) * st.delta
#                 new_t = -np.interp(-t, -t0, -t1, left=0, right=0)
#                 data = np.interp(-t, -new_t, old_data, left=0., right=0.)
#                 tr.data[:index0] = data[::-1]
#             if t0[-1] > t1[-1]:
#                 index0 += 1
#             time0 = st.starttime - onset + index0 * st.delta
#             old_data = tr.data[index0:]
#             t = time0 + np.arange(len(tr) - index0) * st.delta
#             # stretch old times to new times
#             new_t = np.interp(t, t0, t1, left=0, right=None)
#             # interpolate data at new times to data samples
#             try:
#                 data = np.interp(t, new_t, old_data, left=None, right=0.)
#                 tr.data[index0:] = data
#             except ValueError:
#                 continue
#         return stream


#     def ppoint_distance(self, depth, slowness, phase='S'):
#         """
#         Calculate horizontal distance between piercing point and station.

#         :param depth: depth of interface in km
#         :param slowness: ray parameter in s/deg
#         :param phase: 'P' or 'S' for P wave or S wave. Multiples are possible.
#         :return: horizontal distance in km
#         """
#         if len(phase) % 2 == 0:
#             msg = 'Length of phase (%s) should be even'
#             raise ValueError(msg % phase)
#         phase = phase.upper()
#         xp, xs = 0., 0.
#         qp, qs = self.calculate_vertical_slowness(slowness, phase=phase)
#         if 'P' in phase:
#             xp = np.cumsum(self.dz * slowness / DEG2KM / qp)
#         if 'S' in phase:
#             xs = np.cumsum(self.dz * slowness / DEG2KM / qs)
#         x = xp * phase.count('P') + xs * phase.count('S')
#         z = self.z
#         index = np.nonzero(depth < z)[0][0] - 1
#         return x[index] + ((x[index + 1] - x[index]) *
#                            (depth - z[index]) / (z[index + 1] - z[index]))


#     # def ppoint(self, stats, depth, phase='S'):
#     #     """
#     #     Calculate latitude and longitude of piercing point.

#     #     Piercing point coordinates and depth are saved in the pp_latitude,
#     #     pp_longitude and pp_depth entries of the stats object or dictionary.

#     #     :param stats: Stats object or dictionary with entries
#     #         slowness, back_azimuth, station_latitude and station_longitude
#     #     :param depth: depth of interface in km
#     #     :param phase: 'P' for piercing point of P wave, 'S' for piercing
#     #         point of S wave. Multiples are possible, too.
#     #     :return: latitude and longitude of piercing point
#     #     """
#     #     dr = self.ppoint_distance(depth, stats['slowness'], phase=phase)
#     #     lat = stats['station_latitude']
#     #     lon = stats['station_longitude']
#     #     az = stats['back_azimuth']
#     #     plat, plon = direct_geodetic((lat, lon), az, dr)
#     #     stats['pp_depth'] = depth
#     #     stats['pp_latitude'] = plat
#     #     stats['pp_longitude'] = plon
#     #     return plat, plon
