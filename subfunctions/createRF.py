#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:10 2020

@author: pm

toolset to create RFs and RF stacks
"""

import numpy as np
from subfunctions import config
from subfunctions.deconvolve import it, damped
import subfunctions.config
from obspy import read
import shelve
import os
from obspy.taup import velocity_model, velocity_layer
from math import floor
from pkg_resources import resource_filename

_MODEL_CACHE = {}
DEG2KM = 111.2


def createRF(st, dt):
    RF = st.copy()
    # delete the old traces
    while RF.count() > 1:
        del RF[1]
    RF[0].stats.channel = config.phase + "RF"
    stream = {}
    for tr in st:
        stream[tr.stats.channel[2]] = tr.data
    # define denominator v and enumerator u
    if config.phase == "P" and config.rot == "RTZ":
        if "Z" in stream:
            v = stream["Z"]
        elif "3" in stream:
            v = stream["3"]
        u = stream["R"]
    elif config.phase == "P" and config.rot == "LQT":
        v = stream["L"]
        u = stream["Q"]
    elif config.phase == "P" and config.rot == "PSS":
        v = stream["P"]
        u = stream["V"]
    elif config.phase == "S" and config.rot == "RTZ":
        if "Z" in stream:
            u = stream["Z"]
        elif "3" in stream:
            u = stream["3"]
        v = stream["R"]
    elif config.phase == "S" and config.rot == "LQT":
        u = stream["L"]
        v = stream["Q"]
    elif config.phase == "S" and config.rot == "PSS":
        u = stream["P"]
        v = stream["V"]
    if config.decon_meth == "it":
        RF[0].data = it(v, u, dt, shift=config.tz)[0]
    elif config.decon_meth == "dampedf":
        RF[0].data = damped(v, u)
    return RF


def stackRF(network, station, phase=config.phase):
    # moveout correction
    RF_mo = Moveout(station, network, phasen=phase)
    # create trace for stack
    stack = RF_mo[0]
    stack[0].data = np.zeros(stack[0].stats.npts)
    for st in RF_mo:
        st.normalize()  # again, just to be safe
        stack[0].data = stack[0].data + st[0].data
    stack.normalize()
    return stack


def Moveout(station, network, phasen=config.phase):
    """Moveout corretion for pS and sP phases. A modified version of the
    original Fortran program written by Xiaohui Yuan
    INPUT:
        station
        network (make sure data is in RF folder)
        phase: P for Ps moveout, S for Sp moveout"""
    # change for compatibility with the RF module
    if phasen == "S":
        phase = "Sp"
    if phasen == "P":
        phase = "Ps"
    ioloc = config.RF[:-1] + phasen + '/' + network + '/' + station

    RF = []
    for file in os.listdir(ioloc):
        if file[:4] == "info":
            continue
        try:
            st = read(ioloc + '/' + file)
        except IsADirectoryError:
            continue
        st.normalize()  # make sure traces are normalised
        RF.append(st.copy())
    # dt = st[0].stats.delta
    # read info file
    with shelve.open(ioloc + '/info') as info:
        rayp = info["rayp"]
        onset = info["onset"]
    rayp = np.array(rayp)  # create numpy array
    # rayp_km = rayp*1000  # in s/km
    rayp = 111319.9*rayp  # from s/m to s/deg
    model = load_model()
    for ii, st in enumerate(RF):
        st = model.moveout(st, onset[ii], rayp[ii], phase=phase)
    return RF


""" Everything from here on is from the RF model by Tom Eulenfeld, modified
by me"""


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


    def calculate_vertical_slowness(self, slowness, phase='PS'):
        """
        Calculate vertical slowness of P and S wave.

        :param slowness: slowness in s/deg
        :param phase: Whether to calculate only P, only S or both vertical
            slownesses
        :return: vertical slowness of P wave, vertical slowness of S wave
            at different depths (z attribute of model instance)
        """
        phase = phase.upper()
        hslow = slowness / DEG2KM  # convert to horizontal slowness (s/km)
        qp, qs = 0, 0
        # catch warnings because of negative root
        # these values will be nan
        with np.errstate(invalid='ignore'):
            if 'P' in phase:
                qp = np.sqrt(self.vp ** (-2) - hslow ** 2)
            if 'S' in phase:
                qs = np.sqrt(self.vs ** (-2) - hslow ** 2)
        return qp, qs


    def calculate_delay_times(self, slowness, phase='PS'):
        """
        Calculate delay times between direct wave and converted phase.

        :param slowness: ray parameter in s/deg
        :param phase: Converted phase or multiple (e.g. Ps, Pppp)
        :return: delay times at different depths
        """
        phase = phase.upper()
        qp, qs = self.calculate_vertical_slowness(slowness, phase=phase)
        # Well that technically enables to calculate delay for multiples
        dt = (qp * phase.count('P') + qs * phase.count('S') -
              2 * (qp if phase[0] == 'P' else qs)) * self.dz
        return np.cumsum(dt)


    def stretch_delay_times(self, slowness, phase='Ps', ref=6.4):
        """
        Stretch delay times of provided slowness to reference slowness.

        First, calculate delay times (time between the direct wave and
        the converted phase or multiples at different depths) for the provided
        slowness and reference slowness.
        Secondly, stretch the the delay times of provided slowness to reference
        slowness.

        :param slowness: ray parameter in s/deg
        :param phase: 'Ps', 'Sp' or multiples
        :param ref: reference ray parameter in s/deg
        :return: original delay times, delay times stretched to reference
            slowness
        """
        if len(phase) % 2 == 1:
            msg = 'Length of phase (%s) should be divisible by two'
            raise ValueError(msg % phase)
        phase = phase.upper()
        try:
            t_ref = self.t_ref[phase]
        except KeyError:
            self.t_ref[phase] = t_ref = self.calculate_delay_times(ref, phase)
        t = self.calculate_delay_times(slowness, phase)
        if phase[0] == 'S':
            t_ref = -t_ref
            t = -t
        try:
            index = np.nonzero(np.isnan(t))[0][0] - 1
        except IndexError:
            index = len(t)
        t = t[:index]
        t_ref = t_ref[:index]
        return (np.hstack((-t[1:10][::-1], t)),
                np.hstack((-t_ref[1:10][::-1], t_ref)))


    def moveout(self, stream, onset, rayp, phase='Ps', ref=6.4):
        """
        In-place moveout correction to reference slowness.

        :param stream: stream with stats attributes onset and slowness.
        :param phase: 'Ps', 'Sp', 'Ppss' or other multiples
        :param ref: reference slowness (ray parameter) in s/deg
        onset: onset time of primary arrival
        rayp: in s/deg
        """
        for tr in stream:
            st = tr.stats
            # if not (st.starttime <= st.onset <= st.endtime):
            #     msg = 'onset time is not between starttime and endtime of data'
            #     raise ValueError(msg)
            index0 = int(floor((onset - st.starttime) * st.sampling_rate))
            t0, t1 = self.stretch_delay_times(rayp, phase=phase,
                                              ref=ref)
            S_multiple = phase[0].upper() == 'S' and len(phase) > 3
            if S_multiple:
                time0 = st.starttime - st.onset + index0 * st.delta
                old_data = tr.data[:index0][::-1]
                t = -time0 - np.arange(index0) * st.delta
                new_t = -np.interp(-t, -t0, -t1, left=0, right=0)
                data = np.interp(-t, -new_t, old_data, left=0., right=0.)
                tr.data[:index0] = data[::-1]
            if t0[-1] > t1[-1]:
                index0 += 1
            time0 = st.starttime - onset + index0 * st.delta
            old_data = tr.data[index0:]
            t = time0 + np.arange(len(tr) - index0) * st.delta
            # stretch old times to new times
            new_t = np.interp(t, t0, t1, left=0, right=None)
            # interpolate data at new times to data samples
            print(t)
            try:
                data = np.interp(t, new_t, old_data, left=None, right=0.)
                tr.data[index0:] = data
            except ValueError:
                continue
        return stream


    def ppoint_distance(self, depth, slowness, phase='S'):
        """
        Calculate horizontal distance between piercing point and station.

        :param depth: depth of interface in km
        :param slowness: ray parameter in s/deg
        :param phase: 'P' or 'S' for P wave or S wave. Multiples are possible.
        :return: horizontal distance in km
        """
        if len(phase) % 2 == 0:
            msg = 'Length of phase (%s) should be even'
            raise ValueError(msg % phase)
        phase = phase.upper()
        xp, xs = 0., 0.
        qp, qs = self.calculate_vertical_slowness(slowness, phase=phase)
        if 'P' in phase:
            xp = np.cumsum(self.dz * slowness / DEG2KM / qp)
        if 'S' in phase:
            xs = np.cumsum(self.dz * slowness / DEG2KM / qs)
        x = xp * phase.count('P') + xs * phase.count('S')
        z = self.z
        index = np.nonzero(depth < z)[0][0] - 1
        return x[index] + ((x[index + 1] - x[index]) *
                           (depth - z[index]) / (z[index + 1] - z[index]))


    # def ppoint(self, stats, depth, phase='S'):
    #     """
    #     Calculate latitude and longitude of piercing point.

    #     Piercing point coordinates and depth are saved in the pp_latitude,
    #     pp_longitude and pp_depth entries of the stats object or dictionary.

    #     :param stats: Stats object or dictionary with entries
    #         slowness, back_azimuth, station_latitude and station_longitude
    #     :param depth: depth of interface in km
    #     :param phase: 'P' for piercing point of P wave, 'S' for piercing
    #         point of S wave. Multiples are possible, too.
    #     :return: latitude and longitude of piercing point
    #     """
    #     dr = self.ppoint_distance(depth, stats['slowness'], phase=phase)
    #     lat = stats['station_latitude']
    #     lon = stats['station_longitude']
    #     az = stats['back_azimuth']
    #     plat, plon = direct_geodetic((lat, lon), az, dr)
    #     stats['pp_depth'] = depth
    #     stats['pp_latitude'] = plat
    #     stats['pp_longitude'] = plon
    #     return plat, plon
