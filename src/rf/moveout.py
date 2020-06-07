#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:08:42 2020

Contains functions for moveout correction and station stacking

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""

import os
import shelve
import warnings

import numpy as np
from obspy import read
from geographiclib.geodesic import Geodesic
from scipy import interpolate
from scipy.signal.windows import hann

import config
from ..constants import R_EARTH, DEG2KM, maxz, res
from ..utils.createvmodel import load_gyps


_MODEL_CACHE = {}


def moveout(data, st, fname, latb, lonb, taper):
    """
    Depth migration for RF.
    Corrects the for the moveout of a converted phase. Flips time axis
    and polarity of SRF.

    Parameters
    ----------
    data : np.array
        Receiver Function.
    st : obspy.core.AttributeDict
        Stats from stream object.
    fname : string
        1D velocity model for moveout correction. Use '3D' for a 3D raytracing.
    latb : Tuple
        Tuple in Form (minlat, maxlat). To save RAM on 3D raytraycing.
        Will remain unused for 1D RT.
    lonb : Tuple
        Tuple in Form (minlon, maxlon)
    taper : Bool
        If True, the last 10km of the RF will be tapered, which avoids
        jumps in station stacks. If False,
        the upper 5 km will be tapered.
        Should be False for CCP stacks.

    Returns
    -------
    z2 : np.array
        Depth vector in km.
    RF2 : np.array
        Vector containing the depth migrated RF.
    delta : np.array
        Vector containing Euclidian distance of piercing point from the
        station at depth z.

    """

    onset = st.onset
    tas = round((onset - st.starttime)/st.delta)  # theoretical arrival sample
    rayp = st.slowness  # Ray parameter
    phase = st.phase  # Primary phase
    el = st.station_elevation

    if fname[-2:] == '3D':
        test = fname == 'raysum3D'
        if test:
            test = int(st.station)  # The dip of the LAB
        htab, dt, delta = dt_table_3D(
            rayp, phase, st.station_latitude, st.station_longitude,
            st.back_azimuth, el, latb, lonb, test=test)
    else:
        htab, dt, delta = dt_table(rayp, fname, phase, el)

    # queried times
    tq = np.arange(0, round(max(dt), 1), st.delta)
    z = np.interp(tq, dt, htab)  # Depth array

    # Flip SRF
    if phase.upper() == "S":
        data = np.flip(data)
        data = -data
        tas = -tas

    # Shorten RF
    RF = data[tas:tas+len(z)]
    zq = np.hstack((np.arange(min(z), 0, .1), np.arange(0, max(z)+res, res)))

    # interpolate RF
    try:
        tck = interpolate.splrep(z, RF)

    except TypeError as e:
        # Happens almost never, the RF is empty? No data or some bug in the data
        # correction and RF is too short, just return everything 0
        mes = "The length of the Receiver Function is" + str(len(z)) + "and\
        therefore too short, setting = 0."
        warnings.warn(mes, category=UserWarning, stacklevel=1)
        z2 = np.hstack((np.arange(-10, 0, .1), np.arange(0, maxz+res, res)))
        RF2 = np.zeros(z2.shape)
        delta2 = np.empty(z2.shape)
        delta2.fill(np.nan)
        return z2, RF2, delta2

    RF = interpolate.splev(zq, tck)
    if taper:
        # Taper the last 10 km
        tap = hann(20)
        _, down = np.split(tap, 2)

        if len(RF) > len(down):  # That usually doesn't happen, only for extreme
            # discontinuities in 3D model and errors in SRF data
            taper = np.ones(len(RF))
            taper[-len(down):] = down
            RF = np.multiply(taper, RF)
    
    else:
        # Taper the upper 5 km
        i = np.where(z>=5)[0][0]  # Find where rf is depth = 5
        # tap = hann((i+1)*2)
        tap = hann(8)
        up, _ = np.hstack(np.zeros(i-3), np.split(tap, 2))
        if len(RF) > len(up):  # That usually doesn't happen, only for extreme
            # discontinuities in 3D model and errors in SRF data
            taper = np.ones(len(RF))
            taper[:len(up)] = up
            RF = np.multiply(taper, RF)

    z2 = np.hstack((np.arange(-10, 0, .1), np.arange(0, maxz+res, res)))
    RF2 = np.zeros(z2.shape)

    # np.where does not seem to work here
    starti = np.nonzero(np.isclose(z2, htab[0]))[0][0]
    if len(RF)+starti > len(RF2):
        # truncate
        # Honestly do not know why that what happen, but it does once in a
        # million times, perhaps rounding + interpolation.
        mes = "The interpolated RF is too long, truncating."
        warnings.warn(mes, category=UserWarning, stacklevel=1)
        diff = len(RF) + starti - len(RF2)
        RF = RF[:-diff]
    RF2[starti:starti+len(RF)] = RF

    # reshape delta - else that will mess with the CCP stacking
    delta2 = np.empty(z2.shape)
    delta2.fill(np.nan)
    delta2[starti:starti+len(delta)] = delta

    return z2, RF2, delta2


def dt_table_3D(rayp, phase, lat, lon, baz, el, latb, lonb, test=False):
    """
    Creates a phase delay table and calculates piercing points
    for a specific ray parameter,
    using the equation as in Rondenay 2009.
    For SRF: The length of the output vectors is determined by the depth, at
    which Sp conversion is supercritical (vertical slowness = complex).

    Parameters
    ----------
    rayp : float
        ray-parameter given in s/deg.
    fname : string
        1D velocity model, located in data/vmodels.
    phase : string
        Either "S" for Sp or "P" for Ps.
    el : float
        station elevation in m.
    latb : Tuple
        Tuple in Form (minlat, maxlat). To save RAM on 3D raytraycing.
        Will remain unused for 1D RT.
    lonb : Tuple
        Tuple in Form (minlon, maxlon)
    test : Bool
        If True, the raysum 3D model is loaded.

    Returns
    -------
    htab: np.array
        Vector containing conversion depths.
    dt: np.array
        Vector containing delay times between primary arrival and converted
        wave.
    delta: np.array
        Vector containing Euclidian distance of piercing point from the
        station at depth z.

    """

    p = rayp/DEG2KM  # convert to s/km

    # if test:
    #     model = raysum3D(test)
    # else:
    model = load_gyps(latb=latb, lonb=lonb)

    # hypothetical conversion depth tables
    if el > 0:
        htab = np.hstack(  # spherical
                         (np.arange(-round(el/1000, 1), 0, .1),
                          np.arange(0, maxz+res, res)))
    else:
        htab = np.arange(-round(el/1000), maxz+res, res)

    htab_f = -R_EARTH*np.log((R_EARTH-htab)/R_EARTH)  # flat earth depth

    if test:  # then cartesian
        htab_f = htab

    res_f = np.diff(htab_f)  # earth flattened resolution

    # delay times
    dt = np.zeros(np.shape(htab))

    # angular distances of piercing points
    delta = np.zeros(np.shape(htab))

    # vertical slownesses
    q_a = np.zeros(np.shape(htab))
    q_b = np.zeros(np.shape(htab))

    vpf, vsf = model.query(lat, lon, 0)

    q_a[0] = np.sqrt(vpf**-2 - p**2)
    q_b[0] = np.sqrt(vsf**-2 - p**2)

    # Numerical integration
    for kk, h in enumerate(htab_f[1:]):
        # ii = np.where(model.zf > h)[0][0] - 1  # >=

        # # When there is elevation (velocity model starts at depth 0!)
        # if ii == -1:
        #     ii = 0

        with warnings.catch_warnings():
            # Catch Zerodivision warnings
            warnings.filterwarnings('error')
            try:
                delta[kk+1] = ppoint(
                    q_a[kk], q_b[kk], res_f[kk], p, phase) + delta[kk]
            except Warning:
                delta = delta[:kk+1]
                break

        az = baz

        # if np.isnan(delta[kk+1]) or q_a[ii] == 0 or q_b[ii] == 0:
        #     # Supercritical
        #     delta = delta[:kk+1]
        #     break

        coords = Geodesic.WGS84.ArcDirect(lat, lon, az, delta[kk+1])
        lat2, lon2 = coords['lat2'], coords['lon2']

        # Query depth
        qd = htab[kk]
        if qd < 0:  # For elevation, model is not defined for el
            qd = 0

        vpf, vsf = model.query(lat2, lon2, qd)
        # Ignore invalid square root warning
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        q_a[kk+1] = np.sqrt(vpf**-2 - p**2)
        q_b[kk+1] = np.sqrt(vsf**-2 - p**2)

        if np.isnan(q_a[kk+1]) or np.isnan(q_b[kk+1]) or\
                q_a[kk+1] == 0 or q_b[kk+1] == 0:
            # Supercritical, can really only happen for q_a
            delta = delta[:kk+1]
            break

        dt[kk+1] = (q_b[kk]-q_a[kk])*res_f[kk]

    dt = np.cumsum(dt)[:len(delta)]

    return htab[:len(delta)], dt, delta


def dt_table(rayp, fname, phase, el, debug=False):
    """
    Creates a phase delay table and calculates piercing points
    for a specific ray parameter,
    using the equation as in Rondenay 2009.
    For SRF: The length of the output vectors is determined by the depth, at
    which Sp conversion is supercritical (vertical slowness = complex).

    Parameters
    ----------
    rayp : float
        ray-parameter given in s/deg.
    fname : string
        1D velocity model, located in data/vmodels.
    phase : string
        Either "S" for Sp or "P" for Ps.

    Returns
    -------
    htab: np.array
        Vector containing conversion depths.
    dt: np.array
        Vector containing delay times between primary arrival and converted
        wave.
    delta: np.array
        Vector containing angular distance of piercing point from the
        station at depth z.

    """

    p = rayp/DEG2KM  # convert to s/km for flattened model

    model = load_model(fname)
    z = model.z
    if fname == 'raysum.dat' or debug:  # already Cartesian
        vp = model.vp
        vs = model.vs
        zf = model.z
    else:  # Spherical
        vp = model.vpf
        vs = model.vsf
        zf = model.zf

    z = np.append(z, maxz)

    # hypothetical conversion depth tables
    if el > 0:
        htab = np.hstack(  # spherical
                         (np.arange(-round(el/1000, 1), 0, .1),
                          np.arange(0, maxz+res, res)))
    else:
        htab = np.arange(-round(el/1000), maxz+res, res)

    htab_f = -R_EARTH*np.log((R_EARTH-htab)/R_EARTH)  # flat earth depth

    res_f = np.diff(htab_f)  # earth flattened resolution

    if fname == "raysum.dat" or debug:
        htab_f = htab
        res_f = np.diff(htab)

    # delay times
    dt = np.zeros(np.shape(htab))

    # angular distances of piercing points
    delta = np.zeros(np.shape(htab))

    # vertical slownesses
    q_a = np.sqrt(vp**-2 - p**2)
    q_b = np.sqrt(vs**-2 - p**2)

    # Numerical integration
    for kk, h in enumerate(htab_f[1:]):
        ii = np.where(zf >= h)[0][0] - 1
        if ii == -1:
            ii = 0
        dt[kk+1] = (q_b[ii]-q_a[ii])*res_f[kk]
        delta[kk+1] = ppoint(q_a[ii], q_b[ii], res_f[kk], p, phase)
    dt = np.cumsum(dt)
    delta = np.cumsum(delta)

    # Determine if Sp is supercritical
    try:
        index = np.nonzero(np.isnan(dt))[0][0]
    except IndexError:
        index = len(dt)
    return htab[:index], dt[:index], delta[:index]


def ppoint(q_a, q_b, dz, p, phase):
    """
    Calculate Euclidean distance between piercing point and station.
    INPUT has to be in Cartesian/ flat Earth:

    Parameters
    ----------
    q_a : float
        Vertical P-wave slowness in s/km.
    q_b : float
        Vertical S-wave slowness in s/km.
    dz : float
        Vertical distance between two consecutive piercing points [km].
    p : float
        slowness in s/km.
    phase : string
        Either "S" or "P".

    Returns
    -------
    x_delta : float
        Euclidean distance between station and piercing point.

    """

    # Check phase, Sp travels as p, Ps as S from piercing point
    if phase == "S":
        x = dz / q_a * p
    elif phase == "P":
        x = dz / q_b * p
    x_delta = x / DEG2KM  # distance in degree
    return x_delta


def earth_flattening(maxz, z, dz, vp, vs):
    """
    Creates a flat-earth approximated velocity model down to maxz as in
    Peter M. Shearer

    Parameters
    ----------
    maxz : int
        Maximal intepolation depth [km]. Given at beginning of file.
    z : np.array
        Depth vector [km].
    dz : np.array
        Layer thicknesses in km.
    vp : np.array
        P wave velocities [km/s].
    vs : np.array
        S wave velocities [km/s].

    Returns
    -------
    z : np.array
        Same as input, but shortened to maxz.
    dz : np.array
        Same as input, but shortened to maxz.
    zf : np.array
        Depths in a Cartesian coordinate system.
    dzf : np.array
        Layer thicknesse in a Cartesian coordinate system.
    vpf : np.array
        P wave velocities in a Cartesian coordinate system.
    vsf : np.array
        S wave velocities in a Cartesian coordinate system.

    """

    a = R_EARTH  # Earth's radius
    ii = np.where(z > maxz)[0][0]+1
    r = a - z[:ii]
    vpf = (a/r) * vp[:ii]
    vsf = (a/r) * vs[:ii]
    zf = -a * np.log(r/a)
    z = z[:ii]
    dz = dz[:ii]
    dzf = np.diff(zf)

    return z, dz, zf, dzf, vpf, vsf


""" Everything from here on is from the RF model by Tom Eulenfeld, modified
 by me"""


def load_model(fname='iasp91.dat'):
    """
    Load 1D velocity model from file.
    The model file should have 4 columns with depth, vp, vs, n.
    The model file for iasp91 starts like this::

        #IASP91 velocity model
        #depth  vp    vs   n
          0.00  5.800 3.360 0
          0.00  5.800 3.360 0
        10.00  5.800 3.360 4

    Parameters
    ----------
    fname : string, optional
        filename of model in data or 'iasp91'.
        The default is 'iasp91.dat'.

    Returns
    -------
    SimpleModel
        Returns SimpleModel instance.

    """

    try:
        return _MODEL_CACHE[fname]
    except KeyError:
        pass
    values = np.loadtxt('data/velocity_models/' + fname, unpack=True)
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
    Calculated with Earth flattening algorithm as described by P. M. Shearer.

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
        dz = np.diff(z)
        self.z, self.dz, self.zf, self.dzf, self.vpf, self.vsf = \
            earth_flattening(maxz, z[:-1], dz, vp[:-1], vs[:-1])
        # self.dz =
        # self.z = z[:-1]
        self.vp = vp[:len(self.z)]
        self.vs = vs[:len(self.z)]
        self.t_ref = {}
