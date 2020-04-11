#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:08:42 2020

Contains functions for moveout correction and station stacking

@author: pm
"""

import numpy as np
import shelve
import os
from scipy import interpolate
from scipy.signal.windows import hann

from obspy import read
from obspy.geodetics import degrees2kilometers

import config


_MODEL_CACHE = {}
DEG2KM = degrees2kilometers(1)
maxz = 800  # maximum interpolation depth in km
res = 1 # vertical resolution in km for interpolation and ccp bins


def stackRF(network, station, phase=config.phase):
    """
    Outdated. Use fun:'subfunctions.createRF.RFStream.statstack()' instead.

    Creates a moveout corrected receiver function stack of the
    requested station"""
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
        z, RF, _ = moveout(tr, st[ii], onset[jj], rayp[jj], phase)
        RF_mo.append(RF)

    stack = np.average(RF_mo, axis=0)
    return z, stack, RF_mo, data


def moveout(data, st, fname='iasp91.dat'):
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
    fname : string, optional
        1D velocity model for moveout correction. The default is 'iasp91.dat'.

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
    rayp = st.slowness
    phase = st.phase
    # Nq = st.npts - tas
    htab, dt, delta = dt_table(rayp, fname, phase)
    # queried times
    tq = np.arange(0, round(max(dt), 1), st.delta)
    z = np.interp(tq, dt, htab)  # Depth array

    # Flip SRF
    if phase.upper() == "S":
        data = np.flip(data)
        data = -data
        tas = -tas
    RF = data[tas:tas+len(z)]
    zq = np.arange(0, max(z)+res, res)

    # interpolate RF
    tck = interpolate.splrep(z, RF)
    RF = interpolate.splev(zq, tck)

    # Taper the last 3.5 seconds of the RF to avoid discontinuities in stack
    tap = hann(round(7/st.delta))
    tap = tap[round(len(tap)/2):]
    taper = np.ones(len(RF))
    taper[-len(tap):] = tap
    RF = np.multiply(taper, RF)
    z2 = np.arange(0, maxz+res, res)
    RF2 = np.zeros(z2.shape)
    RF2[:len(RF)] = RF
    return z2, RF2, delta


def dt_table(rayp, fname, phase):
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
        Vector containing Euclidian distance of piercing point from the
        station at depth z.

    """

    p = rayp/DEG2KM  # convert to s/km

    model = load_model(fname)
    z = model.z
    if fname == 'raysum.dat':  # already Cartesian
        vp = model.vp
        vs = model.vs
        zf = model.z
    else:  # Spherical
        vp = model.vpf
        vs = model.vsf
        zf = model.zf
    # dz = model.dz
    # dzf = model.dzf
    z = np.append(z, maxz)

    # hypothetical conversion depth tables
    htab = np.arange(0, maxz+res, res)  # spherical
    htab_f = -6371*np.log((6371-htab)/6371)  # flat earth depth
    res_f = np.diff(htab_f)  # earth flattened resolution

    if fname == "raysum.dat":
        htab_f = htab
        res_f = res * np.ones(np.shape(res_f))

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
        dt[kk+1] = (q_b[ii]-q_a[ii])*res_f[kk]
        delta[kk+1] = ppoint(q_a[ii], q_b[ii], res_f[kk], p, phase)
    dt = np.cumsum(dt)
    delta = np.cumsum(delta)

    # Determine if Sp is supercritical
    try:
        index = np.nonzero(np.isnan(dt))[0][0] - 1
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
        x = np.sum(dz / q_a) * p
    elif phase == "P":
        x = dz / q_b * p
    x_delta = x / DEG2KM  # Euclidean distance in degree
    return x_delta


def earth_flattening(maxz, z, dz, vp, vs):
    """
    Creates a flat-earth approximated velocity model down to 800 km as in
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

    r = 6371  # Earth's radius
    ii = np.where(z > maxz)[0][0]+1
    vpf = (r/(r-z[:ii]))*vp[:ii]
    vsf = (r/(r-z[:ii]))*vs[:ii]
    zf = -r*np.log((r-z[:ii+1])/r)[:ii]
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
        self.vp = vp[:-1]
        self.vs = vs[:-1]
        self.t_ref = {}
