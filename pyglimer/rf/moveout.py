#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:08:42 2020

Contains functions for moveout correction and station stacking.

Author:
    Peter Makus (peter.makus@student.uib.no)
  
!The file is split and has a second copyright disclaimer!

Last updated:
"""

import os
from pathlib import Path
import shelve
import warnings

import numpy as np
from obspy import read
from obspy.signal.filter import lowpass
from geographiclib.geodesic import Geodesic
from scipy import interpolate
from scipy.signal.windows import hann

from pyglimer.data  import finddir
from pyglimer.constants import R_EARTH, DEG2KM, maxz, res, maxzm
from pyglimer.utils.createvmodel import load_gyps


_MODEL_CACHE = {}


def moveout(data, st, fname, latb, lonb, taper, multiple:bool=False):
    """
    Depth migration for RF.
    Corrects the for the moveout of a converted phase. Flips time axis
    and polarity of SRF.

    :param data: Receiver Function.
    :type data: 1D np.ndarray
    :param st: Stats from stream object.
    :type st: :class:`~obspy.core.AttributeDict`
    :param fname: 1D velocity model for moveout correction.
        Use '3D' for a 3D raytracing.
    :type fname: str
    :param latb: Tuple in Form (minlat, maxlat). To save RAM on 3D raytraycing.
        Will remain unused for 1D RT.
    :type latb: tuple
    :param lonb: Tuple in Form (minlon, maxlon)
    :type lonb: tuple
    :param taper: If True, the last 10km of the RF will be tapered, which avoids
        jumps in station stacks. If False,
        the upper 5 km will be tapered.
        Should be False for CCP stacks.
    :type taper: bool
    :param multiple: Either False (don't include multiples), 'linear' for
        linear stack, 'PWS' (phase weighted stack or "zk" for a zhu &
        kanamori approach). False by default.
    :type multiple: bool or str, optional
    :return: z2 : Depth vector in km.
        RF2 : Vector containing the depth migrated RF.
        delta : Vector containing Euclidian distance of piercing point from the
            station at depth z.
    :rtype: 3 times 1D np.ndarray
    """

    onset = st.onset
    tas = round((onset - st.starttime)/st.delta)  # theoretical arrival sample
    rayp = st.slowness  # Ray parameter
    phase = st.phase  # Primary phase
    el = st.station_elevation

    phase = phase[-1]

    if fname[-2:] == '3D':
        test = fname == 'raysum3D'
        if test:
            test = int(st.station)  # The dip of the LABx
        htab, dt, delta, dtm1, dtm2 = dt_table_3D(
            rayp, phase, st.station_latitude, st.station_longitude,
            st.back_azimuth, el, latb, lonb, multiple, test=test)
        # Multiple modes

    else:
        # dtm1 is PPS for PRFs (SSP SRFs), dtm2 PSS (SPP SRFs) 
        htab, dt, delta, dtm1, dtm2 = dt_table(
            rayp, fname, phase, el, multiple)

    # queried times
    tq = np.arange(0, round(max(dt)+st.delta, 1), st.delta)
    z = np.interp(tq, dt, htab)  # Depth array for first RF (not evenly spaced)
    
    # Flip SRF
    if phase.upper() == "S":
        data = np.flip(data)
        data = -data
        tas = -tas

    # Shorten RF
    data = data[tas:]

    # Taper the first 2.5 seconds
    i = round(2.5/st.delta)  # Find where rf is depth = 5
    tap = hann((i+1)*2)
    up, _ = np.split(tap, 2)
    if len(data) > len(up):  # That usually doesn't happen, only for extreme
        # discontinuities in 3D model and errors in SRF data
        taperfun = np.ones(len(data))
        taperfun[:len(up)] = up
        data = np.multiply(taperfun, data)

    if len(z) <= len(data):
        RF = data[:len(z)]
    else:  # truncate z not RF
        # 16.07.2020 this shouldn't be happening, but I'm experiencing an error
        # that might be prevented here
        z = z[:len(data)]
        RF = data[:len(z)]
        
    if round(min(z), int(-np.log10(res))) <= 0:
        zq = np.hstack((np.arange(min(z), 0, .1), np.arange(0, max(z)+res, res)))
    else:
        zq = np.arange(round(min(z), int(-np.log10(res))), max(z)+res)

    if multiple:
        # lowpass filter see Tauzin et. al. (2016)
        RF = lowpass(RF, 1, st.sampling_rate, zerophase=True)
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

    # for the multiple modes
    if multiple:
        # Multiples are only useful for the upper part of the lithosphere
        # I will go with the upper ~constants.maxzm km for now (I might have to reduce that)
        if htab[len(dtm1)-1] > maxzm:
            dtm1 = dtm1[:np.where(htab>=maxzm)[0][0]]
        if htab[len(dtm2)-1] > maxzm:
            dtm2 = dtm2[:np.where(htab>=maxzm)[0][0]]
        # if phase == 'P':
        tqm1 = np.arange(0, round(max(dtm1)+st.delta, 1), st.delta)
        tqm2 = np.arange(0, round(max(dtm2)+st.delta, 1), st.delta)
        zm1 = np.interp(tqm1, dtm1, htab[:len(dtm1)])
        zm2 = np.interp(tqm2, dtm2, htab[:len(dtm2)])
        # truncate RF
        RFm1 = data[:len(zm1)]
        RFm2 = data[:len(zm2)]
        # else:
        #     # For SRFs that's a bit more complicated as multiples arrive
        #     # after first arrival, whereas conversions arrive before
        #     tqm1 = np.arange(round(min(dtm1), 1), st.delta/2, st.delta)
        #     tqm2 = np.arange(round(min(dtm2), 1), st.delta/2, st.delta)
        #     zm1 = np.interp(tqm1, np.flip(dtm1), np.flip(htab[:len(dtm1)]))
        #     zm2 = np.interp(tqm2, np.flip(dtm2), np.flip(htab[:len(dtm2)]))
        #     # truncate RFs
        #     RFm1 = data[-len(zm1):tas]
        #     RFm2 = data[tas-len(zm2):tas]
        #     # Now they need to be flipped again, so the deepest depth are at
        #     # the end
        #     zm1 = np.flip(zm1)
        #     zm2 = np.flip(zm2)
        #     RFm1 = np.flip(RFm1)
        #     RFm2 = np.flip(RFm2)
        # lowpass filter see Tauzin et. al. (2016)
        RFm1 = lowpass(RFm1, .25, st.sampling_rate, zerophase=True)
        RFm2 = lowpass(RFm2, .25, st.sampling_rate, zerophase=True)
        try:
            tckm1 = interpolate.splrep(zm1, RFm1)
            tckm2 = interpolate.splrep(zm2, RFm2)
        except TypeError as e:
            multiple = False
            # u, c = np.unique(zm1, return_counts=True)
            # dup = u[c > 1]
            # u, c = np.unique(zm2, return_counts=True)
            # dup2 = u[c > 1]
            mes = "Interpolation error in multiples. Only primary conversion"+\
                " will be used."
            warnings.warn(mes, category=UserWarning, stacklevel=1)
            pass
    
    if multiple:
        # query depths
        if round(min(zm1), int(-np.log10(res))) <= 0:
            zqm1 = np.hstack(
                (np.arange(min(zm1), 0, .1), np.arange(0, max(zm1)+res, res)))
            zqm2 = np.hstack(
                (np.arange(min(zm2), 0, .1), np.arange(0, max(zm2)+res, res)))
        else:
            zqm1 = np.arange(
                round(min(zm1), int(-np.log10(res))), max(zm1)+res)
            zqm2 = np.arange(
                round(min(zm2), int(-np.log10(res))), max(zm2)+res)

        RFm1 = interpolate.splev(zqm1, tckm1)
        RFm2 = -interpolate.splev(zqm2, tckm2)  # negative due to polarity change

    if taper:
        # Taper the last 10 km
        tap = hann(20)
        _, down = np.split(tap, 2)

        if len(RF) > len(down):  # That usually doesn't happen, only for extreme
            # discontinuities in 3D model and errors in SRF data
            taper = np.ones(len(RF))
            taper[-len(down):] = down
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
    if multiple:
        RFm1_2 = np.zeros(RF2.shape)
        RFm2_2 = np.zeros(RF2.shape)
        RFm1_2[starti:starti+len(RFm1)] = RFm1
        RFm2_2[starti:starti+len(RFm2)] = RFm2
        
    else:
        RFm1_2 = None
        RFm2_2 = None

    # reshape delta - else that will mess with the CCP stacking
    delta2 = np.empty(z2.shape)
    delta2.fill(np.nan)
    delta2[starti:starti+len(delta)] = delta

    return z2, RF2, delta2, RFm1_2, RFm2_2


def dt_table_3D(
    rayp, phase, lat, lon, baz, el, latb, lonb, multiple:bool, test=False):    
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

    # travel times for alpha (P) and beta (S)
    dt_a = np.zeros(np.shape(htab))
    dt_b = np.zeros(np.shape(htab))

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
        
        # travel time for P and S, respectively through dz
        dt_a[kk+1] = q_a[kk]*res_f[kk]
        dt_b[kk+1] = q_b[kk]*res_f[kk]
    
    dt_a = np.cumsum(dt_a)[:len(delta)]
    dt_b = np.cumsum(dt_b)[:len(delta)]
    dt = dt_b-dt_a  # Delay primary conversion

    # Find travel times for multiples
    # We will have to assume that the multiples don't cross into a different
    # bin of the velocity model. Else the computation would be blown up crazily
    if multiple:
        # This assumes that the elevation is the same as at the station location
        # possible weakspot! Maybe I should do a lookup?
        # mphase 1 is PPs for P and SSp for S
        # mphase 2 is PSs, and SPp, respectively
        if phase == 'P':
            # dt_mphase1 = dt + 2*dt_a
            # dt_mphase2 = dt + dt_b + dt_a
            # The two are the same, first one just overly complicated
            dt_mphase1 = dt_b + dt_a
            dt_mphase2 = 2*dt_b
            # Truncate The travel time table for multiples
            # if dt_mphase1.max() > 100 or dt_mphase2.max() > 100:
            try:
                dt_mphase1 = dt_mphase1[:np.where(dt_mphase1>=100)[0][0]]
            except IndexError:
                pass
            try:
                dt_mphase2 = dt_mphase2[:np.where(dt_mphase2>=100)[0][0]]
            except IndexError:
                pass
            # elif dt_mphase2.max() > 100:
            #     dt_mphase2 = dt_mphase2[:np.where(dt_mphase2>=100)[0][0]]               
        else:
            # Reduce travel times for S since the data will be flipped
            dt_mphase1 = dt - 2*dt_b
            dt_mphase2 = dt - dt_b - dt_a
            if dt_mphase2.min() < -50:
                dt_mphase1 = dt_mphase1[:np.where(dt_mphase1<=-50)[0][0]]
                dt_mphase2 = dt_mphase2[:np.where(dt_mphase2<=-50)[0][0]]
            elif dt_mphase1.min() < -50:
                dt_mphase1 = dt_mphase1[:np.where(dt_mphase1<=-50)[0][0]]
    else:
        dt_mphase1 = None
        dt_mphase2 = None

    return htab[:len(delta)], dt, delta, dt_mphase1, dt_mphase2


def dt_table(rayp, fname, phase, el, multiple:bool, debug=False):
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
    multiple : bool
        Compute conversion time for PPS and PSS.
    debug : bool, optional
        If True, A Cartesioan rather than a spherical Earth is assumed. By
        default False.


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
    #dt = np.zeros(np.shape(htab))
    dt_a = np.zeros(np.shape(htab))
    dt_b = np.zeros(np.shape(htab))

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
        # dt[kk+1] = (q_b[ii]-q_a[ii])*res_f[kk]
        dt_a[kk+1] = q_a[ii]*res_f[kk]
        dt_b[kk+1] = q_b[ii]*res_f[kk]
        delta[kk+1] = ppoint(q_a[ii], q_b[ii], res_f[kk], p, phase)

    dt_a = np.cumsum(dt_a)
    dt_b = np.cumsum(dt_b)
    dt = dt_b-dt_a  # Travel time difference primary and converted phase
    #dt = np.cumsum(dt)

    delta = np.cumsum(delta)

    # Determine if Sp is supercritical
    try:
        index = np.nonzero(np.isnan(dt))[0][0]
    except IndexError:
        index = len(dt)

    # Find travel times for multiples
    # In the 1D vel model ppoint don't play any role for multiples
    if multiple:
        # This assumes that the elevation is the same as at the station location
        # possible weakspot! Maybe I should do a lookup?
        # mphase 1 is PPs for P and SSp for S
        # mphase 2 is PSs, and SPp, respectively
        if phase == 'P':
            dt_mphase1 = dt_b + dt_a
            dt_mphase2 = 2*dt_b
            # Truncate The travel time table for multiples
            # if dt_mphase1.max() > 100 or dt_mphase2.max() > 100:
            try:
                dt_mphase1 = dt_mphase1[:np.where(dt_mphase1>=100)[0][0]]
            except IndexError:
                pass
            try:
                dt_mphase2 = dt_mphase2[:np.where(dt_mphase2>=100)[0][0]]
            except IndexError:
                pass
            # elif dt_mphase2.max() > 100:
            #     print('h2')
            #     dt_mphase2 = dt_mphase2[:np.where(dt_mphase2>=100)[0][0]]               
        else:
            # Reduce travel times for S since the data will be flipped
            dt_mphase1 = dt - 2*dt_b
            dt_mphase2 = dt - dt_b - dt_a
            if dt_mphase2.min() < -50:
                dt_mphase1 = dt_mphase1[:np.where(dt_mphase1<=-50)[0][0]]
                dt_mphase2 = dt_mphase2[:np.where(dt_mphase2<=-50)[0][0]]
            elif dt_mphase1.min() < -50:
                dt_mphase2 = dt_mphase2[:np.where(dt_mphase1<=-50)[0][0]]
        dt_mphase1 = dt_mphase1[:index]
        dt_mphase2 = dt_mphase2[:index]
    else:
        dt_mphase1 = None
        dt_mphase2 = None

    return htab[:index], dt[:index], delta[:index], dt_mphase1, dt_mphase2


def ppoint(q_a, q_b, dz, p, phase):
    """
    Calculate spherical distance between piercing point and station.
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


"""
All objects and functions below are modified versions as in Tom Eulenfeld's
rf project or contain substantial parts from this code. License below.

The MIT License (MIT)

Copyright (c) 2013-2019 Tom Eulenfeld

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


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
    filepath = os.path.join(finddir(), 'velocity_models', fname)
    values = np.loadtxt(filepath, unpack=True)
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
