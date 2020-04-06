#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:10 2020

@author: pm

toolset to create RFs and RF stacks
"""
import json
from operator import itemgetter
from pkg_resources import resource_filename
import warnings

import numpy as np
import config
from subfunctions.deconvolve import it, spectraldivision, multitaper
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers,\
    direct_geodetic
from obspy.taup import TauPyModel
import shelve
import os

from scipy import interpolate
from scipy.signal.windows import hann

_MODEL_CACHE = {}
DEG2KM = degrees2kilometers(1)
maxz = 800  # maximum interpolation depth in km


def createRF(st_in, dt, phase=config.phase, shift=config.tz,
             method=config.decon_meth, trim=False, event=None, station=None,
             info=None):
    """Creates a receiver function with the defined method from an obspy
    stream.
    INPUT:
        st: stream
        dt: sampling interval (s)
        phase: "S" for Sp or "P" for Ps
        shift: time shift of the main arrival
        method: deconvolution method, "waterlevel", 'dampedf' for constant
        damping level, 'it' for iterative time domain deconvoltuion, 'multit'
        for multitaper or 'fqd' for frequency dependently damped spectral
        division.
        trim: taper/truncate. Given as list [a, b] in s - left,right"""
    st = st_in.copy()
    RF = st.copy()
    st.normalize()
    while RF.count() > 1:
        del RF[1]
    RF[0].stats.channel = phase + "RF"

    # taper traces
    if trim:
        if not type(trim) == list and len(trim) != 2:
            raise Exception("""Trim has to be given as list with two elements
                            [a, b]. Where a and b are the taper length in s
                            on the left and right side, respectively.""")
        # Hann taper with 7.5s taper window
        tap = hann(round(15/dt))
        taper = np.ones(st[0].stats.npts)
        taper[:int((trim[0]-7.5)/dt)] = float(0)
        taper[-int((trim[1]-7.5)/dt):] = float(0)
        taper[int((trim[0]-7.5)/dt):int(trim[0]/dt)] = tap[:round(7.5/dt)]
        taper[-int(trim[1]/dt):-int((trim[1]-7.5)/dt)] =\
            tap[-round(7.5/dt):]
        for tr in st:
            tr.data = np.multiply(tr.data, taper)

    # Identify components
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

    # Deconvolution
    if method == "it":
        if phase == "S":
            width = 1.25  # .75
        elif phase == "P":
            width = 2.5
        RF[0].data = it(v, u, dt, shift=shift, width=width)[0]
    elif method == "dampedf":
        RF[0].data, _ = spectraldivision(v, u, dt, shift, "con", phase=phase)
    elif method == "waterlevel":
        RF[0].data, _ = spectraldivision(v, u, dt, shift, "wat", phase=phase)
    elif method == 'fqd':
        RF[0].data, _ = spectraldivision(v, u, dt, shift, "fqd", phase=phase)
    elif method == 'multit':
        RF[0].data, _, _, _ = multitaper(v, u, dt, shift, "fqd")
        # remove noise caused by multitaper
        RF.filter('lowpass', freq=4.99, zerophase=True, corners=2)
    else:
        raise Exception(method, "is no valid deconvolution method.")

    # create RFTrace object
    # create stats
    stats = rfstats(info=info, starttime=st[0].stats.starttime, event=event,
                    station=station, phase=phase)
    RF = RFTrace(trace=RF[0])
    RF.stats.update(stats)
    return RF


def stackRF(network, station, phase=config.phase):
    """Creates a moveout corrected receiver function stack of the
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
    """Corrects the for the moveout of a converted phase. Flips time axis
    and polarity of SRF.
    INPUT:
        data: data from the trace object
        st: stats from the trace object
        onset: the onset time (UTCDATETIME)
        rayp: ray parameter in sec/deg
        phase: P or S for PRF or SRF, respectively"""
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
    zq = np.arange(0, max(z), 0.25)

    # interpolate RF
    tck = interpolate.splrep(z, RF)
    RF = interpolate.splev(zq, tck)

    # Taper the last 3.5 seconds of the RF to avoid discontinuities in stack
    tap = hann(round(7/st.delta))
    tap = tap[round(len(tap)/2):]
    taper = np.ones(len(RF))
    taper[-len(tap):] = tap
    RF = np.multiply(taper, RF)
    z2 = np.arange(0, maxz+.25, .25)
    RF2 = np.zeros(z2.shape)
    RF2[:len(RF)] = RF
    return z2, RF2, delta


def dt_table(rayp, fname, phase):
    """Creates a phase delay table for a specific ray parameter given in
    s/deg. Using the equation as in Rondenay 2009."""
    p = rayp/DEG2KM  # convert to s/km
    # z, dz, zf, dzf, vp, vs = iasp_flatearth(maxz, fname)
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
    res = 1  # resolution in km

    # hypothetical conversion depth tables
    htab = np.arange(0, maxz+res, res)  # spherical
    htab_f = -6371*np.log((6371-htab)/6371)  # flat earth depth
    res_f = np.diff(htab_f)  # earth flattened resolution
    # htab_f = htab_f[:-1]
    # htab = htab[:-1]

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

    try:
        index = np.nonzero(np.isnan(dt))[0][0] - 1
    except IndexError:
        index = len(dt)
    return htab[:index], dt[:index], delta[:index]


def ppoint(q_a, q_b, dz, p, phase):
    """
    Calculate angular distance between piercing point and station.
    INPUT (have to be in Cartesian/ flat Earth):
    :param depth: depth of interface in km
    :param p: slowness in s/km
    :param phase: 'P' or 'S'
    :return: angular distance in degree
    """
    # The case for the direct - conversion at surface, dz =[]
    # if not len(dz):
    #     x = 0

    # Check phase, Sp travels as p, Ps as S from piercing point
    if phase == "S":
        x = np.sum(dz / q_a) * p
    elif phase == "P":
        x = dz / q_b * p
    x_delta = x / DEG2KM  # angular distance in degree
    return x_delta


def earth_flattening(maxz, z, dz, vp, vs):
    """Creates a flat-earth approximated velocity model down to 800 km as in
    Peter M. Shearer"""
    r = 6371  # earth radius
    ii = np.where(z > maxz)[0][0]+1
    vpf = (r/(r-z[:ii]))*vp[:ii]
    vsf = (r/(r-z[:ii]))*vs[:ii]
    zf = -r*np.log((r-z[:ii+1])/r)
    z = z[:ii]
    dz = dz[:ii]
    dzf = np.diff(zf)
    # if fname == 'raysum.dat':
    #     zf = z
    #     vpf = vp
    #     vsf = vs
    return z, dz, zf[:ii], dzf, vpf, vsf


""" Everything from here on is from the RF model by Tom Eulenfeld, modified
 by me"""


def load_model(fname='iasp91.dat'):
    """
    Load model from file.

    :param fname: filename of model in data or 'iasp91'
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


def __get_event_origin_prop(h):
    def wrapper(event):
        try:
            r = (event.preferred_origin() or event.origins[0])[h]
        except IndexError:
            raise ValueError('No origin')
        if r is None:
            raise ValueError('No origin ' + h)
        if h == 'depth':
            r = r / 1000
        return r
    return wrapper


def __get_event_magnitude(event):
    try:
        return (event.preferred_magnitude() or event.magnitudes[0])['mag']
    except IndexError:
        raise ValueError('No magnitude')


def __get_event_id(event):
    evid = event.get('resource_id')
    if evid is not None:
        evid = str(evid)
    return evid


def __SAC2UTC(stats, head):
    from obspy.io.sac.util import get_sac_reftime
    return get_sac_reftime(stats.sac) + stats[head]


def __UTC2SAC(stats, head):
    from obspy.io.sac.util import get_sac_reftime
    return stats[head] - get_sac_reftime(stats.sac)


_STATION_GETTER = (('station_latitude', itemgetter('latitude')),
                   ('station_longitude', itemgetter('longitude')),
                   ('station_elevation', itemgetter('elevation')))
_EVENT_GETTER = (
    ('event_latitude', __get_event_origin_prop('latitude')),
    ('event_longitude', __get_event_origin_prop('longitude')),
    ('event_depth', __get_event_origin_prop('depth')),
    ('event_magnitude', __get_event_magnitude),
    ('event_time', __get_event_origin_prop('time')),
    ('event_id', __get_event_id))

# header values which will be written to waveform formats (SAC and Q)
# H5 simply writes all stats entries
_HEADERS = (tuple(zip(*_STATION_GETTER))[0] +
            tuple(zip(*_EVENT_GETTER))[0][:-1] + (  # do not write event_id
            'onset', 'type', 'phase', 'moveout',
            'distance', 'back_azimuth', 'inclination', 'slowness',
            'pp_latitude', 'pp_longitude', 'pp_depth',
            'box_pos', 'box_length'))

# The corresponding header fields in the format
# The following headers can at the moment only be stored for H5:
# slowness_before_moveout, box_lonlat, event_id
_FORMATHEADERS = {'sac': ('stla', 'stlo', 'stel', 'evla', 'evlo',
                          'evdp', 'mag', 'o', 'a',
                          'kuser0', 'kuser1', 'kuser2',
                          'gcarc', 'baz', 'user0', 'user1',
                          'user2', 'user3', 'user4',
                          'user5', 'user6'),
                  # field 'COMMENT' is violated for different information
                  'sh': ('COMMENT', 'COMMENT', 'COMMENT',
                         'LAT', 'LON', 'DEPTH',
                         'MAGNITUDE', 'ORIGIN', 'P-ONSET',
                         'COMMENT', 'COMMENT', 'COMMENT',
                         'DISTANCE', 'AZIMUTH', 'INCI', 'SLOWNESS',
                         'COMMENT', 'COMMENT', 'COMMENT',
                         'COMMENT', 'COMMENT')}
_HEADER_CONVERSIONS = {'sac': {'onset': (__SAC2UTC, __UTC2SAC),
                               'event_time': (__SAC2UTC, __UTC2SAC)}}


_TF = '.datetime:%Y-%m-%dT%H:%M:%S'

_H5INDEX = {
    'rf': ('waveforms/{network}.{station}.{location}/{event_time%s}/' % _TF +
           '{channel}_{starttime%s}_{endtime%s}' % (_TF, _TF)),
    'profile': 'waveforms/{channel[2]}_{box_pos}'
}


def read_rf(pathname_or_url=None, format=None, **kwargs):
    """
    Read waveform files into RFStream object.
    See :func:`~obspy.core.stream.read` in ObsPy.
    """
    if pathname_or_url is None:   # use example file
        fname = resource_filename('rf', 'example/minimal_example.tar.gz')
        pathname_or_url = fname
        format = 'SAC'
    stream = read(pathname_or_url, format=format, **kwargs)
    return RFStream(stream)


class RFStream(Stream):

    """
    Class providing the necessary functions for receiver function calculation.
    :param traces: list of traces, single trace or stream object
    To initialize a RFStream from a Stream object use
    >>> rfstream = RFStream(stream)
    To initialize a RFStream from a file use
    >>> rfstream = read_rf('test.SAC')
    Format specific headers are loaded into the stats object of all traces.
    """

    def __init__(self, traces=None):
        self.traces = []
        if isinstance(traces, Trace):
            traces = [traces]
        if traces:
            for tr in traces:
                if not isinstance(tr, RFTrace):
                    tr = RFTrace(trace=tr)
                self.traces.append(tr)

    def __is_set(self, header):
        return all(header in tr.stats for tr in self)

    def __get_unique_header(self, header):
        values = set(tr.stats[header] for tr in self if header in tr.stats)
        if len(values) > 1:
            warnings.warn('Header %s has different values in stream.' % header)
        if len(values) == 1:
            return values.pop()

    @property
    def type(self):
        """Property for the type of stream, 'rf', 'profile' or None"""
        return self.__get_unique_header('type')

    @type.setter
    def type(self, value):
        for tr in self:
            tr.stats.type = value

    @property
    def method(self):
        """Property for used rf method, 'P' or 'S'"""
        phase = self.__get_unique_header('phase')
        if phase is not None:
            return phase.upper()

    @method.setter
    def method(self, value):
        for tr in self:
            tr.stats.phase = value

    def write(self, filename, format, **kwargs):
        """
        Save stream to file including format specific headers.
        See `Stream.write() <obspy.core.stream.Stream.write>` in ObsPy.
        """
        if len(self) == 0:
            return
        for tr in self:
            tr._write_format_specific_header(format)
            if format.upper() == 'Q':
                tr.stats.station = tr.id
        if format.upper() == 'H5':
            index = self.type
            if index is None and 'event_time' in self[0].stats:
                index = 'rf'
            if index:
                import obspyh5
                old_index = obspyh5._INDEX
                obspyh5.set_index(_H5INDEX[index])
        super(RFStream, self).write(filename, format, **kwargs)
        if format.upper() == 'H5' and index:
            obspyh5.set_index(old_index)
        if format.upper() == 'Q':
            for tr in self:
                tr.stats.station = tr.stats.station.split('.')[1]

    def trim2(self, starttime=None, endtime=None, reftime=None, **kwargs):
        """
        Alternative trim method accepting relative times.
        See :meth:`~obspy.core.stream.Stream.trim`.
        :param starttime,endtime: accept UTCDateTime or seconds relative to
            reftime
        :param reftime: reference time, can be an UTCDateTime object or a
            string. The string will be looked up in the stats dictionary
            (e.g. 'starttime', 'endtime', 'onset').
        """
        for tr in self.traces:
            t1 = tr._seconds2utc(starttime, reftime=reftime)
            t2 = tr._seconds2utc(endtime, reftime=reftime)
            tr.trim(t1, t2, **kwargs)
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        return self

    def slice2(self, starttime=None, endtime=None, reftime=None,
               keep_empty_traces=False, **kwargs):
        """
        Alternative slice method accepting relative times.
        See :meth:`~obspy.core.stream.Stream.slice` and `trim2()`.
        """
        traces = []
        for tr in self:
            t1 = tr._seconds2utc(starttime, reftime=reftime)
            t2 = tr._seconds2utc(endtime, reftime=reftime)
            sliced_trace = tr.slice(t1, t2, **kwargs)
            if not keep_empty_traces and not sliced_trace.stats.npts:
                continue
            traces.append(sliced_trace)
        return self.__class__(traces)

    def moveout(self, vmodel_file="iasp91.dat"):
        """
        Depth migration of the receiver functions contained in the stream.
        Also calculates piercing points.
        Overwrites traces in stream!
        """
        for tr in self:
            _, tr.data, delta = moveout(tr.data, tr.stats)
            st = tr.stats
            st.pp_latitude = []
            st.pp_longitude = []
            st.pp_depth = np.linspace(0, 800, len(delta))

            # Calculate ppoint position
            for dis in delta:
                lat = st.station_latitude
                lon = st.station_longitude
                az = st.back_azimuth
                dr = st.distance*DEG2KM
                lat2, lon2 = direct_geodetic((lat, lon), az, dr)
                st.pp_latitude.append(lat2)
                st.pp_longitude.append(lon2)

    def ppoint(self, vmodel_file='iasp91.dat'):
        """
        Calculates piercing points for receiver functions in Stream
        """
        for tr in self:
            st = tr.stats
            htab, _, delta = dt_table(st.slowness, vmodel_file, st.phase)
            st.pp_depth = htab
            st.pp_latitude = []
            st.pp_longitude = []

            for dis in delta:
                lat = st.station_latitude
                lon = st.station_longitude
                az = st.back_azimuth
                dr = st.distance*DEG2KM
                lat2, lon2 = direct_geodetic((lat, lon), az, dr)
                st.pp_latitude.append(lat2)
                st.pp_longitude.append(lon2)

    def station_stack(self, vmodel_file='iasp91.dat'):
        """
        Performs a moveout correction and stacks all receiver functions
        in Stream. Make sure that Stream only contains RF from one station!
        """
        self.moveout(vmodel_file=vmodel_file)
        self.normalize()  # make sure traces are normalised
        RF_mo = []
        for tr in self:
            RF_mo.append(tr.data)
        stack = np.average(RF_mo, axis=0)
        self.append(RFTrace(data=stack))
        z = np.linspace(0, 800, len(stack))
        return z, stack


class RFTrace(Trace):

    """
    Class providing the Trace object for receiver function calculation.
    """

    def __init__(self, data=None, header=None, trace=None):
        if header is None:
            header = {}
        if trace is not None:
            data = trace.data
            header = trace.stats
        super(RFTrace, self).__init__(data=data, header=header)
        st = self.stats
        if ('_format'in st and st._format.upper() == 'Q' and
                st.station.count('.') > 0):
            st.network, st.station, st.location = st.station.split('.')[:3]
        self._read_format_specific_header()

    def __str__(self, id_length=None):
        if 'onset' not in self.stats:
            return super(RFTrace, self).__str__(id_length=id_length)
        out = []
        type_ = self.stats.get('type')
        if type_ is not None:
            m = self.stats.get('phase')
            m = m[-1].upper() if m is not None else ''
            o1 = m + 'rf'
            if type_ != 'rf':
                o1 = o1 + ' ' + type_
            if self.id.startswith('...'):
                o1 = o1 + ' (%s)' % self.id[-1]
            else:
                o1 = o1 + ' ' + self.id
        else:
            o1 = self.id
        out.append(o1)
        t1 = self.stats.starttime - self.stats.onset
        t2 = self.stats.endtime - self.stats.onset
        o2 = '%.1fs - %.1fs' % (t1, t2)
        if self.stats.starttime.timestamp != 0:
            o2 = o2 + ' onset:%s' % self.stats.onset
        out.append(o2)
        out.append('{sampling_rate} Hz, {npts} samples')
        o3 = []
        if 'event_magnitude' in self.stats:
            o3.append('mag:{event_magnitude:.1f}')
        if 'distance' in self.stats:
            o3.append('dist:{distance:.1f}')
        if'back_azimuth' in self.stats:
            o3.append('baz:{back_azimuth:.1f}')
        if 'box_pos' in self.stats:
            o3.append('pos:{box_pos:.2f}km')
        if 'slowness' in self.stats:
            o3.append('slow:{slowness:.2f}')
        if 'moveout' in self.stats:
            o3.append('({moveout} moveout)')
        if np.ma.count_masked(self.data):
            o3.append('(masked)')
        out.append(' '.join(o3))
        return ' | '.join(out).format(**self.stats)

    def _read_format_specific_header(self, format=None):
        st = self.stats
        if format is None:
            if '_format' not in st:
                return
            format = st._format
        format = format.lower()
        if format == 'q':
            format = 'sh'
        try:
            header_map = zip(_HEADERS, _FORMATHEADERS[format])
        except KeyError:
            # file format is H5 or not supported
            return
        read_comment = False
        for head, head_format in header_map:
            if format == 'sh' and read_comment:
                continue
            try:
                value = st[format][head_format]
            except KeyError:
                continue
            else:
                if format == 'sac' and '-12345' in str(value):
                    pass
                elif format == 'sh' and head_format == 'COMMENT':
                    st.update(json.loads(value))
                    continue
                else:
                    st[head] = value
            try:
                convert = _HEADER_CONVERSIONS[format][head][0]
                st[head] = convert(st, head)
            except KeyError:
                pass

    def _write_format_specific_header(self, format):
        st = self.stats
        format = format.lower()
        if format == 'q':
            format = 'sh'
        elif format == 'sac':
            # make sure SAC reference time is set
            from obspy.io.sac.util import obspy_to_sac_header
            self.stats.sac = obspy_to_sac_header(self.stats)
        try:
            header_map = zip(_HEADERS, _FORMATHEADERS[format])
        except KeyError:
            # file format is H5 or not supported
            return
        if format not in st:
            st[format] = AttribDict({})
        if format == 'sh':
            comment = {}
        for head, head_format in header_map:
            if format == 'sh' and head_format == 'COMMENT':
                try:
                    comment[head] = st[head]
                except KeyError:
                    pass
                continue
            try:
                val = st[head]
            except KeyError:
                continue
            try:
                convert = _HEADER_CONVERSIONS[format][head][1]
                val = convert(st, head)
            except KeyError:
                pass
            st[format][head_format] = val
        if format == 'sh' and len(comment) > 0:
            def default(obj):  # convert numpy types
                return np.asscalar(obj)
            st[format]['COMMENT'] = json.dumps(comment, separators=(',', ':'),
                                               default=default)

    def _seconds2utc(self, seconds, reftime=None):
        """Return UTCDateTime given as seconds relative to reftime"""
        from collections import Iterable
        from obspy import UTCDateTime as UTC
        if isinstance(seconds, Iterable):
            return [self._seconds2utc(s, reftime=reftime) for s in seconds]
        if isinstance(seconds, UTC) or reftime is None or seconds is None:
            return seconds
        if not isinstance(reftime, UTC):
            reftime = self.stats[reftime]
        return reftime + seconds

    def moveout(self, vmodel_file="iasp91.dat"):
        """
        Depth migration of the receiver function.
        Also calculates piercing points.
        """
        st = self.stats
        _, self.data, delta = moveout(self.data, st)
        st.pp_latitude = []
        st.pp_longitude = []
        st.pp_depth = np.linspace(0, 800, len(delta))

        # Calculate ppoint position
        for dis in delta:
            lat = st.station_latitude
            lon = st.station_longitude
            az = st.back_azimuth
            dr = st.distance*DEG2KM
            lat2, lon2 = direct_geodetic((lat, lon), az, dr)
            st.pp_latitude.append(lat2)
            st.pp_longitude.append(lon2)

    def ppoint(self, vmodel_file='iasp91.dat'):
        """
        Calculates piercing points for receiver function.
        """
        st = self.stats
        htab, _, delta = dt_table(st.slowness, vmodel_file, st.phase)
        st.pp_depth = htab
        st.pp_latitude = []
        st.pp_longitude = []

        for dis in delta:
            lat = st.station_latitude
            lon = st.station_longitude
            az = st.back_azimuth
            dr = st.distance*DEG2KM
            lat2, lon2 = direct_geodetic((lat, lon), az, dr)
            st.pp_latitude.append(lat2)
            st.pp_longitude.append(lon2)

    def write(self, filename, format, **kwargs):
        """
        Save current trace into a file  including format specific headers.
        See `Trace.write() <obspy.core.trace.Trace.write>` in ObsPy.
        """
        RFStream([self]).write(filename, format, **kwargs)


def obj2stats(event=None, station=None):
    """
    Map event and station object to stats with attributes.
    :param event: ObsPy `~obspy.core.event.event.Event` object
    :param station: station object with attributes latitude, longitude and
        elevation
    :return: ``stats`` object with station and event attributes
    """
    stats = AttribDict({})
    if event is not None:
        for key, getter in _EVENT_GETTER:
            stats[key] = getter(event)
    if station is not None:
        for key, getter in _STATION_GETTER:
            stats[key] = getter(station)
    return stats


def rfstats(info=None, starttime=None, event=None, station=None,
            tt_model="IASP91", phase=config.phase):
    """
    Creates a stats object for a RFTrace object. Provide an info dic and
    starttime or event and station object. Latter will take longer since
    computations are redone.

    INPUT:
        info: dictionary containing information about waveform.
        starttime: Starttime of first trace in stream, used as identifier in
            info dict.
        event: ObsPy `~obspy.core.event.event.Event` object
        station: dictionary like object with items latitude, longitude and
            elevation
        phase: 'S' for SRF or 'P' for PRF.

    :param tt_model: model for travel time calculation.
        (see the `obspy.taup` module, default: iasp91)
    return: `~obspy.core.trace.Stats` object with event and station
        attributes, distance, back_azimuth, onset and
        ray parameter.
    """
    stats = AttribDict({})

    # read info file if provided
    if info and starttime:
        i = info["starttime"].index(starttime)
        stats.update({'distance': info["rdelta"][i],
                      'back_azimuth': info["rbaz"][i],
                      'onset': info["onset"][i],
                      'slowness': info["rayp_s_deg"][i],
                      'phase': phase, 'event_latitude': info["evtlat"][i],
                      'event_longitude': info["evtlon"][i],
                      'event_depth': info["evt_depth"][i],
                      'event_magnitude': info["magnitude"][i],
                      'event_time': UTCDateTime(info["ot_ret"][i]),
                      'station_latitude': info["statlat"],
                      'station_longitude': info["statlon"],
                      'station_elevation': info["statel"]})
        if "evt_id" in info:
            stats.update({"event_id": info["evt_id"][i]})
    elif event is not None and station is not None:
        stats.update(obj2stats(event=event, station=station))
        dist, rbaz, _ = gps2dist_azimuth(stats.station_latitude,
                                         stats.station_longitude,
                                         stats.event_latitude,
                                         stats.event_longitude)

        # Calculate arrival parameters
        dist = dist / 1000 / DEG2KM
        tt_model = TauPyModel(model=tt_model)

        arrivals = tt_model.get_travel_times(stats.event_depth, dist, (phase,))
        if len(arrivals) == 0:
            raise Exception('TauPy does not return phase %s at distance %s' %
                            (phase, dist))
        if len(arrivals) > 1:
            msg = ('TauPy returns more than one arrival for phase %s at '
                   'distance -> take first arrival')
            warnings.warn(msg % (phase, dist))
        arrival = arrivals[0]
        onset = stats.event_time + arrival.time
        rayp = arrival.ray_param_sec_degree
        stats.update({'rdelta': dist, 'rbaz': rbaz, 'onset': onset,
                      'rayp_s_deg': rayp, 'phase': phase})
    return stats
