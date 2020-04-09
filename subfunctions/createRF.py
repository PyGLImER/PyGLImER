#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:10 2020

@author: pm

toolset to create RFs and RF classes
"""
import json
from operator import itemgetter
from pkg_resources import resource_filename
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator

from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth
from geographiclib.geodesic import Geodesic
from obspy.taup import TauPyModel

from scipy.signal.windows import hann

from subfunctions.deconvolve import it, spectraldivision, multitaper
from subfunctions.moveout_stack import DEG2KM, maxz, moveout, dt_table
import config


def createRF(st_in, phase=config.phase, shift=config.tz,
             method=config.decon_meth, trim=None, event=None, station=None,
             info=None):
    """
    Creates a receiver function with the defined method from an obspy
    stream.

    Parameters
    ----------
    st_in : '~obspy.Stream'
        Stream of which components are to be deconvolved.
    phase : string, optional
        Either "P" or "S". The default is config.phase.
    shift : float, optional
        Time shift of theoretical arrival from starttime in seconds.
        The default is config.tz.
    method : string, optional
        Deconvolution method, "waterlevel", 'dampedf' for constant
        damping level, 'it' for iterative time domain deconvoltuion, 'multit'
        for multitaper or 'fqd' for frequency dependently damped spectral
        division. The default is config.decon_meth.
    trim : tuple, optional
        taper/truncate. Given as list [a, b] in s - left,right.
        The default is None.
    event : '~obspy.core.event', optional
        Event File used to extract additional information about arrival.
        Provide either event and station or info dict. The default is None.
    station : '~obspy.core.station', optional
        Dictionary containing station information. The default is None.
    info : dict, optional
        Dictionary containing information about the waveform, used to
        extract additional information for header. The default is None.

    Raises
    ------
    Exception
        If deconvolution method is unknown.

    Returns
    -------
    RF : subfunctons.createRF.RFTrace
        RFTrace object containing receiver function.

    """

    # sampling interval
    dt = st_in[0].stats.delta

    # deep copy stream
    st = st_in.copy()
    RF = st.copy()

    # Normalise stream
    st.normalize()

    # Shorten RF stream
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
    stats.update({"type": "time"})
    RF = RFTrace(trace=RF[0])
    RF.stats.update(stats)
    return RF


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

    Parameters
    ----------
    pathname_or_url : path, optional
        Path to file. The default is None.
    format : string, optional
        e.g. "MSEED" or "SAC". Will attempt to determine from ending if = None.
        The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    subfunctions.createRF.RFStream
        RFStream object from file.

    """
    if pathname_or_url is None:   # use example file
        fname = resource_filename('rf', 'example/minimal_example.tar.gz')
        pathname_or_url = fname
        format = 'SAC'
    stream = read(pathname_or_url, format=format, **kwargs)
    stream = RFStream(stream)
    # Calculate piercing points for depth migrated RF
    for tr in stream:
        if tr.stats.type == "depth" or tr.stats.type == "stastack":
            tr.ppoint()
    return stream


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
            if tr.stats.type == 'depth' or tr.stats.type == 'stastack':
                # Lists cannot be written in header
                # Save maximum depth of pp-calculation
                tr.stats.pp_depth = tr.stats.pp_depth.max()
                tr.stats.pp_longitude = None
                tr.stats.pp_latitude = None

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
        Depth migration of all receiver functions given Stream.
        Also calculates piercing points and adds them to RFTrace.stats.

        Parameters
        ----------
        vmodel_file : file, optional
            Velocity model located in /data/vmodel.
            The default is "iasp91.dat".

        Returns
        -------
        z : np.array
            DESCRIPTION.
        RF_mo : RFTrace
            Depth migrated receiver function.

        """
        RF_mo = []
        for tr in self:
            try:
                z, mo = tr.moveout(vmodel_file=vmodel_file)
            except TypeError:
                # This trace is already depth migrated
                RF_mo.append(tr)
                continue
            RF_mo.append(mo)
        st = RFStream(traces=RF_mo)
        if z not in locals():
            z = np.linspace(0, maxz, len(st[0].data))
        return z, st

    def ppoint(self, vmodel_file='iasp91.dat'):
        """
        Calculates piercing points for all receiver functions in given
        RFStream and adds them to self.stats in form of lat, lon and depth.

        Parameters
        ----------
        vmodel_file : file, optional
            velocity model located in /data/vmodel.
            The default is 'iasp91.dat'.

        Returns
        -------
        None.
        """
        for tr in self:
            tr.ppoint(vmodel_file=vmodel_file)

    def station_stack(self, vmodel_file='iasp91.dat'):
        """
        Performs a moveout correction and stacks all receiver functions
        in Stream. Make sure that Stream only contains RF from one station!

        Parameters
        ----------
        vmodel_file : file, optional
            1D velocity model in data/vmodels. The default is 'iasp91.dat'.

        Returns
        -------
        z : np.array
            Depth Vector.
        stack : subfunctions.createRF.RFTrace
            Object containing station stack.
        RF_mo : subfunctions.createRF.RFStream
            Object containing depth migrated traces.

        """

        z, RF_mo = self.moveout(vmodel_file=vmodel_file)
        RF_mo.normalize()  # make sure traces are normalised
        traces = []
        for tr in RF_mo:
            traces.append(tr.data)
        stack = np.average(traces, axis=0)
        stack = RFTrace(data=stack, header=self[0].stats)
        stack.stats.update({"type": "stastack", "starttime": UTCDateTime(0)})
        return z, stack, RF_mo

    def plot(self, scale=2):
        """
        USE RFTrace.plot() instead

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        """
        # deep copy stream
        traces = []
        for tr in self:
            stats = AttribDict({})
            stats["coordinates"] = {}
            stats["coordinates"]["latitude"] = tr.stats["event_latitude"]
            stats["coordinates"]["longitude"] = tr.stats["event_longitude"]
            stats["network"] = tr.stats["network"]
            stats["station"] = tr.stats["station"]

            # Check type
            if tr.stats.type == "time":
                stats.delta = tr.stats.delta
                data = tr.data
                TAS = round((tr.stats.onset-tr.stats.starttime)/stats.delta)
                if tr.stats.phase == "S":
                    data = -np.flip(data[round(TAS-30/stats.delta):TAS+1])
                elif tr.stats.phase == "P":
                    data = data[TAS:round(TAS + 30/stats.delta)]
                stats["npts"] = len(data)
                trace = Trace(data=data, header=stats)
                trace.normalize()

            elif tr.stats.type == "depth":
                stats.delta = 0.25
                data = tr.data
                stats["npts"] = len(data)
                trace = Trace(data=data, header=stats)
                trace.normalize()

            elif tr.stats.type == "stastack":
                # That should be a single plot
                tr.plot()
            traces.append(trace)
        statlat = tr.stats.station_latitude
        statlon = tr.stats.station_longitude
        st = Stream(traces=traces)
        fig = st.plot(type='section', dist_degree=True,
                      ev_coord=(statlat, statlon), scale=scale, time_down=True,
                      linewidth=1.5, handle=True, fillcolors=('r', 'c'))
        fig.suptitle([tr.stats.network, tr.stats.station])
        ax = fig.get_axes()[0]
        if tr.stats.type == "depth":
            ax.set_ylabel('Depth [km]')
        return fig, ax


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

        Parameters
        ----------
        vmodel_file : file, optional
            Velocity model located in /data/vmodel.
            The default is "iasp91.dat".

        Returns
        -------
        z : np.array
            DESCRIPTION.
        RF_mo : RFTrace
            Depth migrated receiver function.

        """
        if self.stats.type == "depth" or self.stats.type == "stastack":
            raise TypeError("RF is already depth migrated.")
        st = self.stats
        z, RF_mo, delta = moveout(self.data, st)
        st.pp_latitude = []
        st.pp_longitude = []
        st.pp_depth = np.linspace(0, maxz, len(delta))

        # Calculate ppoint position
        for dis in delta:
            lat = st.station_latitude
            lon = st.station_longitude
            az = st.back_azimuth
            coords = Geodesic.WGS84.ArcDirect(lat, lon, az, dis)
            lat2, lon2 = coords['lat2'], coords['lon2']
            st.pp_latitude.append(lat2)
            st.pp_longitude.append(lon2)

        # Create trace object
        RF_mo = RFTrace(data=RF_mo, header=st)
        RF_mo.stats.update({"type": "depth", "npts": len(RF_mo.data)})
        return z, RF_mo

    def ppoint(self, vmodel_file='iasp91.dat'):
        """
        Calculates piercing points for receiver function and adds them to
        self.stats in form of lat, lon and depth.

        Parameters
        ----------
        vmodel_file : file, optional
            velocity model located in /data/vmodel.
            The default is 'iasp91.dat'.

        Returns
        -------
        None.

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
            coords = Geodesic.WGS84.ArcDirect(lat, lon, az, dis)
            lat2, lon2 = coords['lat2'], coords['lon2']
            st.pp_latitude.append(lat2)
            st.pp_longitude.append(lon2)

    def plot(self, grid=False):
        """
        Plots the Receiver function against either depth or time, depending on
        information given in stats.type.

        Parameters
        ----------
        grid : Bool, optional
            Show a grid. The default is False.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure parameter if one wishes to edit figure.
        ax : matplotlib.axes._subplots.AxesSubplot
            If one wishes to edit axes.

        """
        # plt.style.use('data/plot_data/PaperDoubleFig.mplstyle')
        # Make some style choices for plotting
        colourWheel = ['#329932', '#ff6961', 'b', '#6a3d9a', '#fb9a99',
                       '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
                       '#ffff99', '#b15928', '#67001f', '#b2182b', '#d6604d',
                       '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de',
                       '#4393c3', '#2166ac', '#053061']
        dashesStyles = [[3, 1], [1000, 1], [2, 1, 10, 1], [4, 1, 1, 1, 1, 1]]
        fig, ax = plt.subplots(1, 1)

        # Move y-axis and x-axis to centre, passing through (0,0)
        ax.spines['bottom'].set_position("zero")

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')

        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.major.formatter._useMathText = True
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.major.formatter._useMathText = True
        # ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.set_ylabel('normalised Amplitude')
        if grid:
            plt.grid(color='c', which="both", linewidth=.25)

        y = self.data

        # Determine type
        if self.stats.type == "time":

            # theoretical arrival sample in seconds (plotted as 0)
            # TAS = round((self.stats.onset-self.stats.starttime)
            #             / self.stats.delta)
            if self.stats.phase == "S":  # flip trace
                y = np.flip(y)
                y = -y  # polarity
                # TAS = -TAS
                t = np.linspace(self.stats.onset - self.stats.endtime,
                                self.stats.onset - self.stats.starttime,
                                len(self.data))
            elif self.stats.phase == "P":
                t = np.linspace(self.stats.starttime - self.stats.onset,
                                self.stats.endtime - self.stats.onset,
                                len(self.data))

            # plot
            ax.plot(t, y, color="k", linewidth=1.5)
            ax.set_xlabel('conversion time in s')
            ax.set_xlim(0, 30)
            ax.set_title("Receiver Function " + str(self.stats.network) + " " +
                         str(self.stats.station) + " " +
                         str(self.stats.event_time))

        elif self.stats.type == "stastack" or self.stats.type == "depth":
            # plot against depth
            z = np.linspace(0, maxz, len(self.data))
            # plot
            ax.plot(z, y, color="k", linewidth=1.5)

            ax.set_xlabel('Depth in km')
            ax.set_xlim(0, 250)
            if self.stats.type == "depth":
                ax.set_title("Receiver Function " + self.stats.network
                             + " " + self.stats.station + " " +
                             str(self.stats.event_time))
            elif self.stats.type == "stastack":
                ax.set_title("Receiver Function stack " +
                             str(self.stats.network) + " " +
                             str(self.stats.station))
        return fig, ax

    def write(self, filename, format, **kwargs):
        """
        Save current trace into a file  including format specific headers.
        See `Trace.write() <obspy.core.trace.Trace.write>` in ObsPy.
        """
        RFStream([self]).write(filename, format, **kwargs)


def obj2stats(event=None, station=None):
    """
    Map event and station object to stats with attributes.

    Parameters
    ----------
    event : `~obspy.core.event.event.Event`, optional
        Event File. The default is None.
    station : station object, optional
        Station file with attributes lat, lon, and elevation.
        The default is None.

    Returns
    -------
    stats : obspy.core.AttribDict
        Stats object with station and event attributes.

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

    Parameters
    ----------
    info : dict, optional
        Dictionary containing information about waveform. Can be None if
        event and station are provided. Has to be given in combination with
        starttime
    starttime : obspy.core.UTCDateTime, optional
        Starttime of first trace in stream, used as identifier in
            info dict. The default is None.
    event : `~obspy.core.event.event.Event`, optional
        Event file, provide together with station file if info=None.
        The default is None.
    station : station object, optional
        Station file with attributes lat, lon, and elevation.
        The default is None.
    tt_model : obspy.taup.TauPyModel, optional
        TauPy model to calculate arrival, only used if no info file is given.
        The default is "IASP91".
    phase : string, optional
        Either "P" or "S" for main phase. The default is config.phase.


    Returns
    -------
    stats : obspy.core.AttributeDict
        Stats file for a RFTrace object with event and station
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
