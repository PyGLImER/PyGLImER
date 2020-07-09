#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A toolset to create RFs and RF classes

Created on Wed Feb 19 13:41:10 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Some parts of this code are modified versions of the rf module by
Tom Eulenfeld.

Last updated:
"""
import json
import logging
from operator import itemgetter
from pkg_resources import resource_filename
import warnings

from geographiclib.geodesic import Geodesic
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from scipy.signal.windows import hann

from .deconvolve import it, spectraldivision, multitaper, gen_it
from .moveout import DEG2KM, maxz, res, moveout, dt_table, dt_table_3D
from examples.utils.plot_utils import plot_section, plot_single_rf

logger = logging.Logger("rf")


def createRF(st_in, phase, pol='v', onset=None,
             method='it', trim=None, event=None, station=None,
             info=None):
    """
    Creates a receiver function with the defined method from an obspy
    stream.

    :param st_in: Stream of which components are to be deconvolved.
    :type st_in: ~obspy.Stream
    :param phase: Either "P" or "S".
    :type phase: str
    :param pol: Polarisation to be deconvolved from. Only for phase = P,
                defaults to 'v'
    :type pol: str, optional
    :param onset: Is used to shift function rather than shift if provided.
        If info is provided, it will be extracted from info file,
        defaults to None
    :type onset: ~obspy.UTCDateTime, optional
    :param method: Deconvolution method, "waterlevel", 'dampedf' for constant
        damping level, 'it' for iterative time domain deconvoltuion, 'multit'
        for multitaper or 'fqd' for frequency dependently damped spectral
        division. The default is iterative time domain ('it').
    :type method: str, optional
    :param trim: taper/truncate. Given as tuple (a, b) in s - left,right.
        The default is None.
    :type trim: tuple, optional
    :param event: Event File used to extract additional information about
        arrival. Provide either event and station or info dict. The default is
        None.
    :type event: ~obspy.core.event, optional
    :param station: Dictionary containing station information,
        defaults to None
    :type station: '~obspy.core.station', optional
    :param info: Dictionary containing information about the waveform, used to
        extract additional information for header. The default is None.
    :type info: dict, optional
    :raises Exception: If deconvolution method is unknown.
    :raises Exception: For wrong trim input.
    :return: RFTrace object containing receiver function.
    :rtype: :class:`~pyglimer.create.RFTrace`
    """

    pol = pol.lower()
    
    if info:
        ii = info['starttime'].index(st_in[0].stats.starttime)
        shift = info['onset'][ii] - st_in[0].stats.starttime

    elif onset:
        shift = onset - st_in[0].stats.starttime
    else:
        raise ValueError('Provide either info dict as input or give data\
            manually (see docstring).')

    # sampling interval
    dt = st_in[0].stats.delta

    # deep copy stream
    st = st_in.copy()
    RF = st.copy()

    # Normalise stream
    # Don't! Normalisation only for spectraldivision
    #st.normalize()

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
        tap = hann(round(15 / dt))
        taper = np.ones(st[0].stats.npts)
        taper[:int((trim[0] - 7.5) / dt)] = float(0)
        taper[-int((trim[1] - 7.5) / dt):] = float(0)
        taper[int((trim[0] - 7.5) / dt):int(trim[0] / dt)] = \
            tap[:round(7.5 / dt)]
        taper[-int(trim[1] / dt):-int((trim[1] - 7.5) / dt)] = \
            tap[-round(7.5 / dt):]
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
        if pol == 'v':
            u = stream["R"]
        elif pol == 'h':
            u = stream["T"]
    elif phase == "P" and "Q" in stream:
        v = stream["L"]
        if pol == 'v':
            u = stream["Q"]
        elif pol == 'h':
            u = stream['T']
    elif phase == "P" and "V" in stream:
        v = stream["P"]
        if pol == 'v':
            u = stream["V"]
        elif pol == 'h':
            u = stream["H"]
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
    else:
        raise ValueError('Strange Input. Is you stream in a valid coordinate\
            system? Is your phase accepted?\
            For example, PyGLImER does not allow for deconvolution of\
                a stream in NEZ.')

    # Deconvolution
    if method == "it":
        if phase == "S":
            width = 1.5  # change for Kind (2015) to 1
        elif phase == "P":
            width = 2.5
        else:
            raise ValueError('Phase '+phase+' is not supported.')
        lrf = None
        RF[0].data = it(v, u, dt, shift=shift, width=width)[0]
    elif method == "dampedf":
        RF[0].data, lrf = spectraldivision(v, u, dt, shift, "con", phase=phase)
    elif method == "waterlevel":
        RF[0].data, lrf = spectraldivision(v, u, dt, shift, "wat", phase=phase)
    elif method == 'fqd':
        RF[0].data, lrf = spectraldivision(v, u, dt, shift, "fqd", phase=phase)
    elif method == 'multit':
        RF[0].data, lrf, _, _ = multitaper(v, u, dt, shift, "fqd")
        # remove noise caused by multitaper
        RF.filter('lowpass', freq=2.50, zerophase=True, corners=2)
    else:
        raise ValueError(method+ " is no valid deconvolution method.")
    if lrf is not None:
        # Normalisation for spectral division and multitaper
        # In order to do that, we have to find the factor that is necassary to
        # bring the zero-time pulse to 1
        fact = lrf[round(shift/dt)]
        RF[0].data = RF[0].data/fact
        # I could probably create another QC here and check if fact is
        # the maximum of RF[0].data or even close to the maximum. Let's try:
        if abs(fact) < abs(lrf).max()/2:
            raise ValueError('The noise level of the created receiver funciton\
                is too high.')
        

    # create RFTrace object
    # create stats
    stats = rfstats(phase, info=info, starttime=st[0].stats.starttime,
                    event=event, station=station)
    stats.update({"type": "time"})
    RF = RFTrace(trace=RF[0])
    RF.stats.update(stats)
    return RF


def read_by_station(network: str, station: str, phase: str, rfdir: str):
    """
    Convenience method to read all available receiver functions of one station
    and a particular phase into one single stream. Subsequently, one could
    for instance do a station stack by using
    :func:`~pyglimer.rf.create.RFStream.station_stack()`.

    Parameters
    ----------
    network : str
        Network shortcut, two digits, e.g. "IU".
    station : str
        Station shortcut, three digits, e.g. "HRV"
    phase : str
        Primary phase ("S" or "P")
    rfdir : str
        Parental directory, in that the RF database is located

    Returns
    -------
    :class:`~pyglimer.rf.create.RFStream`
        RFStream object containing all receiver function of station x.
    """
    files = os.listdir(os.path.join(rfdir, phase, network, station))
    rflist = []
    for f in files:
        infile = os.path.join(rfdir, phase, network, f)
        # Append RFTrace
        rflist.append(read_rf(infile)[0])
    # Create RFStream object
    rfst = RFStream(traces=rflist)
    return rfst


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


def read_rf(pathname_or_url, format=None, **kwargs):
    """
    Read waveform files into RFStream object.
    See :func:`~obspy.core.stream.read` in ObsPy.

    :param pathname_or_url: Path to file.
    :type pathname_or_url: str
    :param format: str, defaults to None
    :type format: e.g. "MSEED" or "SAC". Will attempt to determine from ending
        if = None. The default is None.
    :return: RFStream object from file.
    :rtype: :class:`~pyglimer.createRF.RFStream`
    """    

    stream = read(pathname_or_url, format=format, **kwargs)
    stream = RFStream(stream)
    # Calculate piercing points for depth migrated RF
    for tr in stream:
        if tr.stats.type == "depth":
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
            if tr.stats.type == 'depth':
                # Lists cannot be written in header
                tr.stats.pp_depth = None
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

    def moveout(self, vmodel, latb=None, lonb=None, taper=True):
        """
        Depth migration of all receiver functions given Stream.
        Also calculates piercing points and adds them to RFTrace.stats.

        :param vmodel: Velocity model located in /data/vmodel.
            Standard options are iasp91.dat and 3D (GYPSuM).
        :type vmodel: str
        :param latb: Tuple in Form (minlat, maxlat). To save RAM on 3D
            raytraycing. Will remain unused for 1D RT, defaults to None
        :type latb: Tuple, optional
        :param lonb: Tuple in Form (minlon, maxlon), defaults to None
        :type lonb: Tuple, optional
        :param taper: If True, the last 10km of the RF will be tapered, which
            avoids jumps in station stacks. Should be False for CCP stacks,
            defaults to True.
        :type taper: bool, optional
        :return: 1D np.ndarray containing depths and an
             object of type depth.
        :rtype: 1D np.ndarray, :class:`~pyglimer.rf.create.RFStream`
        """

        RF_mo = []
        for tr in self:
            try:
                z, mo = tr.moveout(vmodel, latb=latb, lonb=lonb, taper=taper)
            except TypeError:
                # This trace is already depth migrated
                RF_mo.append(tr)
                continue
            RF_mo.append(mo)
        st = RFStream(traces=RF_mo)
        if 'z' not in locals():
            z = np.linspace(0, maxz, len(st[0].data))
        return z, st

    def ppoint(self, vmodel_file='iasp91.dat', latb=None, lonb=None):
        """
        Calculates piercing points for all receiver functions in given
        RFStream and adds them to self.stats in form of lat, lon and depth.

        :param vmodel_file: velocity model located in /data/vmodel.
            The default is 'iasp91.dat'.
        :type vmodel_file: str, optional
        :param latb: Tuple in Form (minlat, maxlat). To save RAM on 3D
            raytraycing. Will remain unused for 1D RT, defaults to None.
        :type latb: tuple, optional
        :param lonb: Tuple in Form (minlon, maxlon), defaults to None
        :type lonb: tuple, optional.
        """        

        for tr in self:
            tr.ppoint(vmodel=vmodel_file, latb=latb, lonb=lonb)

    def station_stack(self, vmodel_file='iasp91.dat'):
        """
        Performs a moveout correction and stacks all receiver functions
        in Stream. Make sure that Stream only contains RF from one station!

        :param vmodel_file: 1D velocity model in data/vmodels.
            The default is 'iasp91.dat'.
        :type vmodel_file: str, optional
        :return: z : Depth Vector.
            stack : Object containing station stack.
            RF_mo : Object containing depth migrated traces.
        :rtype: z : 1D np.ndarray
            stack : :class:`~pyglimer.rf.create.RFTrace`
            RF_mo : :class:`~pyglimer.rf.create.RFStream`
        """        

        latb = (self.station_latitude-10, self.station_latitude+10)
        lonb = (self.statio_longitude-20, self.station_longitude+20)

        z, RF_mo = self.moveout(vmodel=vmodel_file, latb=latb, lonb=lonb)
        # RF_mo.normalize()  # make sure traces are normalised
        traces = []
        for tr in RF_mo:
            traces.append(tr.data)
        stack = np.average(traces, axis=0)
        stack = RFTrace(data=stack, header=self[0].stats)
        stack.stats.update({"type": "stastack", "starttime": UTCDateTime(0),
                            "pp_depth": None, "pp_latitude": None,
                            "pp_longitude": None})
        return z, stack, RF_mo

    def plot(self, channel = "PRF",
            lim: list or tuple or None = None, 
            epilimits: list or tuple or None = None,
            scalingfactor: float = 2.0, ax: plt.Axes = None,
            line: bool = True,
            linewidth: float = 0.25, outputdir: str or None = None, 
            title: str or None = None, show: bool = True):
        """Creates plot of a receiver function section as a function
        of epicentral distance or single plot if len(RFStream)==1.
        
        Parameters
        ----------
        lim : list or tuple or None
            y axis time limits in seconds (if self.stats.type==time)
            or depth in km (if self.stats==depth) (len(list)==2).
            If `None` full traces is plotted.
            Default None.
        epilimits : list or tuple or None = None,
            y axis time limits in seconds (len(list)==2).
            If `None` from 30 to 90 degrees plotted.
            Default None.
        scalingfactor : float
            sets the scale for the traces. Could be automated in 
            future functions(Something like mean distance between
            traces)
            Defaults to 2.0
        line : bool
            plots black line of the actual RF
            Defaults to True
        linewidth: float
            sets linewidth of individual traces
        ax : `matplotlib.pyplot.Axes`, optional
            Can define an axes to plot the RF into. Defaults to None.
            If None, new figure is created.
        outputdir : str, optional
            If set, saves a pdf of the plot to the directory.
            If None, plot will be shown instantly. Defaults to None.
        clean: bool
            If True, clears out all axes and plots RF only.
            Defaults to False.
        
        Returns
        -------
        ax : `matplotlib.pyplot.Axes`
        
        """     
        if self.count() == 1:
            # Do single plot
            ax = plot_single_rf(
                self[0], tlim=lim, ax=ax, outputdir=outputdir)
        else:
            ax = plot_section(
                self, timelimits=lim, epilimits=epilimits,
                scalingfactor=scalingfactor, line=line, linewidth=linewidth,
                ax=ax, outputdir=outputdir)
        return ax
        # deep copy stream
        # if len(self.traces) == 1:
        #     fig, ax = self[0].plot()
        #     return fig, ax

        # traces = []
        # for tr in self:
        #     stats = AttribDict({})
        #     stats["coordinates"] = {}
        #     stats["coordinates"]["latitude"] = tr.stats["event_latitude"]
        #     stats["coordinates"]["longitude"] = tr.stats["event_longitude"]
        #     stats["network"] = tr.stats["network"]
        #     stats["station"] = tr.stats["station"]

        #     # Check type
        #     if tr.stats.type == "time":
        #         stats.delta = tr.stats.delta
        #         data = tr.data
        #         TAS = round((
        #             tr.stats.onset - tr.stats.starttime) / stats.delta)
        #         TAS = 1201
        #         # print(TAS)
        #         if tr.stats.phase == "S":
        #             data = -np.flip(data)

        #         data = data[TAS:round(TAS + 30 / stats.delta)]
        #         stats["npts"] = len(data)
        #         trace = Trace(data=data, header=stats)
        #         #trace.normalize()

        #     elif tr.stats.type == "depth":
        #         stats.delta = res
        #         data = tr.data
        #         stats["npts"] = len(data)
        #         trace = Trace(data=data, header=stats)
        #         #trace.normalize()

        #     elif tr.stats.type == "stastack":
        #         # That should be a single plot
        #         tr.plot()
        #     traces.append(trace)
        # statlat = tr.stats.station_latitude
        # statlon = tr.stats.station_longitude
        # st = Stream(traces=traces)
        # fig = st.plot(type='section', dist_degree=True,
        #               ev_coord=(statlat, statlon), scale=scale, time_down=True,
        #               linewidth=1.5, handle=True, fillcolors=('r', 'c'))
        # fig.suptitle([tr.stats.network, tr.stats.station])
        # ax = fig.get_axes()[0]
        # # ax.set_ylim(0, 30)
        # if tr.stats.type == "depth":
        #     ax.set_ylabel('Depth [km]')
        # return fig, ax


class RFTrace(Trace):
    """
    Class providing the Trace object for receiver function calculation.
    
    This object is a modified version of Tom Eulenfeld's rf project's RFStream
    class. License below.
    
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

    def __init__(self, data=None, header=None, trace=None):
        if header is None:
            header = {}
        if trace is not None:
            data = trace.data
            header = trace.stats
        super(RFTrace, self).__init__(data=data, header=header)
        st = self.stats
        if ('_format' in st and st._format.upper() == 'Q' and
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
        if 'back_azimuth' in self.stats:
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

    def moveout(self, vmodel, latb=None, lonb=None, taper=True):
        """
        Depth migration of the receiver function.
        Also calculates piercing points and adds them to RFTrace.stats.

        :param vmodel: Velocity model located in /data/vmodel.
            Standard options are iasp91.dat and 3D (GYPSuM).
        :type vmodel: str
        :param latb: Tuple in Form (minlat, maxlat). To save RAM on 3D
            raytraycing. Will remain unused for 1D RT, defaults to None
        :type latb: Tuple, optional
        :param lonb: Tuple in Form (minlon, maxlon), defaults to None
        :type lonb: Tuple, optional
        :param taper: If True, the last 10km of the RF will be tapered, which
            avoids jumps in station stacks. Should be False for CCP stacks,
            defaults to True.
        :type taper: bool, optional
        :return: 1D np.ndarray containing depths and an
            RFTrace object of type depth.
        :rtype: 1D np.ndarray, :class:`~pyglimer.rf.create.RFTrace`
        """   

        if self.stats.type == "depth" or self.stats.type == "stastack":
            raise TypeError("RF is already depth migrated.")
        st = self.stats

        z, RF_mo, delta = moveout(self.data, st, vmodel, latb=latb, lonb=lonb,
                                  taper=taper)
        st.pp_latitude = []
        st.pp_longitude = []

        if st.station_elevation > 0:
            st.pp_depth = np.hstack(
                (np.arange(-round(st.station_elevation/1000, 1), 0, .1),
                 np.arange(0, maxz + res)))[:len(delta)]
        else:
            st.pp_depth = np.arange(-round(st.station_elevation/1000),
                                    maxz + res, res)[0:len(delta)]
        # np.arange(0, maxz + res, res)[0:len(delta)]

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

    def ppoint(self, vmodel, latb=None, lonb=None):
        """
        Calculates piercing points for receiver function in and adds them
        to self.stats in form of lat, lon and depth.

        :param vmodel_file: velocity model located in /data/vmodel.
            The default is 'iasp91.dat'.
        :type vmodel_file: str, optional
        :param latb: Tuple in Form (minlat, maxlat). To save RAM on 3D
            raytraycing. Will remain unused for 1D RT, defaults to None.
        :type latb: tuple, optional
        :param lonb: Tuple in Form (minlon, maxlon), defaults to None
        :type lonb: tuple, optional.
        """

        st = self.stats

        if vmodel == '3D':
            htab, _, delta = dt_table_3D(
                st.slowness, st.phase, st.station_latitude,
                st.station_longitude, st.back_azimuth, st.station_elevation,
                latb, lonb)

        else:
            htab, _, delta = dt_table(
                st.slowness, vmodel, st.phase, st.station_elevation)

        st.pp_depth = np.hstack((np.arange(-10, 0, .1), np.arange(0, maxz+res, res)))
        # if st.station_elevation > 0:
        #     st.pp_depth = np.hstack(
        #         (np.arange(-round(st.station_elevation/1000, 1), 0, .1),
        #          np.arange(0, maxz + res)))[:len(delta)]
        # else:
        #     st.pp_depth = np.arange(-round(st.station_elevation/1000),
        #                             maxz + res, res)[0:len(delta)]
        # st.pp_depth = np.hstack(
        #     (np.arange(-10, 0, .1), np.arange(0, htab.max(), res)))
        delta2 = np.empty(st.pp_depth.shape)
        delta2.fill(np.nan)

        # Find first pp depth
        # starti = np.nonzero(np.isclose(st.ppdepth, -st.station_elevation))
        starti = np.nonzero(np.isclose(st.pp_depth, htab[0]))[0][0]

        # print(len(delta))
        # Create full-length distance vector
        delta2[starti:starti+len(delta)] = delta

        st.pp_latitude = []
        st.pp_longitude = []

        for dis in delta2:
            lat = st.station_latitude
            lon = st.station_longitude
            az = st.back_azimuth
            coords = Geodesic.WGS84.ArcDirect(lat, lon, az, dis)
            lat2, lon2 = coords['lat2'], coords['lon2']
            st.pp_latitude.append(lat2)
            st.pp_longitude.append(lon2)
        return delta2

    def plot(self, lim: list or tuple or None = None, 
                   ax: plt.Axes = None, outputdir: str = None, 
                   clean: bool = False):
        """Creates plot of a single receiver function
        
        Parameters
        ----------
        lim: list or tuple or None
            x axis time limits in seconds or km (depth) (len(list)==2).
            If `None` full trace is plotted.
            Default None.
        ax : `matplotlib.pyplot.Axes`, optional
            Can define an axes to plot the RF into. Defaults to None.
            If None, new figure is created.
        outputdir : str, optional
            If set, saves a pdf of the plot to the directory.
            If None, plot will be shown instantly. Defaults to None.
        clean: bool
            If True, clears out all axes and plots RF only.
            Defaults to False.
        
        Returns
        -------
        ax : `matplotlib.pyplot.Axes`
        """
        ax = plot_single_rf(
            self, lim, ax=ax, outputdir=outputdir, clean=clean)
        return ax

        # # plt.style.use('data/plot_data/PaperDoubleFig.mplstyle')
        # # Make some style choices for plotting
        # colourWheel = ['#329932', '#ff6961', 'b', '#6a3d9a', '#fb9a99',
        #                '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
        #                '#ffff99', '#b15928', '#67001f', '#b2182b', '#d6604d',
        #                '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de',
        #                '#4393c3', '#2166ac', '#053061']
        # dashesStyles = [[3, 1], [1000, 1], [2, 1, 10, 1], [4, 1, 1, 1, 1, 1]]
        # fig, ax = plt.subplots(1, 1)

        # # Move y-axis and x-axis to centre, passing through (0,0)
        # ax.spines['bottom'].set_position("zero")

        # # Eliminate upper and right axes
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')

        # # Show ticks in the left and lower axes only
        # ax.xaxis.set_ticks_position('bottom')

        # ax.xaxis.set_major_formatter(ScalarFormatter())
        # ax.yaxis.major.formatter._useMathText = True
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        # ax.xaxis.major.formatter._useMathText = True
        # # ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # ax.set_ylabel('normalised Amplitude')
        # if grid:
        #     plt.grid(color='c', which="both", linewidth=.25)

        # y = self.data

        # # Determine type
        # if self.stats.type == "time":

        #     # theoretical arrival sample in seconds (plotted as 0)
        #     # TAS = round((self.stats.onset-self.stats.starttime)
        #     #             / self.stats.delta)
        #     if self.stats.phase == "S":  # flip trace
        #         y = np.flip(y)
        #         y = -y  # polarity
        #         # TAS = -TAS
        #         t = np.linspace(self.stats.onset - self.stats.endtime,
        #                         self.stats.onset - self.stats.starttime,
        #                         len(self.data))
        #     elif self.stats.phase == "P":
        #         t = np.linspace(self.stats.starttime - self.stats.onset,
        #                         self.stats.endtime - self.stats.onset,
        #                         len(self.data))

        #     # plot
        #     ax.plot(t, y, color="k", linewidth=1.5)
        #     ax.set_xlabel('conversion time in s')
        #     ax.set_xlim(-10, 30)
        #     ax.set_title("Receiver Function " + str(self.stats.network) + " " +
        #                  str(self.stats.station) + " " +
        #                  str(self.stats.event_time))

        # elif self.stats.type == "stastack" or self.stats.type == "depth":
        #     # plot against depth
        #     # z = np.linspace(0, maxz, len(self.data))
        #     z = np.hstack(
        #         ((np.arange(-10, 0, .1)), np.arange(0, maxz+res, res)))

        #     # plot
        #     ax.plot(z, y, color="k", linewidth=1.5)

        #     ax.set_xlabel('Depth in km')
        #     ax.set_xlim(0, 250)
        #     if self.stats.type == "depth":
        #         ax.set_title("Receiver Function " + self.stats.network
        #                      + " " + self.stats.station + " " +
        #                      str(self.stats.event_time))
        #     elif self.stats.type == "stastack":
        #         ax.set_title("Receiver Function stack " +
        #                      str(self.stats.network) + " " +
        #                      str(self.stats.station))
        # return fig, ax

    def write(self, filename, format, **kwargs):
        """
        Save current trace into a file  including format specific headers.
        See `Trace.write() <obspy.core.trace.Trace.write>` in ObsPy.
        """
        RFStream([self]).write(filename, format, **kwargs)


def obj2stats(event=None, station=None):
    """
    Map event and station object to stats with attributes.

    :param event: Event File. The default is None.
    :type event: :class:`~obspy.core.event.event.Event`, optional
    :param station: Station file with attributes lat, lon, and elevation.
        The default is None.
    :type station: :class:`~obspy.core.Station`, optional
    :return: Stats object with station and event attributes.
    :rtype: :class:`~obspy.core.AttribDict`
    """    

    stats = AttribDict({})
    if event is not None:
        for key, getter in _EVENT_GETTER:
            stats[key] = getter(event)
    if station is not None:
        for key, getter in _STATION_GETTER:
            stats[key] = getter(station)
    return stats


def rfstats(phase, info=None, starttime=None, event=None, station=None,
            tt_model="IASP91"):
    """
    Creates a stats object for a RFTrace object. Provide an info dic and
    starttime or event and station object. Latter will take longer since
    computations are redone.

    :param phase: Either "P" or "S" for main phase.
    :type phase: str
    :param info: Dictionary containing information about waveform. Can be None
        if event and station are provided. Has to be given in combination with
        starttime, defaults to None.
    :type info: dict
    :param starttime: Starttime of first trace in stream, used as identifier in
            info dict. The default is None.
    :type starttime: :class:`~obspy.core.UTCDateTime`, optional
    :param event: Event file, provide together with station file if info=None.
        The default is None.
    :type event: :class:`~obspy.core.event.event.Event`, optional
    :param station: Station inventory with attributes lat, lon, and elevation,
        defaults to None
    :type station: :class:`~obspy.core.Station`, optional
    :param tt_model: TauPy model to calculate arrival, only used if no info
        file is given. The default is "IASP91".
    :type tt_model: str, optional
    :raises Exception: For unavailable phases
    :return: Stats file for a RFTrace object with event and station
        attributes, distance, back_azimuth, onset and
        ray parameter.
    :rtype: :class:`~obspy.core.AttribDict`
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
