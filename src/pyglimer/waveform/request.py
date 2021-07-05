#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Contains the request class that is used to initialise the FDSN request for
the waveforms, the preprocessing of the waveforms, and the creation of
time domain receiver functions.


:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 27th April 2020 10:55:03 pm
Last Modified: Monday, 5th July 2021 09:04:23 am
'''
import os
from http.client import IncompleteRead
from datetime import datetime
from warnings import warn

from obspy import read_events, Catalog
from obspy.clients.fdsn import Client as Webclient
from obspy.core.utcdatetime import UTCDateTime
from obspy.taup import TauPyModel
# from obspy.clients.fdsn.client import FDSNException

from pyglimer.waveform.download import downloadwav
from pyglimer.waveform.preprocess import preprocess
from pyglimer import tmp


class Request(object):
    """"Initialises the FDSN request for
    the waveforms, the preprocessing of the waveforms, and the creation of
    time domain receiver functions."""

    def __init__(
        self, phase: str, rot: str, evtloc: str, statloc: str, rawloc: str,
        preproloc: str, rfloc: str, deconmeth: str, starttime: UTCDateTime,
        endtime: UTCDateTime, pol: str = 'v',
        minmag: float or int = 5.5, event_coords: tuple = None,
        network: str = None, station: str = None,
        waveform_client: list = None, re_client=['IRIS'],
            evtcat: Catalog = None, debug=False):
        """
        Create object that is used to start the receiver function
        workflow.

        :param phase: Arrival phase that is to be used as source phase.
            "S" to create S-Sp receiver functions and "P" for P-Ps receiver
            functions, "SKS" or "ScS" are allowed as well.
        :type phase: str
        :param rot: The coordinate system in that the seismogram should be
            rotated
            prior to deconvolution. Options are "RTZ" for radial, transverse,
            vertical; "LQT" for an orthogonal coordinate system computed by
            minimising primary energy on the
            converted component, or "PSS" for a rotation along the polarisation
            directions using the Litho1.0 surface wave tomography model.
        :type rot: str
        :param evtloc: Directory, in which to store the event catalogue (xml).
        :type evtloc: str
        :param statloc: Directory, in which to store the station inventories
                        (xml).
        :type statloc: str
        :param rawloc: Directory, in which to store the raw waveform data.
        :type rawloc: str
        :param preproloc: Directory, in which to store
            the preprocessed waveform data (mseed).
        :type preproloc: str
        :param rfloc: Directory, in which to store the receiver functions in
            time domain (sac).
        :type rfloc: str
        :param deconmeth: The deconvolution method to use for the RF creation.
            Possible options are:
            'it': iterative time domain deconvolution (Ligorria & Ammon, 1999)
            'dampedf': damped frequency deconvolution
            'fqd': frequency dependent damping - not a good choice for SRF
            'waterlevel': Langston (1977)
            'multit': for multitaper (Helffrich, 2006)
            False/None: don't create RFs
        :type deconmeth: str
        :param starttime: Earliest event date to be considered.
        :type starttime: ~obspy.UTCDateTime
        :param endtime: Latest event date to be considered.
        :type endtime: ~obspy.UTCDateTime
        :param pol: Polarisation to use as source wavelet. Either "v" for
            vertically polarised or 'h' for horizontally polarised S-waves.
            Will be ignored if phase='S', by default 'v'.
        :type pol: str, optional
        :param minmag: Minimum magnitude, by default 5.5
        :type minmag: float, optional
        :param event_coords: In case you wish to constrain events to certain
            origns. Given in the form (minlat, maxlat, minlon, maxlon),
            by default None.
        :type event_coords: Tuple, optional
        :param network: Limit the dowloand and preprocessing to a certain
            network or several networks (if type==list).
            Wildcards are allowed, by default None., defaults to None
        :type network: str, optional
        :param station: Limit the download and preprocessing to a certain
            station or several stations. Use only if network!=None.
            Wildcards are allowed, by default None.
        :type station: str, optional
        :param waveform_client: List of FDSN compatible servers to download
            waveforms from.
            See obspy documentation for obspy.Client for allowed acronyms.
            A list of servers by region can be found at
            `<https://www.fdsn.org/webservices/datacenters/>`_. None means
            that all known servers are requested, defaults to None.
        :type waveform_client: list, optional
        :param re_client: Only relevant, when debug=True. List of servers that
            will be used if data is missing and the script will attempt a
            redownload, usually it's easier to just run a request several
            times. Same logic as for waveform_client applies,
            defaults to ['IRIS']
        :type re_client: list, optional
        :param evtcat: In case you want to use an already existing event
            catalogue
            in evtloc. If None a new catalogue will be downloaded (with the
            parameters defined before), by default None, defaults to None
        :type evtcat: str, optional
        :param debug: If True, all loggers will go to DEBUG mode and all
            warnings
            will be shown. That will result in a lot of information being
            shown! Also joblib will fall back to using only few cores,
            by default False.
        :type debug: bool, optional
        :raises NameError: For invalid phases.
        """

        # Allocate variables in self
        self.debug = debug
        tmp.re_client = re_client

        # Set velocity model
        self.model = TauPyModel('iasp91')

        self.phase = phase[:-1] + phase[-1].upper()
        self.pol = pol.lower()
        self.rot = rot.upper()
        self.deconmeth = deconmeth

        # Directories
        self.logdir = os.path.join(
            os.path.dirname(os.path.abspath(statloc)), 'logs')
        os.makedirs(self.logdir, exist_ok=True)
        self.evtloc = evtloc
        self.statloc = statloc
        self.rawloc = os.path.join(rawloc, self.phase)
        self.preproloc = os.path.join(preproloc, self.phase)
        self.rfloc = os.path.join(rfloc, self.phase)

        # minimum magnitude
        self.minmag = minmag

        # Request time window
        self.starttime = starttime
        self.endtime = endtime

        # geographical constraints
        if event_coords:
            (self.eMINLAT, self.eMAXLAT,
             self.eMINLON, self.eMAXLON) = event_coords
        else:
            (self.eMINLAT, self.eMAXLAT,
             self.eMINLON, self.eMAXLON) = None, None, None, None

        # Set event depth and min/max epicentral distances
        # according to phase (see Wilson et. al., 2006)
        # and time window before (tz) and after (ta) first arrival
        self.ta = 120
        if self.phase == 'P':
            self.maxdepth = None
            self.min_epid = 28.1
            self.max_epid = 95.8
            self.tz = 30
        elif self.phase == 'S':
            self.maxdepth = 300
            self.min_epid = 55
            self.max_epid = 80
            self.tz = 120
        # (see Yuan et al. 2006)
        elif self.phase.upper() == 'SCS':
            self.maxdepth = 300
            self.min_epid = 50
            self.max_epid = 75
            self.tz = 120
        elif self.phase.upper() == 'SKS':
            # (see Zhang et. al. (2014))
            self.maxdepth = 300
            self.min_epid = 90
            self.max_epid = 120
            self.tz = 120
        else:
            raise NameError('The phase', self.phase, """is not valid or not
                            implemented yet.""")

        # network and station filters
        self.network = network
        self.station = station

        self.waveform_client = waveform_client
        self.re_client = re_client

        # Download or process available data?
        if evtcat:
            self.evtcat = read_events(os.path.join(self.evtloc, evtcat))
        else:
            self.download_eventcat()

    def download_eventcat(self):
        """
        Download the event catalogue from IRIS DMC.
        """
        # Server settings
        # 2021/02/16 Events only from IRIS as the USGS webserice tends to be
        # unstable and mixing different services will lead to a messed db
        self.webclient = Webclient('IRIS')
        event_cat_done = False

        while not event_cat_done:
            try:
                # 05.07.2021: Shorten length of a to one year, which is a lot
                # more robust
                # :NOTE: perhaps it would be smart to save each year as a file?
                # BUt then again, they have different requirements...
                # Check length of request and split if longer than a year.
                a = 365.25*24*3600  # one yr in seconds
                if self.endtime-self.starttime > a:
                    # Request is too big, break it down ito several requests

                    starttimes = [self.starttime, self.starttime+a]
                    while self.endtime-starttimes[-1] > a:
                        starttimes.append(starttimes[-1]+a)
                    endtimes = []
                    endtimes.extend(starttimes[1:])
                    endtimes.append(self.endtime)

                    # Query
                    self.evtcat = Catalog()
                    for st, et in zip(starttimes, endtimes):
                        self.evtcat.extend(
                            self.webclient.get_events(
                                starttime=st, endtime=et,
                                minlatitude=self.eMINLAT,
                                maxlatitude=self.eMAXLAT,
                                minlongitude=self.eMINLON,
                                maxlongitude=self.eMAXLON,
                                minmagnitude=self.minmag,
                                maxmagnitude=10, maxdepth=self.maxdepth))
                    event_cat_done = True

                else:
                    self.evtcat = self.webclient.get_events(
                        starttime=self.starttime, endtime=self.endtime,
                        minlatitude=self.eMINLAT, maxlatitude=self.eMAXLAT,
                        minlongitude=self.eMINLON, maxlongitude=self.eMAXLON,
                        minmagnitude=self.minmag, maxmagnitude=10,
                        maxdepth=self.maxdepth)

                    event_cat_done = True

            except IncompleteRead:
                # Server interrupted connection, just try again
                # This usually happens with enormeous requests, we should
                # reduce a
                msg = "Server interrupted connection, restarting download..."
                warn(msg, UserWarning)
                print(msg)
                continue

        os.makedirs(self.evtloc, exist_ok=True)
        # check if there is a better format for event catalog
        self.evtcat.write(
            os.path.join(
                self.evtloc,
                datetime.now().strftime("%Y%m%dT%H%M%S")), format="QUAKEML")

    def download_waveforms(self, verbose: bool = False):
        """
        Start the download of waveforms and response files.

        Parameters
        ----------
        verbose : Bool, optional
            Set True if you wish to log the output of the obspy MassDownloader.
        """
        downloadwav(
            self.phase, self.min_epid, self.max_epid, self.model, self.evtcat,
            self.tz, self.ta, self.statloc, self.rawloc, self.waveform_client,
            network=self.network, station=self.station, logdir=self.logdir,
            debug=self.debug, verbose=verbose, saveasdf=False)

    def preprocess(self, hc_filt: float or int or None = None):
        """
        Preprocess an existing database. With parameters defined in self.

        Parameters
        ----------
        hc_filt : float or int or None, optional
            Highcut frequency to filter with right before deconvolution.
            Recommended if time domain deconvolution is used. For spectral
            division, filtering can still be done after deconvolution (i.e.
            set in :func:`~pyglimer.ccp.ccp.CCPStack.compute_stack()`).
            Value for PRFs should usually be lower than 2 Hz and for SRFs lower
            than .4 Hz, by default None.
        """
        preprocess(
            self.phase, self.rot, self.pol, 0.05, self.evtcat, self.model,
            'hann', self.tz, self.ta, self.statloc, self.rawloc,
            self.preproloc, self.rfloc, self.deconmeth, hc_filt,
            netrestr=self.network, statrestr=self.station, logdir=self.logdir,
            debug=self.debug)
