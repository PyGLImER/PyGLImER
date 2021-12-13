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
Last Modified: Monday, 13th December 2021 05:10:41 pm
'''
import os
from http.client import IncompleteRead
from datetime import datetime
import logging
import warnings
import time

from obspy import read_events, Catalog
from obspy.clients.fdsn import Client as Webclient
from obspy.core.utcdatetime import UTCDateTime
from obspy.taup import TauPyModel
from tqdm import tqdm

from pyglimer.constants import onsetP, onsetS
from pyglimer.waveform.download import download_small_db, downloadwav
from pyglimer.waveform.preprocess import preprocess
from pyglimer.utils import utils as pu


class Request(object):
    """"Initialises the FDSN request for
    the waveforms, the preprocessing of the waveforms, and the creation of
    time domain receiver functions."""

    def __init__(
        self, proj_dir: str, raw_subdir: str, prepro_subdir: str,
        rf_subdir: str, statloc_subdir: str, evt_subdir: str, log_subdir: str,
        phase: str, rot: str, deconmeth: str, starttime: UTCDateTime or str,
        endtime: UTCDateTime or str, pol: str = 'v',
        minmag: float or int = 5.5, event_coords: tuple = None,
        network: str = None, station: str = None,
        waveform_client: list = None, evtcat: str = None,
        continue_download: bool = False, loglvl: int = logging.WARNING,
            format: str = 'hdf5'):
        """
        Create object that is used to start the receiver function
        workflow.

        :param proj_dir: parental directory that all project files
            will be saved in (as subfolders).
        :type proj_dir: str
        :param raw_subdir: Directory, in which to store the raw waveform data.
        :type raw_subdir: str
        :param prepro_subdir: Directory, in which to store the preprocessed
            waveform data (mseed). **Irrelevant if format is hdf5**.
        :type prepro_subdir: str
        :param rf_subdir: Directory, in which to store the receiver functions
            in time domain (sac).
        :type rf_subdir: str
        :param statloc_subdir: Directory, in which to store the station
            inventories (xml).
        :type statloc: str
        :param evt_subdir: Directory, in which to store the event catalogue.
        :type evt_subdir: str
        :param log_subdir: Directory that logs are stored in.
        :type log_subir: str
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
        :type starttime: :class:`obspy.UTCDateTime` or str
        :param endtime: Latest event date to be considered.
        :type endtime: :class:`obspy.UTCDateTime` or str
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
            network.
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
        :param evtcat: In case you want to use an already existing event
            catalogue
            in evtloc. If None a new catalogue will be downloaded (with the
            parameters defined before), by default None, defaults to None
        :type evtcat: str, optional
        :param continue_download: Will delete already used events from the
            event catalogue, so that the download will continue at the same
            place after being interrupted. Will make the continuation faster,
            but then old database will not be updated. Only makes sense if you
            define an old catalogue. Defaults to False.
        :type continue_download: bool, optional.
        :param loglvl: Level for the loggers. One of the following:
            CRITICAL, ERROR, WARNING, INFO,or DEBUG
            If the level is DEBUG, joblib will fall back to using only
            few cores and downloads will be retried,
            by default WARNING.
        :type loglvl: str, optional
        :raises NameError: For invalid phases.
        """

        # Allocate variables in self
        loglvl = pu.log_lvl[loglvl.upper()]
        self.loglvl = loglvl

        if format.lower() == ('hdf5' or 'h5'):
            self.h5 = True
        elif format.lower() in ('sac', 'mseed'):
            self.h5 = False
        else:
            raise NotImplementedError(
                'Output format %s is unknown.' % format)

        # Set velocity model
        self.model = TauPyModel('iasp91')

        self.phase = phase[:-1] + phase[-1].upper()
        self.pol = pol.lower()
        self.rot = rot.upper()
        self.deconmeth = deconmeth

        # Directories
        self.logdir = os.path.join(proj_dir, log_subdir)
        os.makedirs(self.logdir, exist_ok=True)
        self.evtloc = os.path.join(proj_dir, evt_subdir)
        self.statloc = os.path.join(proj_dir, statloc_subdir)
        self.rawloc = os.path.join(proj_dir, raw_subdir, self.phase)
        self.preproloc = os.path.join(proj_dir, prepro_subdir, self.phase)
        self.rfloc = os.path.join(proj_dir, rf_subdir, self.phase)

        # logger for the download steps
        self.logger = logging.getLogger('pyglimer.request')
        self.logger.setLevel(loglvl)
        self.fh = logging.FileHandler(
            os.path.join(self.logdir, 'request.log'))
        self.fh.setLevel(loglvl)
        self.logger.addHandler(self.fh)
        fmt = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(fmt)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(loglvl)
        consoleHandler.setFormatter(fmt)
        self.logger.addHandler(consoleHandler)
        # We don't really want to see all the warnings.
        if loglvl > logging.WARNING:
            warnings.filterwarnings("ignore")
        else:
            logging.captureWarnings(True)
            warnings_logger = logging.getLogger("py.warnings")
            warnings_logger.addHandler(self.fh)
            warnings_logger.setLevel(loglvl)

        # minimum magnitude
        self.minmag = minmag

        # Request time window
        self.starttime = UTCDateTime(starttime)
        self.endtime = UTCDateTime(endtime)

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
            self.tz = onsetP
        elif self.phase == 'S':
            self.maxdepth = 300
            self.min_epid = 55
            self.max_epid = 80
            self.tz = onsetS
        # (see Yuan et al. 2006)
        elif self.phase.upper() == 'SCS':
            self.maxdepth = 300
            self.min_epid = 50
            self.max_epid = 75
            self.tz = onsetS
        elif self.phase.upper() == 'SKS':
            # (see Zhang et. al. (2014))
            self.maxdepth = 300
            self.min_epid = 90
            self.max_epid = 120
            self.tz = onsetS
        else:
            raise NameError('The phase', self.phase, """is not valid or not
                            implemented yet.""")

        # network and station filters
        self.network = network
        self.station = station

        self.waveform_client = waveform_client

        # Download or process available data?
        if evtcat == 'None':
            # yaml files don't accept None
            evtcat = None
        if evtcat:
            self.evtfile = os.path.join(self.evtloc, evtcat)
            self.evtcat = read_events(self.evtfile)
        else:
            self.download_eventcat()
        self.continue_download = continue_download

    def download_eventcat(self):
        """
        Download the event catalogue from IRIS DMC.
        """
        # Server settings
        # 2021/02/16 Events only from IRIS as the USGS webserice tends to be
        # unstable and mixing different services will lead to a messed db
        self.webclient = Webclient('IRIS')
        event_cat_done = False

        # 05.07.2021: Shorten length of a to one year, which is a lot
        # more robust
        # :NOTE: perhaps it would be smart to save each year as a file?
        # BUt then again, they have different requirements...
        # Check length of request and split if longer than a year.
        a = 10*365.25*24*3600  # ten yrs in seconds
        if self.endtime-self.starttime > a:
            # Request is too big, break it down ito several requests

            starttimes = [self.starttime, self.starttime+a]
            while self.endtime-starttimes[-1] > a:
                starttimes.append(starttimes[-1]+a)
            endtimes = []
            endtimes.extend(starttimes[1:])
            endtimes.append(self.endtime)
            msg = 'Long request: Breaking it down to %s sub-requests.'\
                % str(len(endtimes))
            self.logger.info(msg)

            # Query
            self.evtcat = Catalog()
            # We convert the iterator to a list, so tqdm workii properly
            for st, et in tqdm(list(zip(starttimes, endtimes))):
                event_cat_done = False
                while not event_cat_done:
                    # Else the server could refuse the connection
                    time.sleep(0.2)
                    try:
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
                    except IncompleteRead:
                        # Server interrupted connection, just try again
                        # This usually happens with enormeous requests, we
                        # should reduce a
                        msg = "Server interrupted connection, \
                            restarting download..."
                        self.logger.warning(msg)
                        continue

        else:
            while not event_cat_done:
                # Else the server could refuse the connection
                time.sleep(0.2)
                try:
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
                    msg = "Server interrupted connection, \
                        restarting download..."
                    self.logger.warning(msg)
                    continue

        os.makedirs(self.evtloc, exist_ok=True)
        self.evtfile = os.path.join(
            self.evtloc, datetime.now().strftime("%Y%m%dT%H%M%S"))
        self.evtcat.write(self.evtfile, format="QUAKEML")
        msg = 'Successfully obtained %s events' % str(self.evtcat.count())
        self.logger.info(msg)

    def download_waveforms(self, verbose: bool = False):
        """
        Start the download of waveforms and response files.

        Parameters
        ----------
        verbose : Bool, optional
            Set True if you wish to log the output of the obspy MassDownloader.

        .. seealso::

            You should check whether the method
            :meth:`~pyglimer.waveform.request.Request.download_waveforms_small\
            _db` might be better suited for your needs.
            Both methods offer unique advantages.
        """
        downloadwav(
            self.phase, self.min_epid, self.max_epid, self.model, self.evtcat,
            self.tz, self.ta, self.statloc, self.rawloc, self.waveform_client,
            self.evtfile, network=self.network, station=self.station,
            log_fh=self.fh, loglvl=self.loglvl, verbose=verbose,
            saveasdf=self.h5, fast_redownload=self.continue_download)

    def download_waveforms_small_db(self, channel: str):
        """
        A different method to download raw waveform data. This method will
        be faster than
        :meth:`~pyglimer.waveform.request.Request.download_waveforms` for
        small databases (e.g., single networks or stations). Another
        advantage of this method is that the attributes network and station
        in :class:`~pyglimer.waveforms.request.Request` can be lists.

        :param channel: The channel you will want to download, accepts
            unix style wildcards.
        :type channel: str

        .. seealso::

            You should check whether the method
            :meth:`~pyglimer.waveform.request.Request.download_Waveforms`
            might be better suited for your needs. Both methods offer unique
            advantages.
        """
        download_small_db(
            self.phase, self.min_epid, self.max_epid, self.model, self.evtcat,
            self.tz, self.ta, self.statloc, self.rawloc, self.waveform_client,
            self.network, self.station, channel, self.h5)

    def preprocess(
        self, client: str = 'joblib',
            hc_filt: float or int or None = None):
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
            netrestr=self.network, statrestr=self.station, client=client,
            saveasdf=self.h5)
