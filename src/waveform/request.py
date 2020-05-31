#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the request class that is used to initialise the FDSN request for
the waveforms, the preprocessing of the waveforms, and the creation of
time domain receiver functions.

Created on Mon Apr 27 10:55:10 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""
import os
import subprocess
from http.client import IncompleteRead
from datetime import datetime
from pathlib import Path

from obspy import read_events, Catalog
from obspy.clients.fdsn import Client as Webclient
from obspy.taup import TauPyModel
from obspy.clients.fdsn.client import FDSNException

import config
from .download import downloadwav
from .preprocess import preprocess


class Request(object):
    """"Initialises the FDSN request for
    the waveforms, the preprocessing of the waveforms, and the creation of
    time domain receiver functions."""

    def __init__(self, phase, starttime, endtime, minmag=5.5, event_coords=None,
                 network=None, station=None, waveform_client=None,
                 re_client=['IRIS'], evtcat=None):
        # Set velocity model
        self.model = TauPyModel('iasp91')

        # Set variables in self
        self.phase = phase.upper()
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
        if self.phase == 'P':
            self.maxdepth = None
            self.min_epid = 28.1
            self.max_epid = 95.8
        elif self.phase == 'S':
            self.maxdepth = 300
            self.min_epid = 55
            self.max_epid = 80
        else:
            raise NameError('The phase', phase, """is not valid or not
                            implemented yet.""")

        # network and station filters
        self.network = network
        self.station = station

        # Server settings
        try:
            self.webclient = Webclient('USGS')
        except FDSNException:
            self.webclient = Webclient('IRIS')
            print('Falling back to IRIS\' event service.')

        self.waveform_client = waveform_client
        self.re_client = re_client

        # Download or process available data?
        if evtcat:
            self.evtcat = read_events(os.path.join(config.evtloc, evtcat))
        else:
            self.download_eventcat()

    def download_eventcat(self):
        event_cat_done = False

        # The USGS webservice does only allow requests of a maximum size of
        # 20000, so I will have to do several requests and join the
        # catalogue afterwards if it's too big for a single download

        while not event_cat_done:
            try:
                self.evtcat = self.webclient.get_events(
                    starttime=self.starttime, endtime=self.endtime,
                    minlatitude=self.eMINLAT, maxlatitude=self.eMAXLAT,
                    minlongitude=self.eMINLON, maxlongitude=self.eMAXLON,
                    minmagnitude=5.5, maxmagnitude=10, maxdepth=self.maxdepth)

                event_cat_done = True
            except FDSNException:
                # Request is too big, break it down ito several requests
                a = 20*365.25*24*3600  # 20 years in seconds
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
                            minlatitude=self.eMINLAT, maxlatitude=self.eMAXLAT,
                            minlongitude=self.eMINLON,
                            maxlongitude=self.eMAXLON, minmagnitude=self.minmag,
                            maxmagnitude=10, maxdepth=self.maxdepth))
                event_cat_done = True

            except IncompleteRead:
                # Server interrupted connection, just try again
                print("Warning: Server interrupted connection,\
                      catalogue download is restarted.")
                continue

        if not Path(config.evtloc).is_dir():
            subprocess.call(["mkdir", "-p", config.evtloc])
            # check if there is a better format for event catalog
        self.evtcat.write(os.path.join(config.evtloc, str(datetime.now())),
                          format="QUAKEML")

    def download_waveforms(self):
        downloadwav(self.min_epid, self.max_epid, self.model, self.evtcat)

    def preprocess(self):
        preprocess(0.05, self.evtcat, self.model, 'hann')
