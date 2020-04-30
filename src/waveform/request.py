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

from obspy import read_events
from obspy.clients.fdsn import Client as Webclient
from obspy.taup import TauPyModel

import config
from .download import downloadwav
from .preprocess import preprocess

webclient = Webclient('IRIS')


class Request(object):
    """"Initialises the FDSN request for
    the waveforms, the preprocessing of the waveforms, and the creation of
    time domain receiver functions."""

    def __init__(self, phase, starttime, endtime, event_coords=None,
                 network=None, station=None, waveform_client=None,
                 re_client=['IRIS'], evtcat=None):
        # Set velocity model
        self.model = TauPyModel('iasp91')

        # Set variables in self
        self.phase = phase.upper()

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
        self.waveform_client = waveform_client
        self.re_client = re_client

        # Download or process available data?
        if evtcat:
            self.evtcat = read_events(os.path.join(config.evtloc, evtcat))
        else:
            self.download_eventcat()

    def download_eventcat(self):
        event_cat_done = False

        while not event_cat_done:
            try:
                self.event_cat = webclient.get_events(
                    starttime=self.starttime, endtime=self.endtime,
                    minlatitude=self.eMINLAT, maxlatitude=self.eMAXLAT,
                    minlongitude=self.eMINLON, maxlongitude=self.eMAXLON,
                    minmagnitude=5.5, maxmagnitude=10, maxdepth=self.maxdepth)

                event_cat_done = True
            except IncompleteRead:
                # Server interrupted connection, just try again
                print("Warning: Server interrupted connection,\
                      catalogue download is restarted.")
                continue

        if not Path(config.evtloc).is_dir():
            subprocess.call(["mkdir", "-p", config.evtloc])
            # check if there is a better format for event catalog
            self.event_cat.write(config.evtloc + '/' + str(datetime.now()),
                                 format="QUAKEML")

    def download_waveforms(self):
        downloadwav(self.min_epid, self.max_epid, self.model, self.event_cat)

    def preprocess(self):
        preprocess(0.05, self.event_cat, webclient, self.model, 'hann')
