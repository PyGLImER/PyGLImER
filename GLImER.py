#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:33:41 2019
Script to start the data accumulation for automatic RF processing.
IMPORTANT:
!!All configuration is done in config.py!!
@author: pm
"""

from datetime import datetime
from http.client import IncompleteRead
from pathlib import Path
# modules
import subprocess
from threading import Thread  # multi-thread processing

from obspy import read_events
from obspy.clients.fdsn import Client as Webclient  # web sevice

import config
# IMPORT PREDEFINED SUBFUNCTIONS
from src.waveform.download import downloadwav
from src.waveform.preprocess import preprocess

# !values are located in subfunctions/config.py!

webclient = Webclient("IRIS")  # client to download event catalogue

##############################################################################
##############################################################################

if not config.evtcat:
    # download event catalogue
    # The server is often not reachable and returns an "IncompleteRead" Error,
    # just try downloading until sucessful
    event_cat_done = 0
    while not event_cat_done:
        try:
            event_cat = webclient.get_events(starttime=config.starttime,
                                             endtime=config.endtime,
                                             minlatitude=config.eMINLAT,
                                             maxlatitude=config.eMAXLAT,
                                             minlongitude=config.eMINLON,
                                             maxlongitude=config.eMAXLON,
                                             minmagnitude=config.minMag,
                                             maxmagnitude=config.maxMag,
                                             maxdepth=config.maxdepth)
            event_cat_done = True
        except IncompleteRead:
            # Server interrupted connection, just try again
            print("Warning: Server interrupted connection,\
                  catalogue download is restarted.")
            continue

    if not Path(config.evtloc).is_dir():
        subprocess.call(["mkdir", "-p", config.evtloc])
    # check if there is a better format for event catalog
    event_cat.write(config.evtloc + '/' + str(datetime.now()),
                    format="QUAKEML")

else:  # use old/downloaded event catalogue
    event_cat = read_events(config.evtloc + '/' + config.evtcat)
model = config.model  # otherwise it calls the model function twice
if config.wavdownload:
    config.folder = "not_started"  # resetting momentary event download folder
    # multi-threading
    if __name__ == '__main__':
        Thread(target=downloadwav,
               args=(config.min_epid, config.max_epid,
                     model, event_cat)).start()

        Thread(target=preprocess,
               args=(config.taper_perc, event_cat, webclient, model,
                     config.taper_type)).start()

else:  # Only preprocess files in waveform location
    config.folder = "finished"
    preprocess(config.taper_perc, event_cat, webclient,
               model)
