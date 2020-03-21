#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:33:41 2019

@author: pm
"""
# reset variables
# from IPython import get_ipython
# get_ipython().magic('reset -sf')


# IMPORT PREDEFINED SUBFUNCTIONS
from subfunctions.download import downloadwav
from subfunctions.preprocess import preprocess
from subfunctions.preprocess import file_in_db
import subfunctions.config as config

# modules
from obspy.core import *
from obspy.clients.iris import Client as iClient  # To calculate ad & backaz
from obspy.core.event.base import *
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel  # arrival times in 1D v-model
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
import os
import logging
import time
import subprocess
from pathlib import Path
from obspy.clients.fdsn import Client as Webclient  # web sevice
from threading import Thread  # multi-thread processing
from datetime import datetime
from pathlib import Path
from http.client import IncompleteRead
from obspy import read_events


# !values are located in subfunctions/config.py!

webclient = Webclient("IRIS")  # client to download event catalogue

##############################################################################
##############################################################################

# logging
logging.basicConfig(filename='preprocess.log',
                    level=logging.WARNING, format='%(asctime)s %(message)s')
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
               args=(config.taper_perc, config.taper_type,
                       event_cat, webclient, model)).start()
        
else:  # Only preprocess files in waveform location
    config.folder = "finished"
    preprocess(config.taper_perc, config.taper_type, event_cat, webclient,
               model)
    
