#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:33:41 2019

@author: pm
"""


# This program does essentially the same as "glimertoobs.py", but builds a list 
# of waveforms to be downloaded. Its purpose is to test which method is the faster.
# reset variables
from IPython import get_ipython
get_ipython().magic('reset -sf')


#### IMPORT PREDEFINED SUBFUNCTIONS
from subfunctions.download import downloadwav
from subfunctions.preprocess import preprocess
import subfunctions.config as config

from obspy.core import *
from obspy.clients.iris import Client as iClient # To calculate angular distance and backazimuth
# careful with using clientas multiprocessing has a client class

from obspy.core.event.base import *
from obspy.geodetics.base import locations2degrees #calculate angular distances
from obspy.taup import TauPyModel #arrival times in 1D v-model
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
import os
import logging
import time
import progressbar
#from multiprocessing import Process,Queue #multi-thread processing - preprocess while downloading
from pathlib import Path
from obspy.clients.fdsn import Client as Webclient #as Webclient #web sevice
from threading import Thread #multi-thread processing


############## DEFINE VARIABLES - may be changed by user ###################
# values are located in subfunctions/config.py

webclient = Webclient("IRIS") #needs to be defined to download event catalogue - is it enough to
# exclusively use IRIS?

##############################################################################
##############################################################################

# logging
logging.basicConfig(filename='preprocess.log',level=logging.DEBUG) #DEBUG

logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
#logger.setLevel(logging.DEBUG)
#logger.basicConfig(filename='download.log',level=logging.DEBUG) #DEBUG

# download event catalogue
event_cat = webclient.get_events(starttime = config.starttime, endtime = config.endtime,
                                  minlatitude = config.eMINLAT, maxlatitude = config.eMAXLAT,
                                  minlongitude = config.eMINLON, maxlongitude = config.eMAXLON,
                                  minmagnitude = config.minMag, maxmagnitude = config.maxMag)


config.folder = "not_started" #resetting momentary event download folder

# multi-threading
if __name__ == '__main__':
    Thread(target = downloadwav,
           args = (webclient,config.min_epid,config.max_epid,config.model,event_cat)).start()
    
    Thread(target = preprocess,
           args = (config.taper_perc,config.taper_type,event_cat,webclient,config.model)).start()
   
   
