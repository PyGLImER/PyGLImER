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





webclient = Webclient("IRIS") #needs to be defined to download event catalogue - is it enough to
# exclusively use IRIS?

# station values:
# Set values to "None" if they aren't requried / desired
#MINLAT = 56.0
#MAXLAT = 62.0
#MINLON = -150.0
#MAXLON = -145.0
starttime = UTCDateTime("2018-01-01")
endtime = UTCDateTime("2018-06-02")

###### EVENT VALUES ######
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
eMINLAT = -30.00
eMAXLAT = 20.0
eMINLON = -100.0
eMAXLON = -90.0
minMag = 5.5
maxMag = 10.0
# epicentral distances:
min_epid = 30
max_epid = 90


# define 1D velocity model
model = TauPyModel(model="iasp91")

### PRE-PROCESSING VALUES #####
taper_perc = 0.05 #max taping percentage - float (0.05)
taper_type = 'hann' #define type of taper, Options: {cosine,barthann,bartlett,blackman,blackmannharris,
                                                    #bohman,boxcar,chebwin,flattop,gaussian,general_gaussian,
                                                    #hamming,hann,kaiser,nuttall,parzen,slepian,triang}
                    # type = string                 ('hann')




##############################################################################
##############################################################################

# logging
logging.basicConfig(filename='waveform.log',level=logging.DEBUG) #DEBUG

# download event catalogue
event_cat = webclient.get_events(starttime = starttime, endtime = endtime,
                                  minlatitude = eMINLAT, maxlatitude = eMAXLAT,
                                  minlongitude = eMINLON, maxlongitude = eMAXLON,
                                  minmagnitude = minMag, maxmagnitude = maxMag)

#state = downloadwav(webclient,starttime,endtime,eMINLAT,eMAXLAT,eMINLON,eMAXLON,minMag,maxMag,min_epid,max_epid,model,event_cat)
#preprocess(taper_perc,taper_type,waveform,outputloc,event_cat,webclient,state)
config.state = False #is download finished?
config.ei = 0 #event ID

if __name__ == '__main__':
    Thread(target = downloadwav,
           args = (webclient,min_epid,max_epid,model,event_cat)).start()
#    time.sleep(30)
#    Thread(target = preprocess,
#           args = (taper_perc,taper_type,event_cat,webclient)).start()

    

    # New thought: Toss waveforms in a folder that is named by event-ID 
   
   
