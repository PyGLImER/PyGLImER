#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:14:47 2019

@author: pm
"""
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel #arrival times in 1D v-model

### changeable by user


#### P or S ####
phase = "S" #put string either "P" or "S" - case-sensitive

#### Rotation ####
rot = "LQT" #"RTZ","LQT","PSS" latter is not implemented yet

#### DIRECTORY CONFIGURATION
lith1 = '/home/pm/LITHO1.0/bin/access_litho' #location of lith1 file

waveform = "waveforms/raw/"+phase
outputloc = "waveforms/preprocessed/"+phase
failloc = "waveforms/rejected" #Not in use anymore
statloc = "stations"
evtloc = "event_catalogues"

###### EVENT VALUES ######
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
starttime = UTCDateTime("2004-04-25")
endtime = UTCDateTime("2008-01-02")
eMINLAT = -90
eMAXLAT = 90
eMINLON = -180
eMAXLON = 180
minMag = 5.5
maxMag = 10.0

# epicentral distances:
if phase == "P":
    min_epid = 28.1
    max_epid = 95.8 
elif phase == "S": # (see Wilson et. al., 2006)
    min_epid = 55
    max_epid = 75 # 80 may be possible     
    
# event depth (see Wilson et. al., 2006):
if phase == "P":
    maxdepth = None #km
elif phase == "S":
    maxdepth = 300 #km


# define 1D velocity model
model = TauPyModel(model="iasp91")

# Clients to download waveform
# !No specified providers (None) will result in all known ones being queued.!
# Else "IRIS","ORFEUS",etc.
waveform_client = None
#
# clients on which the download should be retried:
re_clients = ["IRIS","ORFEUS","ODC","GFZ","SCEDC","TEXNET","BGR","ETH","GEONET","ICGC","INGV","IPGP","KNMI",
              "KOERI","LMU","NCEDC","NIEP","NOA","RESIF",'USP']

### PRE-PROCESSING VALUES #####
# time window before and after first arrival
if phase == "P":
    tz = 30 #time window before 
    ta = 120 #time window after
elif phase == "S":
    tz = 45
    ta = 120

taper_perc = 0.05 #max taping percentage - float (0.05)
taper_type = 'hann' #define type of taper, Options: {cosine,barthann,bartlett,blackman,blackmannharris,
                                                    #bohman,boxcar,chebwin,flattop,gaussian,general_gaussian,
                                                    #hamming,hann,kaiser,nuttall,parzen,slepian,triang}
                    # type = string                 ('hann')

# low-cut-off frequencies for SNR check
lowco = [0.03, 0.1, 0.5]

# SNR criteria
SNR_criteria = [7.5, 1, 10] #[snrr, snrr2/snrr, snrz]



################################################################3
                    
                    
                    
###### DON'T change program will change automatically. #################
                    
folder = "undefined" #Subdirectory inside of waveforms folder - the download is happening here right now