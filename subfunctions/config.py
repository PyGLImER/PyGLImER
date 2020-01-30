#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:14:47 2019

@author: pm
"""
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel #arrival times in 1D v-model


### changeable by user
#### DIRECTORY CONFIGURATION
waveform = "waveforms/raw"
outputloc = "waveforms/preprocessed"
failloc = "waveforms/rejected" #Not in use anymore
statloc = "stations"
evtloc = "event_catalogues"

###### EVENT VALUES ######
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
starttime = UTCDateTime("2011-04-10")
endtime = UTCDateTime("2015-04-12")
eMINLAT = -90
eMAXLAT = 90
eMINLON = -180
eMAXLON = 180
minMag = 5.5
maxMag = 10.0

# epicentral distances:
min_epid = 28.1
max_epid = 95.8


# define 1D velocity model
model = TauPyModel(model="iasp91")

# Clients to download waveform
# !No specified providers (None) will result in all known ones being queued.!
# Else "IRIS","ORFEUS",etc.
waveform_client = None

### PRE-PROCESSING VALUES #####
# time window before and after first arrival
tz = 30 #time window before 
ta = 120 #time window after

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