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
waveform = "waveforms"
outputloc = "preprocessed"
failloc = "rejected"
statloc = "stations"

###### EVENT VALUES ######
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
starttime = UTCDateTime("2018-01-01")
endtime = UTCDateTime("2018-06-02")
eMINLAT = -30.00
eMAXLAT = 20.0
eMINLON = -100.0
eMAXLON = -90.0
minMag = 5.5
maxMag = 10.0

# epicentral distances:
min_epid = 28.1
max_epid = 95.8


# define 1D velocity model
model = TauPyModel(model="iasp91")

### PRE-PROCESSING VALUES #####
taper_perc = 0.05 #max taping percentage - float (0.05)
taper_type = 'hann' #define type of taper, Options: {cosine,barthann,bartlett,blackman,blackmannharris,
                                                    #bohman,boxcar,chebwin,flattop,gaussian,general_gaussian,
                                                    #hamming,hann,kaiser,nuttall,parzen,slepian,triang}
                    # type = string                 ('hann')

# low-cut-off frequencies for SNR check
lowco = [0.03, 0.1, 0.5]



################################################################3
                    
                    
                    
###### DON'T change program will change automatically. #################
                    
folder = "undefined" #Subdirectory inside of waveforms folder - the download is happening here right now