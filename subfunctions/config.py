#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:14:47 2019

@author: pm
"""
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel  # arrival times in 1D v-model

# general settings
# Define if you want to download new files or use old
evtcat = "2020-02-29 15:49:43.388928"  # either None (downloads new) or file
wavdownload = False  # Bool - true for completing download, false: only
# processes already existing waveforms in waveform of phase phase.

decon_meth = "it"  # it=iterative deconvolution (Ligorria & Ammon, 1999),
# dampedf: damped frequency deconvolution - not recommended
# False/None: don't create RF

#### P or S ####
# put string either "P" or "S" - case-sensitive
phase = "S"

#### Rotation ####
# "RTZ","LQT","PSS" latter is not implemented yet
rot = "PSS"

#### DIRECTORY CONFIGURATION
lith1 = '/home/pm/LITHO1.0/bin/access_litho'  # location of lith1 file

RF = "waveforms/RF/" + phase #save RF here
waveform = "waveforms/raw/" + phase
outputloc = "waveforms/preprocessed/" + phase
failloc = "waveforms/rejected"  # Not in use anymore
statloc = "stations"
evtloc = "event_catalogues"

###### EVENT VALUES ######
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
starttime = UTCDateTime("1988-01-01")
endtime = UTCDateTime("2019-01-10")
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
# (see Wilson et. al., 2006)
elif phase == "S":
    min_epid = 55
    max_epid = 80
# event depth in km (see Wilson et. al., 2006):
if phase == "P":
    maxdepth = None
elif phase == "S":
    maxdepth = 300


# define 1D velocity model
model = TauPyModel(model="iasp91")

# Clients to download waveform
# !No specified providers (None) will result in all known ones being queued.!
# Else "IRIS","ORFEUS",etc.
waveform_client = None
#
# clients on which the download should be retried:
re_clients = ["IRIS", "ORFEUS", "ODC", "GFZ", "SCEDC", "TEXNET", "BGR", "ETH",
              "GEONET", "ICGC", "INGV", "IPGP", "KNMI", "KOERI", "LMU",
              "NCEDC", "NIEP", "NOA", "RESIF", 'USP']

### PRE-PROCESSING VALUES #####
# time window before and after first arrival
if phase == "P":
    # time window before
    tz = 30
# time window after
    ta = 120
elif phase == "S":
    tz = 120
    ta = 120

taper_perc = 0.05  # max taping percentage - float (0.05)
taper_type = 'hann'
# define type of taper, Options: {cosine,barthann,bartlett,blackman,
# blackmannharris, bohman,boxcar,chebwin, flattop,gaussian,general_gaussian,
# hamming,hann,kaiser,nuttall,parzen,slepian,triang}
# type = string                 ('hann')

# low-cut-off frequencies for SNR check
lowco = [0.03, 0.1, 0.5]

# SNR criteria
SNR_criteria = [7.5, 1, 10]  # [snrr, snrr2/snrr, snrz]

###### DON'T change program will change automatically. #################
folder = "undefined"  # the download is happening here right now