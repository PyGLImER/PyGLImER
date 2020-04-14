#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:14:47 2019

General configuration for GLImER.

@author: pm
"""
import numpy as np
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel  # arrival times in 1D v-model

# %% general settings
# Define if you want to download new files or use old
evtcat = '2020-04-14 13:35:26.136417' #None  # either None (downloads new) or file
wavdownload = False  # Bool - true for completing download, false: only
# processes already existing waveforms in waveform of phase phase.

decon_meth = "it"  # it=iterative deconvolution (Ligorria & Ammon, 1999)
# dampedf: damped frequency deconvolution
# fqd: frequency dependent damping - not a good choice for SRF
# waterlevel Langston (1977)
# multit - for multitaper (Helffrich, 2006)
# False/None: don't create RF

# %% P or S ####
# put string either "P" or "S"
phase = "S"
# don't change
phase = phase.upper()

# %% DIRECTORY CONFIGURATION
lith1 = '/home/pm/LITHO1.0/bin/access_litho'  # location of lith1 file

RF = "output/waveforms/RF/" + phase  # save RF here
waveform = "output/waveforms/raw/" + phase
outputloc = "output/waveforms/preprocessed/" + phase
failloc = "output/waveforms/rejected"  # Not in use anymore
statloc = "output/stations"
evtloc = "output/event_catalogues"
ratings = "data/ratings/"
ccp = "output/ccps"

# %% EVENT AND DOWNLOAD VALUES
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
starttime = UTCDateTime("2009-01-01")
endtime = UTCDateTime("2011-12-31")
eMINLAT = None
eMAXLAT = None
eMINLON = None
eMAXLON = None
if phase == "P":
    minMag = 5.5
elif phase == "S":
    minMag = 5.5
maxMag = 10.0

# Station and Network codes
# type : str
# wildcards allowed. If None, all that are available are requested
network = "YP"
station = None

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
# None is not recommended as some clients are unstable or do not provide any
# data, waiting for these clients causes the script to be very slow.
# Else "IRIS","ORFEUS",etc.
waveform_client = ["IRIS"]  # , "NCEDC", "ORFEUS"] #, "ODC", "TEXNET", "BGR", "ETH",
# "GEONET", "ICGC", "INGV", "IPGP", "KNMI", "KOERI", "NCEDC", "NIEP", "NOA", "RESIF", 'USP']
# None  # ["IRIS", "NCEDC"]
#
# clients on which the download should be retried, list:
re_clients = ["IRIS"]  # , "NCEDC", "ORFEUS"] #, "ODC", "TEXNET", "BGR", "ETH",
# "GEONET", "ICGC", "INGV", "IPGP", "KNMI", "KOERI", "NCEDC", "NIEP", "NOA", "RESIF", 'USP']
# Clients that cause problems and are excluded:
# ['GFZ', 'LMU', 'SCEDC']
# %% PRE-PROCESSING VALUES #####

# Rotation #
# "RTZ", "LQT", "LQT_min", or "PSS"
rot = "PSS"

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
lowco = [.03, .1, .5]  # only for PRF
lowcoS = 0.03  # 0.05 lowco frequency for SRF
highco = np.linspace(.33, .175, 4)
# highco = [0.175]  # Rychert et al
# highco = [.5, .33, .25]  # only for SRF

# SNR criteria for QC
QC = True  # Do quality control or not
if phase == "P":
    SNR_criteria = [7.5, 1, 10]  # [snrr, snrr2/snrr, snrz]
elif phase == "S":
    # SNR_criteria = [7.5, .2, .66]
    SNR_criteria = [20, .5, 1] #QC2
    #SNR_criteria = [35, .5, 1]  #QC1 to reproduce Rychert
    # [primary/noise, sidelobe/primary, r/z conversions]

# %% DON'T change program will change automatically!
folder = "undefined"  # the download is happening here right now
