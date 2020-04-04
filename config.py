#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:14:47 2019

@author: pm
"""
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel  # arrival times in 1D v-model
import numpy as np

# %% general settings
# Define if you want to download new files or use old
evtcat = None #2020-02-29 15:49:43.388928"  # either None (downloads new) or file
wavdownload = True  # Bool - true for completing download, false: only
# processes already existing waveforms in waveform of phase phase.

decon_meth = None  # it=iterative deconvolution (Ligorria & Ammon, 1999)
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

RF = "waveforms/RF/" + phase  # save RF here
waveform = "waveforms/raw/" + phase
outputloc = "waveforms/preprocessed/" + phase
failloc = "waveforms/rejected"  # Not in use anymore
statloc = "stations"
evtloc = "event_catalogues"
ratings = "data/ratings/"

# %% EVENT AND DOWNLOAD VALUES
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
starttime = UTCDateTime("1980-01-01")
endtime = UTCDateTime("2019-03-10")
eMINLAT = None
eMAXLAT = None
eMINLON = None
eMAXLON = None
if phase == "P":
    minMag = 5.5
elif phase == "S":
    minMag = 5.8
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
# None is not recommended as some clients are unstable or do not provide any
# data, waiting for these clients causes the script to be very slow.
# Else "IRIS","ORFEUS",etc.
waveform_client = ["IRIS", "NCEDC", "ORFEUS", "ODC", "TEXNET", "BGR", "ETH",
 "GEONET", "ICGC", "INGV", "IPGP", "KNMI", "KOERI", "NCEDC", "NIEP", "NOA", "RESIF", 'USP']
# None  # ["IRIS", "NCEDC"]
#
# clients on which the download should be retried, list:
re_clients = ["IRIS", "NCEDC", "ORFEUS", "ODC", "TEXNET", "BGR", "ETH",
 "GEONET", "ICGC", "INGV", "IPGP", "KNMI", "KOERI", "NCEDC", "NIEP", "NOA", "RESIF", 'USP']
# Clients that cause problems and are excluded:
# ['GFZ', 'LMU', 'SCEDC']
# %% PRE-PROCESSING VALUES #####

# Rotation #
# "RTZ","LQT","PSS"
rot = "RTZ"

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
QC = False  # Do quality control or not
if phase == "P":
    SNR_criteria = [7.5, 1, 10]  # [snrr, snrr2/snrr, snrz]
elif phase == "S":
    # SNR_criteria = [7.5, .2, .66]
    # SNR_criteria = [7.5, .2, 1]
    SNR_criteria = [45, .5, 1]  # to reproduce Rychert
    # [primary/noise, sidelobe/primary, r/z conversions]

# %% DON'T change program will change automatically!
folder = "undefined"  # the download is happening here right now
