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
evtcat = None  # either None (downloads new) or file
wavdownload = True  # Bool - true for completing download, false: only
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
starttime = UTCDateTime("1970-01-01")
endtime = UTCDateTime("2019-05-19")
eMINLAT = None
eMAXLAT = None
eMINLON = None
eMAXLON = None

# Station and Network codes
# type : str
# wildcards allowed. If None, all that are available are requested
network = None
station = None

# define 1D velocity model for arrival time calculation
model = TauPyModel(model="iasp91")

# Clients to download waveform
# !No specified providers (None) will result in all known ones being queued.!
# None is not recommended as some clients are unstable or do not provide any
# data, waiting for these clients causes the script to be very slow.
# Else "IRIS","ORFEUS",etc.
# !!NOTE: For machines with little RAM, the script might interrupt if too
# many clients are chosen.
#
waveform_client = ["IRIS", 'NCEDC', 'TEXNET', 'SCEDC']
# See https://www.fdsn.org/webservices/datacenters/
# Possible options:
# 'http://auspass.edu.au/'
# ‘BGR’, ‘EMSC’, ‘ETH’, ‘GEONET’, ‘GFZ’, ‘ICGC’, ‘INGV’, ‘IPGP’, ‘IRIS’, ‘ISC’,
# ‘KNMI’, ‘KOERI’, ‘LMU’, ‘NCEDC’, ‘NIEP’, ‘NOA’, ‘ODC’, ‘ORFEUS’,
# ‘RASPISHAKE’, ‘RESIF’, ‘SCEDC’, ‘TEXNET’, ‘USP’
#
# clients on which the download should be retried, list:
re_clients = ["IRIS"]  # It's usually enoguh to have only IRIS here
# as IRIS tends to be very unreliable
#
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
# lowcoS = 0.03  # 0.05 lowco frequency for SRF
lowcoS = .01  # Changed 18.05 to compare with Hopper et. al. 2018
highco = np.linspace(.33, .175, 4)

# highco = [0.175]  # Rychert et al
# highco = [.5, .33, .25]  # only for SRF

# SNR criteria for QC
QC = True  # Do quality control or not

SNR_criteriaP = [7.5, 1, 10]  # [snrr, snrr2/snrr, snrz]

SNR_criteriaS = [24, .4, 1]  # QC4
# SNR_criteriaS = [27.5, .4, 1]  # QC3
# SNR_criteriaS = [20, .5, 1]  # QC2
# SNR_criteriaS = [35, .4, 1]  # QC1
# [primary/noise, sidelobe/primary, r/z conversions]

# %% CCP settings
# Should the 3D velocity models be saved? That saves some computation time
# (something about 5 seconds), but costs disk space. However, when working
# with many cores this can lead to unforseen errors (usually UnpicklingErrors)
# The model will however still be cached
savevmodel = False

# %% DON'T change program will change automatically!
folder = "undefined"  # the download is happening here right now
