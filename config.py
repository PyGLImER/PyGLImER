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
# If debug setting is true, all loggers will go to DEBUG mode and all warnings
# will be shown. That will result in a lot of information being shown!
# debug = False

# Define if you want to download new files or use old
evtcat = None # 'all_greater55'  # either None (downloads new) or string of filename
# in evtloc

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

# polarisation of converted wave for PRFs ("h" or "v")
#pol = "v"
# don't change
#pol = pol.lower()
#phase = phase.upper()

# %% DIRECTORY CONFIGURATION
# lith1 = '/home/pm/LITHO1.0/bin/access_litho'  # location of lith1 file

RF = "output/waveforms/RF/"  # save RF here
waveform = "output/waveforms/raw/"
outputloc = "output/waveforms/preprocessed/"
statloc = "output/stations"
evtloc = "output/event_catalogues"
ratings = "data/ratings/"
ccp = "output/ccps"

# %% EVENT AND DOWNLOAD VALUES
# Set values to "None" if they aren't requried / desired
# Time frame is identical to the station inventory
starttime = UTCDateTime("2020-04-15")
endtime = UTCDateTime("1970-01-01")
minmag = 5.5

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
waveform_client = ['http://auspass.edu.au/', "IRIS",  'BGR', 'EMSC', 'ETH', 'GEONET', 'GFZ', 'ICGC', 'INGV', 'IPGP', 'IRIS', 'ISC',
 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', 'ODC', 'ORFEUS',
 'RESIF', 'SCEDC', 'TEXNET', 'USP']  # ["IRIS", 'NCEDC', 'TEXNET', 'SCEDC']
# See https://www.fdsn.org/webservices/datacenters/
# Possible options:
# 'http://auspass.edu.au/'
# 'BGR', 'EMSC', 'ETH', 'GEONET', 'GFZ', 'ICGC', 'INGV', 'IPGP', 'IRIS', 'ISC',
# 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', 'ODC', 'ORFEUS',
# 'RASPISHAKE', 'RESIF', 'SCEDC', 'TEXNET', 'USP'
#
# clients on which the download should be retried, list:
re_clients = ['http://auspass.edu.au/', "IRIS",  'BGR', 'EMSC', 'ETH', 'GEONET', 'GFZ', 'ICGC', 'INGV', 'IPGP', 'IRIS', 'ISC',
 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', 'ODC', 'ORFEUS',
 'RESIF', 'SCEDC', 'TEXNET', 'USP']  # It's usually enoguh to have only IRIS here
# as IRIS tends to be very unreliable
#
# %% PRE-PROCESSING VALUES #####

# Rotation #
# "RTZ", "LQT", "LQT_min", or "PSS"
rot = "PSS"


# %% CCP settings
# Should the 3D velocity models be saved? That saves some computation time
# (something about 5 seconds), but costs disk space. However, when working
# with many cores this can lead to unforseen errors (usually UnpicklingErrors)
# The model will however still be cached
savevmodel = False
