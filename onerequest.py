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

#from obspy.signal.invsim import corn_freq_2_paz
from obspy.core import *
from obspy.clients.iris import Client as iClient # To calculate angular distance and backazimuth
from obspy.clients.fdsn import Client,header #web sevice
from obspy.core.event.base import *
from obspy.geodetics.base import locations2degrees #calculate angular distances
from obspy.taup import TauPyModel #arrival times in 1D v-model
import logging
import time
import progressbar
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader


############## DEFINE VARIABLES - may be changed by user ###################

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

# Set client - if desired list of clients
client = Client("IRIS")

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

############# CREATE ERROR LOG #####################
logging.basicConfig(filename='waveform.log',level=logging.WARNING) #DEBUG
logging.info('%(asctime)s')




#################### CREATE STATION INVENTORY ###############################
# Not needed as all available staitons in the given radius will be requested
"""station_inv = client.get_stations(starttime = starttime,
                                endtime = endtime, level = "response",
                               minlongitude = MINLON, maxlongitude = MAXLON,
                                minlatitude = MINLAT, maxlatitude = MAXLAT,
                                channel = 'BH*,HH*') #level="channel""""

############# Download simulation station's response function #################
station_simulate = client.get_stations(level = "response",channel = 'BH*',network = 'IU',station = 'HRV') #a proper network and station has to be inserted here
paz_sim = station_simulate[0][0][0].response #instrument response of the channel downloaded above

# create dictionary with necassary data - no correction for sensitivity is conducted
paz_sim = {"gain":paz_sim._get_overall_sensitivity_and_gain(output="VEL")[0],"sensitivity":paz_sim._get_overall_sensitivity_and_gain(output="VEL")[1], "poles":paz_sim.get_paz().poles,"zeros":paz_sim.get_paz().zeros}


##################### CREATE EVENT CATALOGUE ################################
# By default only the preferred origin is downloaded
# Meaning, there will always be only one origin
event_cat = client.get_events(starttime = starttime, endtime = endtime,
                              minlatitude = eMINLAT, maxlatitude = eMAXLAT,
                              minlongitude = eMINLON, maxlongitude = eMAXLON,
                              minmagnitude = minMag, maxmagnitude = maxMag)


# Calculate the min and max theoretical arrival time after event time
min_time = model.get_travel_times(source_depth_in_km=500,
                                     distance_in_degree=min_epid,
                                     phase_list=["P"])

max_time = model.get_travel_times(source_depth_in_km=0.001,
                                     distance_in_degree=max_epid,
                                     phase_list=["P"])




########## NEW PART OF THE PROGRAM ###################
# Circular domain around the epicenter. This will download all data between
# 70 and 90 degrees distance from the epicenter. This module also offers
# rectangular and global domains. More complex domains can be defined by
# inheriting from the Domain class.



# Loop over each event
for event in event_cat:
    # fetch event-data   
    origin_time = event.origins[0].time
    evtlat=event.origins[0].latitude
    evtlon=event.origins[0].longitude
    
     domain = CircularDomain(latitude=evtlat, longitude=evtlong,
                            minradius=min_epid, maxradius=max_epid)
    
    restrictions = Restrictions(
        # Get data from sufficient time before earliest arrival and after the latest arrival
        # Note: All the traces will still have the same length
        starttime=origin_time + min_time - 30,
        endtime=origin_time + max_time + 120,
        # You might not want to deal with gaps in the data. If this setting is
        # True, any trace with a gap/overlap will be discarded.
        reject_channels_with_gaps=True,
        # And you might only want waveforms that have data for at least 95 % of
        # the requested time span. Any trace that is shorter than 95 % of the
        # desired total duration will be discarded.
        minimum_length=1.00,
        # No two stations should be closer than 1 km to each other. This is
        # useful to for example filter out stations that are part of different
        # networks but at the same physical station. Settings this option to
        # zero or None will disable that filtering.
        # Guard against the same station having different names.
        minimum_interstation_distance_in_m=1000.0,
        # Only HH or BH channels. If a station has BH channels, those will be
        # downloaded, otherwise the HH. Nothing will be downloaded if it has
        # neither. You can add more/less patterns if you like.
        channel_priorities=["BH[ZNE]","HH[ZNE]"],
        # Location codes are arbitrary and there is no rule as to which
        # location is best. Same logic as for the previous setting.
        location_priorities=["", "00", "10"])



