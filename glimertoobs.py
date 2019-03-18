#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:33:41 2019

@author: pm
"""

# reset variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

from obspy.signal.invsim import corn_freq_2_paz
from obspy.core import *
from obspy.clients.fdsn import Client,header #web sevice
from obspy.core.event.base import *
from obspy.geodetics.base import locations2degrees #calculate angular distances
from obspy.taup import TauPyModel #arrival times in 1D v-model
import logging
import time
import progressbar


############## DEFINE VARIABLES - may be changed by user ###################

# station values:
# Set values to "None" if they aren't requried / desired
MINLAT = 56.0
MAXLAT = 62.0
MINLON = -150.0
MAXLON = -145.0
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
station_inv = client.get_stations(starttime = starttime,
                                endtime = endtime, level = "response",
                                minlongitude = MINLON, maxlongitude = MAXLON,
                                minlatitude = MINLAT, maxlatitude = MAXLAT,
                                channel = 'BH*,HH*') #level="channel"

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




#######################  CREATE OUTPUT LIST ##################################
requ = [] # list of events with arrival times & dates, station IDs, network iDs
requ.append([]) #list of events (times) - index = 0
requ.append([]) #list of network codes, contains lists - index  = 1
requ.append([]) #list of stations' codes, contains lists - index = 2
requ.append([]) #list of arrival times, contains lists - index = 3
requ.append([]) #list of waveforms, contains list - index = 4




################## WRITE DATA IN LIST AND FETCH WAVEFORMS ################
ei = 0 #event index

# progressbar
bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
bi = 0 #progressbar index

for event in event_cat:
    ni = 0 #network index
    requ[0].append(event.origins[0].time)
    requ[1].append([])
    requ[2].append([])
    requ[3].append([])
    requ[4].append([])
    

    for network in station_inv:
        si = 0 #station index
        for station in network:
            
            # check if station was active at event time and if station has channels:
            if station.total_number_of_channels and station.is_active(time=requ[0][ei]):
                
                # calculate epicentral distance:
                ang_d = locations2degrees(event.origins[0].latitude, event.origins[0].longitude,
                                  station.latitude, station.longitude)
                
                # check if station is within defined epicentral distance limits:
                if min_epid < ang_d < max_epid:
                    requ[1][ei].append(station_inv[ni].code) #add network code
                    requ[2][ei].append(station_inv[ni][si].code) #add station code
                    requ[3][ei].append(requ[0][ei]+model.get_travel_times(source_depth_in_km=event.origins[0].depth/1000,
                                     distance_in_degree=ang_d,
                                     phase_list=["P"])[0].time) # calculate arrival times in UTC
                                                                #index [0] = P-arrival
                                                                #.time /.ray_param/.incident_angle
                    
                    # fetch waveforms; client is already set to IRIS
                    # download Brodband if available, else download HH
                    if station.get_contents()["channels"][0].find("BHZ"):
                        cha = "BH*"
                    else:
                        cha = "HH*"
                    
                    #if error FDSNNoDataException occurs (data is for some reason not available)
                    try:
                        requ[4][ei].append(client.get_waveforms(requ[1][ei][si], requ[2][ei][si], "*", cha, requ[3][ei][si]-20, requ[3][ei][si]+120, attach_response=True))
                    
                    ###### DEMEAN AND DETREND #########
                        requ[4][ei][si].detrend(type='demean')
                        
                    ############# TAPER ###############
                        requ[4][ei][si].taper(max_percentage=taper_perc,type=taper_type,max_length=None,side='both')
                    
                    ### REMOVE INSTRUMENT RESPONSE + convert to vel + SIMULATE ####
                        requ[4][ei][si].remove_response(inventory=station_inv,output='VEL',water_level=60)
                        requ[4][ei][si].remove_sensitivity(inventory=station_inv) #Should this step be done?
                    # simulate for another instrument like harvard (a stable good one)
                        requ[4][ei][si].simulate(paz_remove=None, paz_simulate=paz_sim, simulate_sensitivity=True) #simulate has a fuction paz_remove='self', which does not seem to work properly
                        
                    #  rotate from NEZ to radial, transverse, z
                        #requ[4][ei][si].rotate(method='NE->RT',inventory=station_inv)
                        # A back-azimuth has to be defined
                        # obpsy assumes the data to be properly alined alreadty ( has to be done before)
                    except header.FDSNException as e: 
                        logging.exception(e,'for (Event,Network,Station)',ei,ni,si)
                        requ[4][ei].append('exception')
                        
                bi = bi + 1
                bar.update(bi)
                si = si + 1
        ni = ni + 1
    ei = ei + 1


    


