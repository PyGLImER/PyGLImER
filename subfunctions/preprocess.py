#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:31:05 2019

@author: pm
"""
# import config
import subfunctions.config as config
####### PREPROCESSING SUBFUNCTION ############
from obspy.core import *
from obspy.clients.iris import Client as iClient # To calculate angular distance and backazimuth
# careful with using clientas multiprocessing has a client class
from obspy.clients.fdsn import Client,header #web sevice
from obspy.core.event.base import *
from obspy.geodetics.base import locations2degrees #calculate angular distances
from obspy.taup import TauPyModel #arrival times in 1D v-model
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
import os
import logging
import time
import progressbar
import subprocess
#from multiprocessing import Process,Queue #multi-thread processing - preprocess while downloading
from pathlib import Path
from obspy.clients.fdsn import Client as Webclient #as Webclient #web sevice
from obspy import read
from obspy import read_inventory



#######################

def preprocess(taper_perc,taper_type,event_cat,webclient,model):
    # create output folder
    subprocess.call(["mkdir",config.outputloc])
    # needed for a station simulation
    station_simulate = webclient.get_stations(level = "response",channel = 'BH*',network = 'IU',station = 'HRV') #simulate one of the harvard instruments
    paz_sim = station_simulate[0][0][0].response #instrument response of the channel downloaded above
        
    #create dictionary with necassary data
    paz_sim = {"gain":paz_sim._get_overall_sensitivity_and_gain(output="VEL")[0],"sensitivity":paz_sim._get_overall_sensitivity_and_gain(output="VEL")[1], "poles":paz_sim.get_paz().poles,"zeros":paz_sim.get_paz().zeros}
    
    # Define class for backazimuth calculation
    iclient = iClient()
    for event in event_cat: #For each event
        # fetch event-data   
        origin_time = event.origins[0].time
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude
        prepro_folder = config.waveform+"/"+str(evtlat)+"_"+str(evtlon)+"_"+str(origin_time) #Folder, in which the preprocessing is actually happening

        while prepro_folder==config.folder or config.folder=="not_started":  #only start preprocessing after all waveforms for the first event have been downloaded
            print('preprocessing suspended, awaiting download')
            time.sleep(5)
        else:
            for file in os.listdir(prepro_folder): #preprocess each file per event
                st = read(prepro_folder+'/'+file)
                station = st[0].stats.station
                network = st[0].stats.network
                if not file_in_db(config.outputloc+'/'+network+'/'+station,file): #If the file hasn't been downloaded and preprocessed in an earlier iteration of the program
                    try: #There are sometimes bugs occuring here - an errorlog needs to be created to see for which files the preprocessing failed
                        subprocess.call(["mkdir",config.failloc]) #create folder for rejected streams
                        
                        if len(st) < 3: #If the stream does not contain all three components 
                            raise Exception("The stream contains less than three traces")
                        # create directory
                        subprocess.call(["mkdir",config.outputloc+'/'+network])
                        subprocess.call(["mkdir",config.outputloc+'/'+network+'/'+station])
                        # read station inventory
                        station_inv = read_inventory(config.statloc+"/"+network+"."+station+".xml")
                        
                        
                    ###### CLIP TO RIGHT LENGTH BEFORE AND AFTER FIRST ARRIVAL #####
                        # calculate backazimuth, distance
                        result = iclient.distaz(station_inv[0][0].latitude, stalon=station_inv[0][0].longitude, evtlat=evtlat,
                                                evtlon=evtlon)
                        
                        # compute time of first arrival
                        origin_time = event.origins[0].time
                        first_arrival = origin_time + model.get_travel_times(source_depth_in_km=event.origins[0].depth/1000,
                                         distance_in_degree=result['distance'],
                                         phase_list=["P"])[0].time
                        starttime = first_arrival-30
                        endtime = first_arrival+120
                                                                             
                        #clip to according length
                        st.trim(starttime=starttime,endtime=endtime)
                        
                    ###### DEMEAN AND DETREND #########
                        st.detrend(type='demean')
                        
                    ############# TAPER ###############
                        st.taper(max_percentage=taper_perc,type=taper_type,max_length=None,side='both')
                    
                    ### REMOVE INSTRUMENT RESPONSE + convert to vel + SIMULATE ####
                        st.remove_response(inventory=station_inv,output='VEL',water_level=60)
                        st.remove_sensitivity(inventory=station_inv) #Should this step be done?
                    # simulate for another instrument like harvard (a stable good one)
                        st.simulate(paz_remove=None, paz_simulate=paz_sim, simulate_sensitivity=True) #simulate has a fuction paz_remove='self', which does not seem to work properly
    
                    ##### rotate from NEZ to radial, transverse, z ######
                        st.rotate(method='->ZNE', inventory=station_inv) #If channeles weren't properly aligned it will be done here
                        st.rotate(method='NE->RT',inventory=station_inv,back_azimuth=result["backazimuth"])
                                    
                        # write to new file
                        st.write(config.outputloc+'/'+network+'/'+station+'/'+network+'.'+station+'.'+str(starttime)+'.mseed', format="MSEED") 
                        print("preprocessing successful") #test
                    except:
                        print("preprocessing failed") #test
                        logging.exception(file)
                        if not file_in_db(config.failloc,str(origin_time)+file):
                #            subprocess.call(["mkdir",config.failloc+'/'+str(evtlat)+"_"+str(evtlon)+"_"+str(origin_time)])
                            subprocess.call(["cp",prepro_folder+'/'+file,config.failloc+'/'+str(origin_time)+file]) #copy the mseed file for which the process failed
                    
# Check if file is already preprocessed          
def file_in_db(loc,filename):
    path = Path(loc+"/"+filename)
    if path.is_file():
        return True
    else:
        return False