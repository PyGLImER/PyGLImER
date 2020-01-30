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
from obspy.core.utcdatetime import UTCDateTime
import os
import logging
import time
#import progressbar
import subprocess
#from multiprocessing import Process,Queue #multi-thread processing - preprocess while downloading
from pathlib import Path
from obspy import read
from obspy import read_inventory
from obspy.signal import filter
#from importlib import import_module
import shelve
import numpy as np



#######################




def preprocess(taper_perc,taper_type,event_cat,webclient,model):
    """ Preprocesses waveforms to create receiver functions with the following steps:

        1. Clips waveform to the right length (config.tz before and config.ta after theorethical arrival.)
        2. Demean & Detrend
        3. Tapering
        4. Remove Instrument response, convert to velocity & simulate havard station.
        5. Rotation to NEZ and, subsequently, to RTZ.
        6. Compute SNR for highpass filtered waveforms (highpass f defined in config.lowco).
         If SNR lower than in config.SNR_criteria for all filters, it rejects waveform.
        7. Write finished and filtered waveforms to folder specified in config.outputloc.
        8. Write info file with shelf containing station, event and waveform information.
        
        Only starts after all waveforms of the event have been downloaded by download.py.
        (checked over the dynamic variables prepro_folder and config.folder)
        
        INPUT:
            taper_perc: taper percentage (config)
            taper_type: taper_type (config)
            event_cat: event catalogue
            webclient: webclient that is used to fetch response of Havard station
            model: velocity model to calculate arrival time (config)
            
            only in config file:
            outputloc: output location for preprocessed files
            failloc: location for files that were not preprocessed
            waveform: folder, in which the raw data is dumped
            folder: folder, in which  the download is happening
            statloc: location of station inventory
            tz: clip time before theoretical arrival
            ta: clip time after theoretical arrival
            lowco: list with low cut-off frequencies for SNR check
            SNR_criteria: SNR criteria for accepting/rejecting stream
            
         OUTPUT:
             saves preprocessed waveform files
             copies failed files
             creates info file to save parameters.     
        """
    # needed for a station simulation
    station_simulate = webclient.get_stations(level = "response",channel = 'BH*',network = 'IU',station = 'HRV') #simulate one of the harvard instruments
    paz_sim = station_simulate[0][0][0].response #instrument response of the channel downloaded above
        
    #create dictionary with necassary data
    paz_sim = {"gain":paz_sim._get_overall_sensitivity_and_gain(output="VEL")[0],"sensitivity":paz_sim._get_overall_sensitivity_and_gain(output="VEL")[1], "poles":paz_sim.get_paz().poles,"zeros":paz_sim.get_paz().zeros}
    
    # Define class for backazimuth calculation
    iclient = iClient()
    for event in event_cat: #For each event
        
        # fetch event-data   
        # is it ok to always use the first origin ([0])? - THere seems to be always only one
        origin_time = event.origins[0].time
        ot_fiss = UTCDateTime(origin_time).format_fissures()
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude
        depth = event.origins[0].depth
        
        # make folder that will contain softlinks
        if not Path(config.outputloc+'/'+'by_event/'+ot_fiss).is_dir():
            subprocess.call(["mkdir","-p",config.outputloc+'/'+'by_event/'+ot_fiss])
        
        # # Not necassary, there is never more than one gien
        # magnitude = []
        # magnitude_type = []
        # for mag in event.magnitudes: #append all the different magnitudes
        #     magnitude.append(mag.mag)
        #     magnitude_type.append(mag.magnitude_type)

        prepro_folder = config.waveform+"/"+ot_fiss+'_'+str(evtlat)+"_"+str(evtlon) #Folder, in which the preprocessing is actually happening

        while prepro_folder==config.folder or config.folder=="not_started":  #only start preprocessing after all waveforms for the first event have been downloaded
            print('preprocessing suspended, awaiting download')
            time.sleep(5)
        else:
            for file in os.listdir(prepro_folder): #preprocess each file per event
                st = read(prepro_folder+'/'+file)
                station = st[0].stats.station
                network = st[0].stats.network
                
                
                # Create directory for preprocessed file
                if not Path(config.outputloc+'/'+'by_station/'+network+'/'+station).is_dir():
                    subprocess.call(["mkdir","-p",config.outputloc+'/'+'by_station/'+network+'/'+station])
                    
                
                # Has there already an event been recorded on this station?
                # This is important so the lists can be created that I will append to later
                # Also, it might save a bit of computational power
                old_info = file_in_db(config.outputloc+'/'+'by_station/'+network+'/'+station,'info.dat') #either True or False
                info = shelve.open(config.outputloc+'/'+'by_station/'+network+'/'+station+'/'+'info',writeback=True)
                if not old_info:
                    info['events_all'] = []
                info['events_all'].append(ot_fiss)
                
                if not file_in_db(config.outputloc+'/by_station/'+network+'/'+station,file): #If the file hasn't been downloaded and preprocessed in an earlier iteration of the program
                    try:
                        
                        # I don't copy the failed files anymore as it's a waste of harddisk space
                        # if not Path(config.failloc).is_dir():
                        #     subprocess.call(["mkdir","-p",config.failloc]) #create folder for rejected streams
                        
                            
                        # read station inventory
                        station_inv = read_inventory(config.statloc+"/"+network+"."+station+".xml", format = "STATIONXML")
                        
                        
                    ###### CLIP TO RIGHT LENGTH BEFORE AND AFTER FIRST ARRIVAL #####
                        # calculate backazimuth, distance
                        result = iclient.distaz(station_inv[0][0].latitude, stalon=station_inv[0][0].longitude, evtlat=evtlat,
                                                evtlon=evtlon)
                        
                        # compute time of first arrival
                        first_arrival = origin_time + model.get_travel_times(source_depth_in_km=depth/1000,
                                         distance_in_degree=result['distance'],
                                         phase_list=["P"])[0].time
                        starttime = first_arrival-config.tz
                        endtime = first_arrival+config.ta
                        
                   ##### Check if stream has three channels   
                        # This might be a bit cumbersome, but should work.
                        # Right now its trying two times to redownload the data
                        if len(st) < 3:
                            client = Client("IRIS") # I have not found out how to find the original Client that has been used for the download
                            # to solve that I could implement another try / except loop in the except part down there, very cumbersome though
                            for iii in range(3):
                                try:
                                    if len(st) < 3: #If the stream does not contain all three components 
                                        raise Exception
                                    else:
                                        break
                                except:
                                    st = client.get_waveforms(network,station,'*',st[0].stats.channel[0:2]+'*',starttime,endtime)
                                #finally: # Check one last time. If stream to short raise Exception
                        if len(st) <3:
                            raise Exception("The stream contains less than three traces")
                                                                             
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
                        st.rotate(method='->ZNE', inventory=station_inv) #If channeles weren't properly aligned it will be done here, I doubt that's even necassary. Assume the next method does that as well
                        st.rotate(method='NE->RT',inventory=station_inv,back_azimuth=result["backazimuth"])
                        
                    ##### SNR CRITERIA #####
                        # according to the original Matlab script, we will be workng with several bandpass (low-cut) filters to evaluate the SNR
                        # Then, the SNR will be evaluated for each of the traces and the traces will be accepted or rejected depending on their SNR.
                        
                        #Find channel type - obspy uses the order T-R-Z
                        #create stream dictionary
                        
                        # This way is actually not robust
                        # stream = {
                        #         st[0].stats.channel[2]: st[0].data,
                        #         st[1].stats.channel[2]: st[1].data,
                        #         st[2].stats.channel[2]: st[2].data}
                        # 25.08.2019
                        # 
                        
                        dt = st[0].stats.delta #sampling interval
                        sampling_f = st[0].stats.sampling_rate
                        
                        # noise
                        ptn1=round(5/dt)
                        ptn2=round((config.tz-5)/dt)
                        nptn=ptn2-ptn1+1
                        #First part of the signal
                        pts1=round(config.tz/dt)
                        pts2=round((config.tz+7.5)/dt)
                        npts=pts2-pts1+1;
                        #Second part of the signal
                        ptp1=round((config.tz+15)/dt)
                        ptp2=round((config.tz+22.5)/dt)
                        nptp=ptp2-ptp1+1
                        # The original script does check the length of the traces here and pads them with zeroes if necassary.
                        # However,  would not consider this necassary as the bulkdownload should have already filtered anything that is shorter then wanted.
                        # Then, I'll have to filter in the for-loop and calculate the SNR
                        crit = False #criterium for accepting
                        
                        noisemat = np.zeros((len(config.lowco),3),dtype=float) #matrix to save SNR
                        
                        for ii,f in enumerate(config.lowco):
                            ftcomp = filter.bandpass(st[0], f,2.5, sampling_f, corners=4, zerophase=True)
                            frcomp = filter.bandpass(st[1], f,2.5, sampling_f, corners=4, zerophase=True)
                            fzcomp = filter.bandpass(st[2], f,2.5, sampling_f, corners=4, zerophase=True)      
                            
                            # Compute the SNR for given frequency bands
                            snrr = (sum(np.square(frcomp[pts1:pts2]))/npts)/(sum(np.square(frcomp[ptn1:ptn2]))/nptn)
                            snrr2 = (sum(np.square(frcomp[ptp1:ptp2]))/nptp)/(sum(np.square(frcomp[ptn1:ptn2]))/nptn)
                            snrz = (sum(np.square(fzcomp[pts1:pts2]))/npts)/(sum(np.square(fzcomp[ptn1:ptn2]))/nptn)
#                            snrz2 = (sum(np.square(fzcomp[ptp1:ptp2]))/nptp)/(sum(np.square(fzcomp[ptn1:ptn2]))/nptn)
                            #snrz2 is not used (yet), so I commented it
                            
                            # Reject or accept traces depending on their SNR
                            # #1: snr1 > 10 (30-37.5s, near P)
                            # snr2/snr1 < 1 (where snr2 is 45-52.5 s, in the coda of P)
                            # note: - next possibility might be to remove those events that
                            # generate high snr between 200-250 s
                            
                            noisemat[ii,0] = snrr
                            noisemat[ii,1] = snrr2/snrr
                            noisemat[ii,2] = snrz
                            
                            if snrr > config.SNR_criteria[0] and snrr2/snrr < config.SNR_criteria[1] and snrz > config.SNR_criteria[2]: #accept
                                crit = True
                                # overwrite the old traces with the sucessfully filtered ones
                                st[0].data = ftcomp
                                st[1].data = frcomp
                                st[2].data = fzcomp
                                break #waveform is accepted no further tests needed
                        
                        if not crit:
                            raise SNRError(noisemat)
                        
                        
                 ####### FIND VS ACCORDING TO BOSTOCK, RONDENAY (1999) #########
                 # suspended for now
#                        tpn1 = round((config.tz-10)/dt)
#                        tpn2 = round(config.tz/dt)
#                        tpn3 = round((config.tz+10)/dt)
#                        
                        # . . Integration in the frequency domain
                        # I'll have to find out how to proceed from here on
#                        drcmp=integr(rcmp,dt)
#                        dtcmp=integr(tcmp,dt)
#                        dzcmp=integr(zcmp,dt)
                        
                        # write to new file
                        # create directory
                        

                        st.write(config.outputloc+'/by_station/'+network+'/'+station+'/'+network+'.'+station+'.'+ot_fiss+'.mseed', format="MSEED") 
                        subprocess.call(["ln","-s",'../../by_station/'+network+'/'+station+'/'+network+'.'+station+'.'+ot_fiss+'.mseed', config.outputloc+'/by_event/'+ot_fiss+'/'+network+station])
                    ############# WRITE AN INFO FILE #################
                        

                        if not old_info:
                            # station specific parameters
                            info['dt'] = dt
                            info['sampling_rate'] = sampling_f
                            info['network'] = network
                            info['station'] = station
                            info['statlat'] = station_inv[0][0][0].latitude
                            info['statlon'] = station_inv[0][0][0].longitude
                            info['statel'] = station_inv[0][0][0].elevation
                            
                            # create list for event/waveform parameters
                            info['magnitude'] = []
                            info['magnitude_type'] = []
                            info['evtlat'] = []
                            info['evtlon'] = []
                            info['origin_time'] = []
                            info['evt_depth'] = []
                            info['noisemat'] = []
                            info['lowco_f'] = []
                            info['npts'] = []
                            info['numret'] = 0
                            info['T'] = []
                            info['R'] = []
                            info['Z'] = []
                            info['rbaz'] = []
                            info['rdelta'] = []
                            info['events_ret'] = []
                        # Append information for this event
                        info['magnitude'].append(event.magnitudes[0].mag)
                        info['magnitude_type'].append(event.magnitudes[0].magnitude_type)
                        info['evtlat'].append(evtlat)
                        info['evtlon'].append(evtlon)
                        info['origin_time'].append(ot_fiss)
                        info['evt_depth'] = depth
                        info['noisemat'].append(noisemat)
                        info['lowco_f'].append(f)
                        info['npts'].append(st[1].stats.npts)
                        info['numret'] = info['numret']+1
                        info['T'].append(st[0].data)
                        info['R'].append(st[1].data)
                        info['Z'].append(st[2].data)
                        info['rbaz'].append(result["backazimuth"])
                        info['rdelta'].append(result["distance"])
                        info['events_ret'].append(ot_fiss)
                        
                        
                        
                        
#                        # When writing pay attention that variables aren't 1 overwritten 2: written twice 3: One can append variables to arrays
#                        finfo.writelines([dt, network, station, ])
                        print("Stream accepted. Preprocessing successful") #
                    except SNRError as e: #These should not be in the log as they mask real errors
                        print("Stream rejected - SNR too low with",e) #test
                        # if not file_in_db(config.failloc,file):
                        #     subprocess.call(["cp",prepro_folder+'/'+file,config.failloc+'/'+file])
                        
                    except:
                        print("Stream rejected") #test
                        logging.exception(file)
                        # if not file_in_db(config.failloc,file):
                        #     subprocess.call(["cp",prepro_folder+'/'+file,config.failloc+'/'+file]) #move the mseed file for which the process failed
                            # The mseed file that I copy is the raw one (as downloaded from webservvice)
                            # Does that make sense?
                    finally: #Is executed regardless if an exception occurs or not
                        info.close()
                    
# Check if file is already preprocessed          
def file_in_db(loc,filename):
    """Checks if file "filename" is already in location "loc"."""
    path = Path(loc+"/"+filename)
    if path.is_file():
        return True
    else:
        return False



# program-specific Exceptions 
class SNRError(Exception):
   """raised when the SNR is too high"""
   # Constructor method
   def __init__(self, value):
      self.value = value
   # __str__ display function
   def __str__(self):
      return(repr(self.value))
# class SNRError(Error):
#     """raised when SNR is too low
    
#         Attributes:
#         expression -- input expression in which the error occurred
#         message -- explanation of the error
#     """
    
#     def __init__(self,message):
#         self.message = message

# class LengthError(Error):
#     """raised when stream contains less than three traces
    
#         Attributes:
#         expression -- input expression in which the error occurred
#         message -- explanation of the error
#     """
    
#     def __init__(self,message):
#         self.message = message
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    