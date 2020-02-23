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
from subfunctions.SignalProcessingTB import rotate_LQT, rotate_PSV
from subfunctions.deconvolve import it, damped
from subfunctions.createRF import createRF


def preprocess(taper_perc, taper_type, event_cat, webclient, model):
    """ Preprocesses waveforms to create receiver functions

        1. Clips waveform to the right length
        (config.tz before and config.ta after theorethical arrival.)
        2. Demean & Detrend
        3. Tapering
        4. Remove Instrument response, convert to velocity &
        simulate havard station.
        5. Rotation to NEZ and, subsequently, to RTZ.
        6. Compute SNR for highpass filtered waveforms
        (highpass f defined in config.lowco).
        If SNR lower than in config.SNR_criteria for all filters,
        rejects waveform.
        7. Write finished and filtered waveforms to folder
        specified in config.outputloc.
        8. Write info file with shelf containing station,
        event and waveform information.

        Only starts after all waveforms of the event have been
        downloaded by download.py.
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
    # needed for a station simulation - Harvard
    station_simulate = webclient.get_stations(level="response",
                                              channel='BH*', network='IU',
                                              station='HRV')
    # instrument response of the channel downloaded above
    paz_sim = station_simulate[0][0][0].response
    
    # create dictionary with necassary data
    paz_sim = {"gain": paz_sim._get_overall_sensitivity_and_gain
               (output="VEL")[0],
               "sensitivity": paz_sim._get_overall_sensitivity_and_gain
                   (output="VEL")[1],
               "poles": paz_sim.get_paz().poles,
               "zeros": paz_sim.get_paz().zeros}

    # Define class for backazimuth calculation
    iclient = iClient()
    for event in event_cat:  # For each event

        # fetch event-data
        origin_time = event.origins[0].time
        ot_fiss = UTCDateTime(origin_time).format_fissures()
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude
        depth = event.origins[0].depth

        # make folder that will contain softlinks
        if not Path(config.outputloc+'/'+'by_event/'+ot_fiss).is_dir():
            subprocess.call(["mkdir", "-p",
                             config.outputloc+'/'+'by_event/'+ot_fiss])


# Folder, in which the preprocessing is actually happening
        prepro_folder = config.waveform + "/" + ot_fiss + '_' + str(evtlat) +\
            "_" + str(evtlon)
# only start preprocessing after waveforms for foregoing are downloaded
        while prepro_folder == config.folder or config.folder == "not_started":
            print('preprocessing suspended, awaiting download')
            time.sleep(5)
        else:
            # preprocess each file per event
            for file in os.listdir(prepro_folder):
                try:
                    st = read(prepro_folder+'/'+file)
                except FileNotFoundError:  # file has not been downloaded yet
                    break  # I will still want to have the RF
                station = st[0].stats.station
                network = st[0].stats.network

                # Create directory for preprocessed file
                if not Path\
                    (config.outputloc+'/'+'by_station/'+network+'/'+station)\
                        .is_dir():
                    subprocess.call(["mkdir", "-p",
                                     config.outputloc + '/' + 'by_station/' +
                                         network + '/' + station])

        # If the file hasn't been downloaded and preprocessed
        # in an earlier iteration of the program
                if not file_in_db(config.outputloc + '/by_station/' + network
                                  + '/' + station, network + '.' + station +
                                  '.' + ot_fiss + '.mseed'):
                    crit = False  # criterion to retain

                    try:
                        # I don't copy the failed files anymore as it's
                        # a waste of harddisk space
                        # if not Path(config.failloc).is_dir():
                        #     subprocess.call(["mkdir","-p",config.failloc])

                        # read station inventory
                        try:
                            station_inv = read_inventory(config.statloc + "/"
                                                         + network + "." +
                                                         station + ".xml",
                                                         format="STATIONXML")
                        except FileNotFoundError:
                            for c in config.re_clients:
                                try:
                                    client = Client(c)
                                    station_inv = client.get_stations(
                                        level="response",
                                        channel=st[0].stats.channel[0:2] + '*',
                                        network=network, station=station)
                                    # write the new, working stationxml file
                                    station_inv.write(config.statloc + "/"
                                                      + network + "." +
                                                      station + ".xml",
                                                      format="STATIONXML")
                                    break
                                except (header.FDSNNoDataException,
                                        header.FDSNException):  # wrong client
                                    pass

                            
                        
                    ###### CLIP TO RIGHT LENGTH BEFORE AND AFTER FIRST ARRIVAL #####
                        # 06.02.2020: Change function so that it also returns the ray-parameters (later for rotation to LQT)
                        # calculate backazimuth, distance
                        result = iclient.distaz(station_inv[0][0].latitude, stalon=station_inv[0][0].longitude, evtlat=evtlat,
                                                evtlon=evtlon)
                        
                        # compute time of first arrival & ray parameter
                        arrival = model.get_ray_paths(source_depth_in_km=depth/1000,
                                         distance_in_degree=result['distance'],
                                         phase_list=config.phase)[0]
                        # ray parameter in s/m
                        rayp = arrival.ray_param/(111319.9*result['distance'])
                        first_arrival = origin_time + arrival.time
                        # first_arrival = origin_time + model.get_travel_times(source_depth_in_km=depth/1000,
                        #                  distance_in_degree=result['distance'],
                        #                  phase_list=config.phase)[0].time
                        starttime = first_arrival - config.tz
                        endtime = first_arrival + config.ta
                        
                        
                   ##### Check if stream has three channels   
                        # This might be a bit cumbersome, but should work.
                        # Right now its trying three times to redownload the data for each client
                        # That might be quite slow. I have however not found out how to find out which client is the one I need
                        
                        if len(st) < 3:
                            for c in config.re_clients:
                                client = Client(c) # I have not found out how to find the original Client that has been used for the download
                                # to solve that I could implement another try / except loop in the except part down there, very cumbersome though
                                for iii in range(3):
                                    try:
                                        if len(st) < 3: #If the stream does not contain all three components 
                                            raise Exception
                                        else:
                                            break
                                    except:
                                        try:
                                            st = client.get_waveforms(network,station,'*',st[0].stats.channel[0:2]+'*',starttime,endtime)
                                        except (header.FDSNNoDataException,ValueError): #wrong client chosen
                                            break
                                if len(st) == 3:
                                    break
                                
                                        
                                #finally: # Check one last time. If stream to short raise Exception
                        if len(st) < 3:
                            raise Exception("The stream contains less than three traces")
                                                                             
                        # clip to according length
                        st.trim(starttime = starttime, endtime = endtime)
                        
                    ###### DEMEAN AND DETREND #########
                        st.detrend(type='demean')
                        
                    ############# TAPER ###############
                        st.taper(max_percentage=taper_perc,type=taper_type,max_length=None,side='both')
                        
                    ##### Write clipped and resampled files into raw-file folder to save space ####
                        st.resample(10) # resample streams with 10Hz sampling rate
                        st.write(prepro_folder+'/'+file,format="mseed") #write file

                    ### REMOVE INSTRUMENT RESPONSE + convert to vel + SIMULATE ####

                        # maybe I don't need to do these two steps, I will just want an output in velocity
                        # Bugs occur here due to station inventories without response information
                        # Looks like the bulk downloader sometimes donwloads station inventories without response files
                        # I could fix that here by redowloading the response file (alike to the 3 traces problem)
                        try:
                            st.remove_response(inventory=station_inv,output='VEL',water_level=60)
                            
                        except ValueError: #Occurs for "No matching response file found"
                            print("The stationxml has to be redownloaded")
                            #01.02.20: This bug occurs because the channels are named 1 & 2 instead of N and E
                            
                            # for c in config.re_clients:
                            #     try:
                            #         client = Client(c)
                            #         station_inv = client.get_stations(level = "response",channel = st[0].stats.channel,network = network,station = station) #simulate one of the harvard instruments
                            #         station_inv.write(config.statloc+"/"+network+"."+station+".xml", format = "STATIONXML") #write the new, working stationxml file
                            #         st.remove_response(inventory=station_inv,output='VEL',water_level=60)
                            #     except (header.FDSNNoDataException, header.FDSNException): #wrong client
                            #         pass
                            #     except ValueError: #the response file doesn't seem to be available at all
                            #         break
                            
                            # Turns out the problem is in the stream not in the response file, so just redownload the stream
                            # Not sure how necassary the whole thing is after changing download.py
                            # But better safe than sorry, leave it inside - doesn't take any computation power
                            # as long as it isn't called.
                            for c in config.re_clients:
                                try:
                                    client = Client(c)
                                    #st = client.get_waveforms(network,station,'*',st[0].stats.channel[0:2]+'*',starttime,endtime)
                                    station_inv = client.get_stations(level = "response",channel = st[0].stats.channel[0:2]+'*',network = network,station = station) #simulate one of the harvard instruments
                                    st.remove_response(inventory=station_inv,output='VEL',water_level=60)
                                    station_inv.write(config.statloc+"/"+network+"."+station+".xml", format = "STATIONXML") #write the new, working stationxml file
                                    if len(st) <3:
                                        raise Exception("The stream contains less than three traces")
                                    break
                                except (header.FDSNNoDataException, header.FDSNException): #wrong client
                                    pass
                                except ValueError: #the response file doesn't seem to be available at all
                                    break
                                
                                
                        st.remove_sensitivity(inventory=station_inv) #Should this step be done?
                        # st.remove_response(inventory=station_simulate,output='VEL',water_level=60)
                        # st.remove_sensitivity(inventory=station_simulate) #Should this step be done?
                    # simulate for another instrument like harvard (a stable good one)
                        st.simulate(paz_remove=None, paz_simulate=paz_sim, simulate_sensitivity=True) #simulate has a fuction paz_remove='self', which does not seem to work properly


############################ ROTATION ########################################
                    ##### rotate from NEZ to radial, transverse, z ######
                        try:
                            # If channeles weren't properly aligned
                            st.rotate(method ='->ZNE', inventory = station_inv)
                        except ValueError: #Error: The directions are not linearly independent, for some reason redownloading seems to help here
                            for c in config.re_clients:
                                client = Client(c) # I have not found out how to find the original Client that has been used for the download
                                # to solve that I could implement another try / except loop in the except part down there, very cumbersome though
                                while len(st) < 3:
                                    try:
                                        st = client.get_waveforms(network,station,'*',st[0].stats.channel[0:2]+'*',starttime,endtime)
                                        st.remove_response(inventory=station_inv,output='VEL',water_level=60)
                                        st.remove_sensitivity(inventory=station_inv)
                                        st.simulate(paz_remove=None, paz_simulate=paz_sim, simulate_sensitivity=True)
                                        st.rotate(method='->ZNE', inventory=station_inv)
                                    except (header.FDSNNoDataException,
                                            header.FDSNException,
                                            ValueError): #wrong client chosen
                                        break

                        st.rotate(method='NE->RT',inventory=station_inv,back_azimuth=result["backazimuth"])
                        st.normalize
                        
                        # Sometimes streams contain more than 3 traces:
                        if st.count() > 3:
                            stream = {}
                            for tr in st:
                                stream[tr.stats.channel[2]] = tr
                            if "Z" in stream:
                                st = Stream(stream["Z"], stream["R"],
                                            stream["T"])
                            elif "3" in stream:
                                st = Stream(stream["3"], stream["R"],
                                            stream["T"])
                            del stream

                        # Rotate to LQT
                        if config.rot == "LQT":
                            st = rotate_LQT(st)
                            # channel labels
                            P = "L"
                            S = "Q"
                            T = "T"
                        elif config.rot == "PSS":
                            avp, avs, st = rotate_PSV(
                                station_inv[0][0][0].latitude,
                                station_inv[0][0][0].longitude,
                                rayp, st)
                            # channel labels
                            P = "P"
                            S = "V"
                            T = "H"
                        else:
                            P = "Z"
                            S = "R"
                            T = "T"

                    ##### SNR CRITERIA #####
                        # 04.02.2020: 
                        # Now, as I start implementing S receiver functions I will have to figure criteria to put here

                        # according to the original Matlab script, we will be workng with several bandpass (low-cut) filters to evaluate the SNR
                        # Then, the SNR will be evaluated for each of the traces and the traces will be accepted or rejected depending on their SNR.
                        
                        #create stream dictionary
                        # stream = {
                        #         st[0].stats.channel[2]: st[0].data,
                        #         st[1].stats.channel[2]: st[1].data,
                        #         st[2].stats.channel[2]: st[2].data}

                        # 25.08.2019
                        # 
                        
                        dt = st[0].stats.delta #sampling interval
                        sampling_f = st[0].stats.sampling_rate
                        
                        if config.phase == "P":
                        # noise
                            # Create stream dict
                            stream = {}
                            for tr in st:
                                stream[tr.stats.channel[2]] = tr.data
                            ptn1 = round(5/dt)
                            ptn2 = round((config.tz-5)/dt)
                            nptn = ptn2-ptn1+1
                            # First part of the signal
                            pts1 = round(config.tz/dt)
                            pts2 = round((config.tz+7.5)/dt)
                            npts = pts2-pts1+1;
                            # Second part of the signal
                            ptp1 = round((config.tz+15)/dt)
                            ptp2 = round((config.tz+22.5)/dt)
                            nptp = ptp2-ptp1+1

                            # Then, I'll have to filter in the for-loop
                            # and calculate the SNR

                            # matrix to save SNR
                            noisemat = np.zeros((len(config.lowco),
                                                 3), dtype=float)
                            
                            for ii,f in enumerate(config.lowco):
                                ftcomp = filter.bandpass(stream[T], f,4.99, sampling_f, corners=4, zerophase=True)
                                frcomp = filter.bandpass(stream[S], f,4.99, sampling_f, corners=4, zerophase=True)
                                if "Z" in stream:
                                    fzcomp = filter.bandpass(stream["Z"], f,4.99, sampling_f, corners=4, zerophase=True)      
                                elif "3" in stream:
                                    fzcomp = filter.bandpass(stream["3"], f,4.99, sampling_f, corners=4, zerophase=True)
                                else:
                                    fzcomp = filter.bandpass(stream[P], f,4.99, sampling_f, corners=4, zerophase=True)
                                # Compute the SNR for given frequency bands
                                snrr = (sum(np.square(frcomp[pts1:pts2]))/npts)/(sum(np.square(frcomp[ptn1:ptn2]))/nptn)
                                snrr2 = (sum(np.square(frcomp[ptp1:ptp2]))/nptp)/(sum(np.square(frcomp[ptn1:ptn2]))/nptn)
                                snrz = (sum(np.square(fzcomp[pts1:pts2]))/npts)/(sum(np.square(fzcomp[ptn1:ptn2]))/nptn)
                               # snrz2 = (sum(np.square(fzcomp[ptp1:ptp2]))/nptp)/(sum(np.square(fzcomp[ptn1:ptn2]))/nptn)
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
                                    # st[0].data = ftcomp
                                    # st[1].data = frcomp
                                    # st[2].data = fzcomp
                                for tr in st:
                                    if tr.stats.channel[2] == S:
                                        tr.data = frcomp
                                    elif tr.stats.channel[2] == "Z"\
                                        or tr.stats.channel[2] == "3" or\
                                            tr.stats.channel[2] == P:
                                        tr.data = fzcomp
                                    elif tr.stats.channel[2] == T:
                                        tr.data = ftcomp
                                        
                                break  # waveform is accepted
                            
                            if not crit:
                                raise SNRError(noisemat)
                                
                        elif config.phase=="S": #This is where later the criteria for SP receiver functions have to be defined
                            noisemat = None #just so it doesn't rise an error
                            f = None
                                
                        # Write all the data to the stats of the traces
                        # it does not save these. They only stay in RAM
                        # for tr in st:
                        #     tr.stats.onset = first_arrival
                        #     tr.stats.rayp = rayp
                        #     tr.stats.ba = result["backazimuth"]
                        #     tr.stats.ot = ot_fiss
                        # write to new file

                        st.write(config.outputloc+'/by_station/'+network+'/'+station+'/'+network+'.'+station+'.'+ot_fiss+'.mseed', format="MSEED") 
                        subprocess.call(["ln","-s",'../../by_station/'+network+'/'+station+'/'+network+'.'+station+'.'+ot_fiss+'.mseed', config.outputloc+'/by_event/'+ot_fiss+'/'+network+station])
                    
                    ############# WRITE AN INFO FILE #################
                        append_inf = [['magnitude',event.magnitudes[0].mag],['magnitude_type',event.magnitudes[0].magnitude_type],
                                      ['evtlat',evtlat],['evtlon',evtlon],['ot_ret',ot_fiss],['ot_all',ot_fiss],
                                      ['evt_depth',depth],['noisemat',noisemat],['lowco_f',f],['npts',st[1].stats.npts],
                                      ['T',st[0].data],['R',st[1].data],["Z",st[2].data],['rbaz',result["backazimuth"]],
                                      ['rdelta', result["distance"]], ['rayp', rayp], ["onset", first_arrival]]
                        #append_info: put first key for dic then value. [key,value]
                        
                        with shelve.open(config.outputloc+'/'+'by_station/'+network+'/'+station+'/'+'info',writeback=True) as info:
                            # if not old_info:
                            #     # station specific parameters
                            
                            #Check if values are already in dic
                            for key,value in append_inf:
                                info.setdefault(key, []).append(value)
                            
                            info['num'] = len(info['ot_all'])
                            info['numret'] = len(info['ot_ret'])
                            info['dt'] = dt
                            info['sampling_rate'] = sampling_f
                            info['network'] = network
                            info['station'] = station
                            info['statlat'] = station_inv[0][0][0].latitude
                            info['statlon'] = station_inv[0][0][0].longitude
                            info['statel'] = station_inv[0][0][0].elevation
                        
                            info.sync()
                            
                            # I should create an extra script for the RF
                            # creation, where I can decide if I rather want to
                            # use mseed files in preprocessed!



#                        # When writing pay attention that variables aren't 1 overwritten 2: written twice 3: One can append variables to arrays
#                        finfo.writelines([dt, network, station, ])
                        print("Stream accepted. Preprocessing successful") #
                    except SNRError: #These should not be in the log as they mask real errors
                        #print("Stream rejected - SNR too low with",e) #test
                        with shelve.open(config.outputloc+'/'+'by_station/'+network+'/'+station+'/'+'info',writeback=True) as info:
                            if not 'ot_all' in info or not ot_fiss in info['ot_all']: #Don't count rejected events double
                                info.setdefault('ot_all', []).append(ot_fiss)
                                info['num'] = len(info['ot_all'])
                                info.sync()
                        # if not file_in_db(config.failloc,file):
                        #     subprocess.call(["cp",prepro_folder+'/'+file,config.failloc+'/'+file])
                        
                    except:
                        print("Stream rejected") #test
                        logging.exception([prepro_folder,file])
                        # if not file_in_db(config.failloc,file):
                        #     subprocess.call(["cp",prepro_folder+'/'+file,config.failloc+'/'+file]) #move the mseed file for which the process failed
                            # The mseed file that I copy is the raw one (as downloaded from webservvice)
                            # Does that make sense?
                    #finally: #Is executed regardless if an exception occurs or not

                else:  # The file was already processed and passed the crit
                    crit = True
                    st = read(config.outputloc + '/by_station/' + network
                                  + '/' + station + '/' + network + '.' +
                                  station + '.' + ot_fiss + '.mseed')
                        
    ################### create RF ##################
                # Check if RF was already computed and if it should be
                # computed at all, and if the waveform was retained (SNR)
                if config.decon_meth and not\
                    file_in_db(config.RF + '/' + network + '/' + station,
                               network + '.' + station + '.' + ot_fiss
                         + '.mseed') and crit == True:
                    RF = createRF(st, .1)  # dt is always .1

                # Write RF
                    if not Path(config.RF + '/' + network + '/' + station
                                ).is_dir():
                        subprocess.call(["mkdir", "-p",
                         config.RF + '/' + network + '/' + station])
                    RF.write(config.RF + '/' + network + '/' + station +
                             '/' +network + '.' + station + '.' + ot_fiss
                             + '.mseed', format="MSEED")                        
                    # copy info files
                    subprocess.call(["cp", prepro_folder + "/info*",
                                     config.RF + '/' + network + '/' +
                                     station])

# Check if file is already preprocessed          
def file_in_db(loc, filename):
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
