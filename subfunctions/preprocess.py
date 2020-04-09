#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:31:05 2019

@author: pm
"""
import os
import logging
import time
import shelve
import numpy as np
# import progressbar
import subprocess
from pathlib import Path
from obspy import read
from obspy import read_inventory
from obspy.signal import filter
from obspy.core import Stream
from obspy.clients.iris import Client as iClient
# careful with using clientas multiprocessing has a client class
from obspy.core.utcdatetime import UTCDateTime

import config
from subfunctions.rotate import rotate_LQT_min, rotate_PSV, rotate_LQT
from subfunctions.createRF import createRF
from subfunctions.Errorhandler import redownload, redownload_statxml, \
    NoMatchingResponseHandler, NotLinearlyIndependentHandler


def preprocess(taper_perc, event_cat, webclient, model, taper_type="hann"):
    """
     Preprocesses waveforms to create receiver functions

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

        Saves preprocessed waveform files.
        Creates info file to save parameters.

    Parameters
    ----------
    taper_perc : FLOAT
        Percemtage to be tapered in beginning and at the end of waveforms.
    taper_type : STRING, optional
        DESCRIPTION. The default is "hann".
    event_cat : event catalogue
        catalogue containing all events of waveforms.
    webclient : obspy.clients.iris
        Used to fetch IU.HRV response file (for station simulation).
    model : obspy.taup.TauPyModel
        1D velocity model to calculate arrival.

    Returns
    -------
    None.

        """
    ###########
    # logging
    logger = logging.Logger("subfunctions.preprocess")
    logger.setLevel(logging.INFO)

    # Create handler to the log
    fh = logging.FileHandler('logs/preprocess.log')
    fh.setLevel(logging.WARNING)
    logger.addHandler(fh)

    # Create Formatter
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)

    #########

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

    __event_loop(event_cat, taper_perc, taper_type, webclient,
                 model, paz_sim, iclient, logger)
    print("Download and preprocessing finished.")


def __event_loop(event_cat, taper_perc, taper_type, webclient, model,
                 paz_sim, iclient, logger):
    """
    Loops over each event in the event catalogue
    """
    for event in event_cat:
        # fetch event-data
        origin = (event.preferred_origin() or event.origins[0])
        origin_time = origin.time
        ot_fiss = UTCDateTime(origin_time).format_fissures()
        evtlat = origin.latitude
        evtlon = origin.longitude
        depth = origin.depth

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
            try:  # If one event has no folder it interrupts else
                os.listdir(prepro_folder)
            except FileNotFoundError:
                logger.warning(""""All files have been processed. No
                                waveforms are for event""", ot_fiss, """Restart
                                download to obtain missing files
                                in database.""")
                continue
            __waveform_loop(taper_perc, taper_type, event_cat, webclient,
                            model, paz_sim, iclient, origin_time, ot_fiss,
                            evtlat, evtlon, depth, prepro_folder, event,
                            logger)


def __waveform_loop(taper_perc, taper_type, event_cat, webclient, model,
                    paz_sim, iclient, origin_time, ot_fiss, evtlat, evtlon,
                    depth, prepro_folder, event, logger):
    """
    Loops over each waveform for a specific event and a specific station
    """

    # loop over all files for event x
    for file in os.listdir(prepro_folder):
        try:
            st = read(prepro_folder+'/'+file)
        except FileNotFoundError:  # file has not been downloaded yet
            continue  # I will still want to have the RF
        except:  # Unknown erros
            logger.exception([prepro_folder, file])
            continue
        station = st[0].stats.station
        network = st[0].stats.network

        # Location definitions
        # Info file
        infof = config.outputloc+'/by_station/'+network+'/'+station + \
            '/'+'info'
        outf = config.outputloc+'/by_station/'+network+'/'+station+'/'\
            + network+'.'+station+'.'+ot_fiss+'.mseed'

        # Create directory for preprocessed file
        if not Path(config.outputloc + '/' + 'by_station/' +
                    network + '/' + station).is_dir():
            subprocess.call(["mkdir", "-p",
                             config.outputloc + '/' + 'by_station/' +
                             network + '/' + station])

    # If the file hasn't been downloaded and preprocessed
    # in an earlier iteration of the program
        if not __file_in_db(config.outputloc + '/by_station/' + network
                            + '/' + station, network + '.' + station +
                            '.' + ot_fiss + '.mseed'):
            crit = False  # criterion to retain

            try:  # From here on, all exceptions are logged
                try:
                    station_inv = read_inventory(config.statloc + "/"
                                                 + network + "." +
                                                 station + ".xml",
                                                 format="STATIONXML")
                except FileNotFoundError:
                    station_inv = redownload_statxml(st, network, station)

            # Trim TO RIGHT LENGTH BEFORE AND AFTER FIRST ARRIVAL  #

                result = iclient.distaz(station_inv[0][0].latitude,
                                        station_inv[0][0].longitude, evtlat,
                                        evtlon)

                # compute time of first arrival & ray parameter
                arrival = model.get_travel_times(source_depth_in_km=depth/1000,
                                                 distance_in_degree=result['distance'],
                                                 phase_list=config.phase)[0]
                rayp_s_deg = arrival.ray_param_sec_degree
                rayp = rayp_s_deg/111319.9  # apparent slowness
                first_arrival = origin_time + arrival.time
                # start and endtime of stream
                starttime = first_arrival - config.tz
                endtime = first_arrival + config.ta

                if st.count() < 3:
                    st = redownload(network, station, starttime, endtime, st)

                # Check one last time. If stream to short raise Exception
                if st.count() < 3:
                    raise Exception("The stream contains less than 3 traces")

                # trim to according length
                # Anti-Alias
                st.filter(type="lowpass", freq=4.95, zerophase=True, corners=2)
                st.resample(10)  # resample streams with 10Hz sampling rate
                st.trim(starttime=starttime, endtime=endtime)
                # After trimming length has to be checked again (recording may
                # be empty now)
                if st.count() < 3:
                    raise Exception("The stream contains less than 3 traces")

                # DEMEAN AND DETREND #
                st.detrend(type='demean')

                # TAPER #
                st.taper(max_percentage=taper_perc, type=taper_type,
                         max_length=None, side='both')

                # Write trimmed and resampled files into raw-file folder
                # to save space
                st.write(prepro_folder+'/'+file, format="mseed")

# REMOVE INSTRUMENT RESPONSE + convert to vel + SIMULATE #
# Bugs occur here due to station inventories without response information
# Looks like the bulk downloader sometimes donwloads
# station inventories without response files. I could fix that here by
# redowloading the response file (alike to the 3 traces problem)
                try:
                    st.remove_response(inventory=station_inv, output='VEL',
                                       water_level=60)
                except ValueError:
                    # Occurs for "No matching response file found"
                    station_inv, st = NoMatchingResponseHandler(st, network,
                                                                station)

                st.remove_sensitivity(inventory=station_inv)

            # simulate for another instrument like harvard (a stable good one)
                st.simulate(paz_remove=None, paz_simulate=paz_sim,
                            simulate_sensitivity=True)

    #       ROTATION      #
                # try:
                # If channeles weren't properly aligned
                st.rotate(method='->ZNE', inventory=station_inv)
# Error: The directions are not linearly independent,
# there doesn't seem to be a fix for this
                # except ValueError:
                st = NotLinearlyIndependentHandler(st, network, station,
                                                   starttime, endtime,
                                                   station_inv, paz_sim)

                st.rotate(method='NE->RT', inventory=station_inv,
                          back_azimuth=result["backazimuth"])
                st.normalize

                # Sometimes streams contain more than 3 traces:
                if st.count() > 3:
                    stream = {}
                    for tr in st:
                        stream[tr.stats.channel[2]] = tr
                    if "Z" in stream:
                        st = Stream([stream["Z"], stream["R"], stream["T"]])
                    elif "3" in stream:
                        st = Stream([stream["3"], stream["R"], stream["T"]])
                    del stream

            # SNR CRITERIA
                dt = st[0].stats.delta  # sampling interval
                sampling_f = st[0].stats.sampling_rate

                if not config.QC:
                    crit, f, noisemat = True, None, None
                elif config.phase == "P":
                    st, crit, f, noisemat = QC_P(st, dt, sampling_f)
                    if not crit:
                        raise SNRError(noisemat)
                elif config.phase == "S":
                    st, crit, f, noisemat = QC_S(st, dt, sampling_f)
                    # crit, f, noisemat = None, None, None
                    if not crit:
                        raise SNRError(noisemat)

                #      WRITE FILES     #
                st.write(outf, format="MSEED")
                # create softlink
                subprocess.call(["ln", "-s", '../../by_station/'+network+'/' +
                                 station+'/'+network+'.'+station+'.'+ot_fiss
                                 + '.mseed', outf])

            # WRITE AN INFO FILE
                # append_info: [key,value]
                append_inf = [['magnitude', (event.preferred_magnitude() or
                               event.magnitudes[0])['mag']],
                              ['magnitude_type', (event.preferred_magnitude()
                                                  or event.magnitudes[0])['magnitude_type']],
                              ['evtlat', evtlat], ['evtlon', evtlon],
                              ['ot_ret', ot_fiss], ['ot_all', ot_fiss],
                              ['evt_depth', depth],
                              ['evt_id', event.get('resource_id')],
                              ['noisemat', noisemat],
                              ['co_f', f], ['npts', st[1].stats.npts],
                              ['rbaz', result["backazimuth"]],
                              ['rdelta', result["distance"]],
                              ['rayp_s_deg', rayp_s_deg],
                              ["onset", first_arrival],
                              ['starttime', st[0].stats.starttime]]

                with shelve.open(infof, writeback=True) as info:
                    # Check if values are already in dict
                    for key, value in append_inf:
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

                print("Stream accepted. Preprocessing successful")

            # Exceptions & logging

            except SNRError:  # QR rejections
                logger.info("QC was not met, SNR ratios are", noisemat, file)
                with shelve.open(infof, writeback=True) as info:
                    if 'ot_all' not in info or ot_fiss not in info['ot_all']:
                        # Don't count rejected events twice
                        info.setdefault('ot_all', []).append(ot_fiss)
                        info['num'] = len(info['ot_all'])
                        info.sync()

            # Everythng else that might have gone wrong
            except:
                print("Stream rejected")  # test
                logger.exception([prepro_folder, file])

        else:  # The file was already processed and passed the crit
            crit = True
            st = read(outf)

    #       CREATE RF   +    ROTATION TO PSS /   LQT    #

        # Check if RF was already computed and if it should be
        # computed at all, and if the waveform was retained (SNR)
        if config.decon_meth and not\
            __file_in_db(config.RF + '/' + network + '/' + station, network +
                         '.' + station + '.' + ot_fiss + '.sac') and crit:

            # Logging for RF creation
            RF_logger = logging.getLogger("subfunctions.preprocess.RF")
            RF_logger.setLevel(logging.INFO)

            # Create handler to the log
            fhrf = logging.FileHandler('logs/RF.log')
            fhrf.setLevel(logging.WARNING)
            RF_logger.addHandler(fhrf)

            # Create Formatter
            fmtrf = logging.Formatter(fmt="""%(asctime)s -
                                      %(levelname)s - %(message)s""")
            fhrf.setFormatter(fmtrf)

            ####
            try:
                # Rotate to LQT or PSS
                if config.rot == "LQT_min":
                    st, ia = rotate_LQT_min(st)
                    # addional QC
                    if ia < 5 or ia > 75:
                        crit = False
                        raise SNRError("""The estimated incidence angle is
                                       unrealistic with """, ia, 'degree.')

                if config.rot == "LQT":
                    st, b = rotate_LQT(st)
                    # addional QC
                    if b > 0.75 or b < 1.5:
                        crit = False
                        raise SNRError("""The energy ratio between Q and L
                                       at theoretical arrival is too close
                                       to 1 with """, b, '.')

                elif config.rot == "PSS":
                    avp, avs, st = rotate_PSV(
                        station_inv[0][0][0].latitude,
                        station_inv[0][0][0].longitude,
                        rayp, st)

                # Create RF object
                with shelve.open(infof) as info:
                    i = info["starttime"].index(st[0].stats.starttime)
                    # Make a trim dependt on epicentral distance
                    trim = [40, 0]
                    if info["rdelta"][i] >= 70:
                        trim[1] = config.ta - (-2*info["rdelta"][i] + 180)
                    else:
                        trim[1] = config.ta - 40
                    RF = createRF(st, info=info, trim=trim)

            # Write RF
                if not Path(config.RF + '/' + network + '/' + station
                            ).is_dir():
                    subprocess.call(["mkdir", "-p",
                                     config.RF + '/' + network + '/' +
                                     station])

                RF.write(config.RF + '/' + network + '/' + station +
                         '/' + network + '.' + station + '.' + ot_fiss
                         + '.sac', format="SAC")

                # copy info files
                subprocess.call(["cp", infof + ".dir",
                                 config.RF + '/' + network + '/' +
                                 station + '/'])
                subprocess.call(["cp", infof + ".bak",
                                 config.RF + '/' + network + '/' +
                                 station + '/'])
                subprocess.call(["cp", infof+".dat",
                                 config.RF + '/' + network + '/' +
                                 station + '/'])

            # Exception that occured in the RF creation
            # Usually don't happen
            except SNRError as e:
                RF_logger.info(e)
                continue

            except:
                print("RF creation failed")
                RF_logger.exception([network, station, ot_fiss])


def QC_P(st, dt, sampling_f):
    """
    Quality control for the downloaded waveforms that are used to create
    PRFS. Works with various filters and SNR criteria

    Parameters
    ----------
    st : '~obspy.Stream'
        Input stream.
    dt : FLOAT
        Sampling interval [s].
    sampling_f : FLOAT
        Sampling frequency (Hz).

    Returns
    -------
    st : '~obspy.Stream'
        Output stream. If stream was accepted, then this will contain the
        filtered stream, filtered with the broadest accepted filter.
    crit : BOOL
        True if stream was accepted, false if it wasn't.
    f : FLOAT
        Last used low-cut frequency. If crit=True, this will be the frequency
        for which the stream was accepted.
    noisemat : np.array
        SNR values in form of a matrix. Rows represent the different filters
        and columns the different criteria.

    """
    # Create stream dict to identify channels
    stream = {}
    for tr in st:
        stream[tr.stats.channel[2]] = tr.data
    ptn1 = round(5/dt)
    ptn2 = round((config.tz-5)/dt)
    nptn = ptn2-ptn1+1
    # First part of the signal
    pts1 = round(config.tz/dt)
    pts2 = round((config.tz+7.5)/dt)
    npts = pts2-pts1+1
    # Second part of the signal
    ptp1 = round((config.tz+15)/dt)
    ptp2 = round((config.tz+22.5)/dt)
    nptp = ptp2-ptp1+1

    # Then, I'll have to filter in the for-loop
    # and calculate the SNR

    # matrix to save SNR
    noisemat = np.zeros((len(config.lowco),
                         3), dtype=float)

    for ii, f in enumerate(config.lowco):
        ftcomp = filter.bandpass(stream["T"], f,
                                 4.99, sampling_f, corners=2, zerophase=True)
        frcomp = filter.bandpass(stream["R"], f,
                                 4.99, sampling_f, corners=2, zerophase=True)
        if "Z" in stream:
            fzcomp = filter.bandpass(stream["Z"], f, 4.99,
                                     sampling_f, corners=2, zerophase=True)
        elif "3" in stream:
            fzcomp = filter.bandpass(stream["3"], f, 4.99,
                                     sampling_f, corners=2, zerophase=True)

        # Compute the SNR for given frequency bands
        snrr = (sum(np.square(frcomp[pts1:pts2]))/npts)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        snrr2 = (sum(np.square(frcomp[ptp1:ptp2]))/nptp)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        snrz = (sum(np.square(fzcomp[pts1:pts2]))/npts)\
            / (sum(np.square(fzcomp[ptn1:ptn2]))/nptn)

        # Reject or accept traces depending on their SNR
        # #1: snr1 > 10 (30-37.5s, near P)
        # snr2/snr1 < 1 (where snr2 is 45-52.5 s, in the coda of P)
        # note: - next possibility might be to remove those events that
        # generate high snr between 200-250 s

        noisemat[ii, 0] = snrr
        noisemat[ii, 1] = snrr2/snrr
        noisemat[ii, 2] = snrz

        if snrr > config.SNR_criteria[0] and\
            snrr2/snrr < config.SNR_criteria[1] and\
                snrz > config.SNR_criteria[2]:  # accept
            crit = True

            # overwrite the old traces with the sucessfully filtered ones
            for tr in st:
                if tr.stats.channel[2] == "R":
                    tr.data = frcomp
                elif tr.stats.channel[2] == "Z"\
                    or tr.stats.channel[2] == "3" or\
                        tr.stats.channel[2] == "Z":
                    tr.data = fzcomp
                elif tr.stats.channel[2] == "T":
                    tr.data = ftcomp
            break  # waveform is accepted
        else:
            crit = False
    return st, crit, f, noisemat


def QC_S(st, dt, sampling_f):
    """
    Quality control for waveforms that are used to produce SRF. In contrast
    to the ones used for PRF this is a very rigid criterion and will reject
    >95% of the waveforms.

    Parameters
    ----------
    st : '~obspy.Stream'
        Input stream.
    dt : FLOAT
        Sampling interval [s].
    sampling_f : FLOAT
        Sampling frequency (Hz).

    Returns
    -------
    st : '~obspy.Stream'
        Output stream. If stream was accepted, then this will contain the
        filtered stream, filtered with the broadest accepted filter.
    crit : BOOL
        True if stream was accepted, false if it wasn't.
    f : FLOAT
        Last used high-cut frequency. If crit=True, this will be the frequency
        for which the stream was accepted.
    noisemat : np.array
        SNR values in form of a matrix. Rows represent the different filters
        and columns the different criteria.

    """
    # Create stream dict to identify channels
    stream = {}
    for tr in st:
        stream[tr.stats.channel[2]] = tr.data

    ptn1 = round(5/dt)  # Hopefully relatively silent time
    ptn2 = round((config.tz-5)/dt)
    nptn = ptn2-ptn1
    # First part of the signal
    pts1 = round(config.tz/dt)  # theoretical arrival time
    pts2 = round((config.tz+10)/dt)
    npts = pts2-pts1
    # Second part of the signal
    ptp1 = round((config.tz+10)/dt)
    ptp2 = round((config.tz+25)/dt)
    nptp = ptp2-ptp1
    # part where the Sp converted arrival should be strong
    ptc1 = round((config.tz-50)/dt)
    ptc2 = round((config.tz-10)/dt)
    nptc = ptc2-ptc1

    # filter
    # matrix to save SNR
    noisemat = np.zeros((len(config.highco), 3), dtype=float)

    # Filter
    # At some point, I might also want to consider to change the lowcof
    for ii, hf in enumerate(config.highco):
        ftcomp = filter.bandpass(stream["T"], config.lowcoS,
                                 hf, sampling_f, corners=2, zerophase=True)
        frcomp = filter.bandpass(stream["R"], config.lowcoS,
                                 hf, sampling_f, corners=2, zerophase=True)
        if "Z" in stream:
            fzcomp = filter.bandpass(stream["Z"], config.lowcoS, hf,
                                     sampling_f, corners=2, zerophase=True)
        elif "3" in stream:
            fzcomp = filter.bandpass(stream["3"], config.lowcoS, hf,
                                     sampling_f, corners=2, zerophase=True)

        # Compute the SNR for given frequency bands
        # strength of primary arrival
        snrr = (sum(np.square(frcomp[pts1:pts2]))/npts)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        # how spiky is the arrival?
        snrr2 = (sum(np.square(frcomp[ptp1:ptp2]))/nptp)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        # horizontal vs vertical
        # snrz = (sum(np.square(frcomp[pts1:pts2]))/npts)\
        #     / (sum(np.square(fzcomp[pts1:pts2]))/npts)
        snrc = (sum(np.square(frcomp[ptc1:ptc2]))/nptc)\
            / sum(np.square(fzcomp[ptc1:ptc2])/nptc)

        noisemat[ii, 0] = snrr
        noisemat[ii, 1] = snrr2/snrr
        noisemat[ii, 2] = snrc

        # accept if
        if snrr > config.SNR_criteria[0] and\
            snrr2/snrr < config.SNR_criteria[1] and\
                snrc < config.SNR_criteria[2]:
            # This bit didn't give good results
            # max(frcomp) == max(frcomp[round((config.tz-2)/dt):
            #                           round((config.tz+10)/dt)]):
            crit = True

            # overwrite the old traces with the sucessfully filtered ones
            for tr in st:
                if tr.stats.channel[2] == "R":
                    tr.data = frcomp
                elif tr.stats.channel[2] == "Z"\
                    or tr.stats.channel[2] == "3" or\
                        tr.stats.channel[2] == "Z":
                    tr.data = fzcomp
                elif tr.stats.channel[2] == "T":
                    tr.data = ftcomp
            break  # waveform is accepted
        else:
            crit = False
    return st, crit, hf, noisemat


def __file_in_db(loc, filename):
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
