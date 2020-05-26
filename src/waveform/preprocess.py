'''
Author: Peter Makus (peter.makus@student.uib.no)

Created: Tuesday, 19th May 2019 8:59:40 pm
Last Modified: Tuesday, 26th May 2020 07:30:55
'''

#!/usr/bin/env python3d
# -*- coding: utf-8 -*-

import fnmatch
import logging
import os
from pathlib import Path
import shelve
import subprocess
import time
import itertools
import warnings

import numpy as np
from joblib import Parallel, delayed, cpu_count
from obspy import read, read_inventory, Stream, UTCDateTime
from obspy.clients.iris import Client
from obspy.clients.fdsn import Client as Webclient
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

import config
from .errorhandler import redownload, redownload_statxml, \
    NoMatchingResponseHandler, NotLinearlyIndependentHandler
from ..constants import DEG2KM
from .qc import qcp, qcs
from .rotate import rotate_LQT_min, rotate_PSV, rotate_LQT
from ..rf.create import createRF
from ..utils.roundhalf import roundhalf
from ..utils.utils import dt_string, chunks


def preprocess(taper_perc, event_cat,model, taper_type="hann"):
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
    model : obspy.taup.TauPyModel
        1D velocity model to calculate arrival.

    Returns
    -------
    None.

    """
    ###########
    # logging
    logger = logging.Logger("src.waveform.preprocess")
    logger.setLevel(logging.WARNING)
    if config.debug:
        logger.setLevel(logging.DEBUG)

    # Create handler to the log
    fh = logging.FileHandler('logs/preprocess.log')
    fh.setLevel(logging.WARNING)
    if config.debug:
        fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Create Formatter
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)

    #########
    # Logging for RF creation
    rflogger = logging.getLogger("src.waveform.preprocess.RF")
    rflogger.setLevel(logging.WARNING)
    if config.debug:
        rflogger.setLevel(logging.DEBUG)

    # Create handler to the log
    fhrf = logging.FileHandler('logs/RF.log')
    fhrf.setLevel(logging.WARNING)
    if config.debug:
        fhrf.setLevel(logging.DEBUG)
    rflogger.addHandler(fhrf)

    # Create Formatter
    fmtrf = logging.Formatter(fmt="""%(asctime)s -
                                          %(levelname)s - %(message)s""")
    fhrf.setFormatter(fmtrf)

    # To get rid of the annoying and useless mseed warning
    # if not config.debug:
    #     warnings.filterwarnings("ignore", category=UserWarning)

    #########

    # needed for a station simulation - Harvard
    webclient = Webclient('IRIS')
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

    
    if not config.wavdownload:
        # Here, we work with all available cores to speed things up
        # Split up event catalogue to mitigate danger of data loss
        # Now the infodicts will be written in an even way
        # i.e. every 100 events
        
        # Number of cores is often a multiple of 8 (therefore 128)
        n_split = int(np.ceil(event_cat.count()/128))
        
        # All error handlers rely on download via IRIS webservice.
        # However, there is a maximum number for connections (3).
        # So I don't really want to flood everything with exceptions.
        if cpu_count() > 12:
            eh = False
        else:
            eh = True
        
        # Returns generator object with evtcats with each 100 events
        evtcats = chunks(event_cat, n_split)
        for evtcat in evtcats:
            out = Parallel(n_jobs=-1)(
                    delayed(__event_loop)(
                        event, taper_perc, taper_type, model,
                        paz_sim, logger, rflogger, eh)
                    for event in evtcat)

            # For simultaneous download, dicts are written
            #  while the processing is happening
            # (Not robust for multicore)

            # The multicore process returns a list of lists of dictionaries
            dictlist = list(itertools.chain.from_iterable(out))

            # 1. Write all dictionaries in a "masterdictionary", where the keys
            # are Network and station code. Then, extent these subdictionaries
            # with the new information from new dictionaries for the same
            # station.
            masterdict = {}

            for d in dictlist:
                if not d:  # Some of the dictionaries might be empty
                    continue
                try:
                    net = d['network']
                    stat = d['station']
                    key = net + '.' + stat
                except (ValueError, KeyError) as e:
                    logger.exception([e, d])
                    continue
                if key in masterdict:
                    for k in d:
                        if type(d[k]) == list:
                            masterdict[key].setdefault(k, []).extend(d[k])
                else:
                    masterdict[key] = d

            # Write dictionaries in the folder
            for d in masterdict:
                try:
                    net, stat = d.split('.')
                    write_info(net, stat, masterdict[d])
                except (ValueError, KeyError) as e:
                    logger.exception([e, d])
                    continue

    else:
        # It won't be faster to use several cores, when the download is running
        # So for stability's sake just one core and dicts are written at the
        # same time.
        eh = False  # Don't send additional requests to the FDSN servers!
        # Else it'll overload and no data will arrive
        Parallel(n_jobs=1)(
            delayed(__event_loop)(event, taper_perc, taper_type,
                                model, paz_sim, logger, rflogger, eh)
            for event in event_cat)

    print("Download and preprocessing finished.")


def __event_loop(event, taper_perc, taper_type, model,
                 paz_sim, logger, rflogger, eh):
    """
    Loops over each event in the event catalogue
    """
    # create list for what will later be the info files
    infolist = []

    # fetch event-data
    origin = (event.preferred_origin() or event.origins[0])
    origin_time = origin.time
    ot_fiss = UTCDateTime(origin_time).format_fissures()
    evtlat = origin.latitude
    evtlon = origin.longitude
    depth = origin.depth

    # Rounded for filenames
    ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]
    evtlat_loc = str(roundhalf(evtlat))
    evtlon_loc = str(roundhalf(evtlon))
    by_event = os.path.join(
        config.outputloc, 'by_event', ot_loc + '_' + evtlat_loc
        + '_' + evtlon_loc)

    # make folder that will contain softlinks
    if not Path(by_event).is_dir():
        subprocess.call(["mkdir", "-p", by_event])

    # Folder, in which the preprocessing is actually happening
    prepro_folder = os.path.join(
        config.waveform, ot_loc + '_' + evtlat_loc + '_' + evtlon_loc)

    while prepro_folder == config.folder or config.folder == "not_started" \
        and config.wavdownload:
        print('preprocessing suspended, awaiting download')
        time.sleep(2.5)

    try:  # If one event has no folder it interrupts else
        # Remove empty folders in the raw directory
        if not os.listdir(prepro_folder):
            subprocess.call(['rmdir', prepro_folder])
            return infolist  # It's important to return empty lists!
    except FileNotFoundError:
        # If we are not downloading that's entirely normal as
        # an earlier iteration just deletes empty directories
        if config.wavdownload:
            logger.exception([ot_fiss,
                             """Waveforms missing in database."""])
        return infolist

    # Preprocessing just for some stations?
    # Then skip files that should not be preprocessed

    if config.network:
        pattern = config.network + '.' + (config.station or '') + '*'
        files = fnmatch.filter(os.listdir(prepro_folder), pattern)
    else:
        files = os.listdir(prepro_folder)

    for file in files:
        try:
            info = __waveform_loop(file, taper_perc, taper_type, model,
                                paz_sim, origin_time, ot_fiss, evtlat, evtlon,
                                depth, prepro_folder, event, logger, rflogger,
                                by_event, eh)
            infolist.append(info)
        except Exception as e:
            # Unhandled exceptions should not cause the loop to quit
            # processing one event
            logger.exception([file, e])
            continue

    return infolist


def __waveform_loop(file, taper_perc, taper_type, model,
                    paz_sim, origin_time, ot_fiss, evtlat, evtlon,
                    depth, prepro_folder, event, logger, rflogger,
                    by_event, eh):
    """
    Loops over each waveform for a specific event and a specific station
    """
    infodict = {}  # empty dictionary that will be dumped in a shelve file
    # at the end of the program

    start = time.time()
    # Open files that should be processed
    try:
        st = read(os.path.join(prepro_folder, file))
    except FileNotFoundError:  # file has not been downloaded yet
        return  # I will still want to have the RFs
    except Exception as e:  # Unknown erros
        logger.exception([prepro_folder, file, e])
        return
    station = st[0].stats.station
    network = st[0].stats.network

    # Location definitions
    outdir = os.path.join(config.outputloc, 'by_station', network, station)

    # Info file
    infof = os.path.join(outdir, 'info')

    ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]

    outf = os.path.join(outdir, network+'.'+station+'.'+ot_loc+'.mseed')
    
    statfile = os.path.join(config.statloc, network + '.' + station + '.xml')

    # Create directory for preprocessed file
    if not Path(outdir).is_dir():
        subprocess.call(["mkdir", "-p", outdir])

    # If the file hasn't been downloaded and preprocessed
    # in an earlier iteration of the program
    if not __file_in_db(outdir, 'info.dat') or ot_fiss not in \
            shelve.open(infof, flag='r')['ot_all']:
        crit = False  # criterion to retain

        try:  # From here on, all exceptions are logged

            try:
                station_inv = read_inventory(statfile, format="STATIONXML")
            except FileNotFoundError:
                if eh:
                    station_inv = redownload_statxml(
                        st, network, station, statfile)
                else:
                    raise FileNotFoundError(
                        ["Station XML not available for station",
                          network, station])

            # compute theoretical arrival

            distance, baz, _ = gps2dist_azimuth(station_inv[0][0].latitude,
                                                station_inv[0][0].longitude,
                                                evtlat, evtlon)
            distance = kilometer2degrees(distance/1000)

            # compute time of first arrival & ray parameter
            arrival = model.get_travel_times(source_depth_in_km=depth / 1000,
                                             distance_in_degree=distance,
                                             phase_list=config.phase)[0]
            rayp_s_deg = arrival.ray_param_sec_degree
            rayp = rayp_s_deg / 111319.9  # apparent slowness
            first_arrival = origin_time + arrival.time

            end = time.time()
            logger.info("Before cut and resample")
            logger.info(dt_string(end-start))

            # Check if step is already done
            if st[0].stats.sampling_rate != 10:
                st = __cut_resample(st, logger, first_arrival, network,
                                    station, prepro_folder, file,
                                    taper_perc, taper_type, eh)

            # Finalise preprocessing
            st, crit, infodict = __rotate_qc(st, station_inv, network, station,
                                             paz_sim, baz, distance, outf,
                                             ot_fiss, event, evtlat, evtlon,
                                             depth, rayp_s_deg, first_arrival,
                                             infof, logger, infodict, by_event,
                                             eh)

        # Exceptions & logging

        except SNRError as e:  # QR rejections
            logger.debug([file, "QC was not met, SNR ratios are",
                         e])

            if __file_in_db(outdir, 'info.dat'):
                with shelve.open(infof, flag='r') as info:
                    if 'ot_all' not in info or ot_fiss not in info['ot_all']:
                        # Don't count rejected events twice
                        infodict.setdefault('ot_all', []).append(ot_fiss)

            else:
                infodict.setdefault('ot_all', []).append(ot_fiss)

            return infodict

        # Everything else that might have gone wrong
        except Exception as e:
            logger.exception([prepro_folder, file, e])

        finally:
            end = time.time()
            logger.info("File preprocessed.")
            logger.info(dt_string(end-start))

    else:  # The file was already processed

        # Did it pass QC?
        with shelve.open(infof, flag='r') as info:
            if "ot_ret" not in info or ot_fiss not in info["ot_ret"]:
                return
            else:
                crit = True
                st = read(outf)
                station_inv = read_inventory(statfile)
                j = info["ot_ret"].index(ot_fiss)
                rayp = info["rayp_s_deg"][j] / 111319.9
                distance = info["rdelta"][j]

#       CREATE RF   +    ROTATION TO PSS /   LQT    #

    # Check if RF was already computed and if it should be
    # computed at all, and if the waveform was retained (SNR)
    if config.decon_meth and not\
        __file_in_db(config.RF + '/' + network + '/' + station, network +
                     '.' + station + '.' + ot_loc + '.sac') and crit:

        # 21.04.2020 Second highcut filter
        if config.phase == "P":
            st.filter('lowpass', freq=1.5, zerophase=True, corners=2)
        elif config.phase == 'S':
            st.filter('lowpass', freq=0.25, zerophase=True, corners=2)  # freq=0.175Hz

        start = time.time()

        ####
        try:
            # Rotate to LQT or PSS
            if config.rot == "LQT_min":
                st, ia = rotate_LQT_min(st)
                # additional QC
                if ia < 5 or ia > 75:
                    crit = False
                    raise SNRError("""The estimated incidence angle is
                                   unrealistic with """ + str(ia) + 'degree.')

            if config.rot == "LQT":
                st, b = rotate_LQT(st)
                # addional QC
                if b > 0.75 or b < 1.5:
                    crit = False
                    raise SNRError("""The energy ratio between Q and L
                                   at theoretical arrival is too close
                                   to 1 with """ + str(b) + '.')

            elif config.rot == "PSS":
                avp, avs, st = rotate_PSV(
                    station_inv[0][0][0].latitude,
                    station_inv[0][0][0].longitude,
                    rayp, st, config.phase)

            # Create RF object
            if config.phase == "S":
                trim = [40, 0]
                if distance >= 70:
                    trim[1] = config.ta - (-2*distance + 180)
                else:
                    trim[1] = config.ta - 40
            elif config.phase == "P":
                trim = False

            if not infodict:
                info = shelve.open(infof, flag='r')

                RF = createRF(st, info=info, trim=trim)

            else:
                RF = createRF(st, info=infodict, trim=trim)

        # Write RF
            rfdir = os.path.join(config.RF, network, station)

            if not Path(rfdir).is_dir():
                subprocess.call(["mkdir", "-p", rfdir])

            RF.write(os.path.join(rfdir, network + '.' + station + '.' + ot_loc
                     + '.sac'), format='SAC')

            end = time.time()
            rflogger.info("RF created")
            rflogger.info(dt_string(end-start))

        # Exception that occured in the RF creation
        # Usually don't happen
        except SNRError as e:
            rflogger.info(e)
            # return infodict

        except Exception as e:
            print("RF creation failed")
            rflogger.exception([network, station, ot_loc, e])

        finally:
            if config.wavdownload and infodict:
                # Single-core case
                write_info(network, station, infodict)

                # just to save RAM - not needed for single-core
                infodict = None
            else:  # The multicore case
                return infodict

    # return infodict


def __cut_resample(st, logger, first_arrival, network, station,
                   prepro_folder, file, taper_perc, taper_type, eh):
    """Cut and resample raw file. Will overwrite original raw"""

    start = time.time()

    # Trim TO RIGHT LENGTH BEFORE AND AFTER FIRST ARRIVAL  #
    # start and endtime of stream
    starttime = first_arrival - config.tz
    endtime = first_arrival + config.ta

    if st.count() < 3:
        if not eh:
            raise ValueError(
                ["The stream contains less than three traces.", file])
        st = redownload(network, station, starttime, endtime, st)

    # Check one last time. If stream to short raise Exception
    if st.count() < 3:
        raise Exception("The stream contains less than 3 traces")

    # Change dtype
    for tr in st:
        np.require(tr.data, dtype=np.float64)
        tr.stats.mseed.encoding = 'FLOAT64'
    
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
    try:
        st.write(os.path.join(prepro_folder, file), format="MSEED")
    except ValueError:
        # Occurs for dtype=int32
        for tr in st:
            del tr.stats.mseed
        st.write(os.path.join(prepro_folder, file),
                  format="MSEED")


    end = time.time()
    logger.info("Unprocessed file rewritten")
    logger.info(dt_string(end - start))

    return st


def __rotate_qc(st, station_inv, network, station, paz_sim, baz,
                distance, outf, ot_fiss, event, evtlat, evtlon, depth,
                rayp_s_deg, first_arrival, infof, logger, infodict, by_event,
                eh):
    """REMOVE INSTRUMENT RESPONSE + convert to vel + SIMULATE
    Bugs occur here due to station inventories without response information
    Looks like the bulk downloader sometimes donwnloads
    station inventories without response files. I could fix that here by
    redownloading the response file (alike to the 3 traces problem)"""

    try:
        st.remove_response(inventory=station_inv, output='VEL',
                           water_level=60)
    except ValueError:
        # Occurs for "No matching response file found"
        try:
            if eh:
                station_inv, st = NoMatchingResponseHandler(
                    st, network, station)
            else:
                raise ValueError(["No matching response file found for",
                                  network, station])
        except UnboundLocalError:
            logger.exception(["Could not download response information.",
                             network, station])
            return

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
    # st = NotLinearlyIndependentHandler(st, network, station,
    #                                    st[0].stats.starttime,
    #                                    st[0].stats.endtime,
    #                                    station_inv, paz_sim)

    st.rotate(method='NE->RT', inventory=station_inv,
              back_azimuth=baz)
    st.normalize()

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
        st, crit, f, noisemat = qcp(st, dt, sampling_f)
        if not crit:
            infodict['dt'] = dt
            infodict['sampling_rate'] = sampling_f
            infodict['network'] = network
            infodict['station'] = station
            infodict['statlat'] = station_inv[0][0][0].latitude
            infodict['statlon'] = station_inv[0][0][0].longitude
            infodict['statel'] = station_inv[0][0][0].elevation
            raise SNRError(np.array2string(noisemat))

    elif config.phase == "S":
        st, crit, f, noisemat = qcs(st, dt, sampling_f)

        if not crit:
            infodict['dt'] = dt
            infodict['sampling_rate'] = sampling_f
            infodict['network'] = network
            infodict['station'] = station
            infodict['statlat'] = station_inv[0][0][0].latitude
            infodict['statlon'] = station_inv[0][0][0].longitude
            infodict['statel'] = station_inv[0][0][0].elevation
            raise SNRError(np.array2string(noisemat))

    #      WRITE FILES     #
    try:
        st.write(outf, format="MSEED")
    except ValueError:
        # Occurs for weird mseed encodings
        for tr in st:
            del tr.stats.mseed
        st.write(outf, format="MSEED", encoding="ASCII")

    # create softlink
    subprocess.call(["ln", "-s", outf, by_event])

    # WRITE AN INFO FILE
    # append_info: [key,value]
    append_inf = [['magnitude', (event.preferred_magnitude() or
                                 event.magnitudes[0])['mag']],
                  ['magnitude_type', (
                      event.preferred_magnitude() or event.magnitudes[0])[
                          'magnitude_type']],
                  ['evtlat', evtlat], ['evtlon', evtlon],
                  ['ot_ret', ot_fiss], ['ot_all', ot_fiss],
                  ['evt_depth', depth],
                  ['evt_id', event.get('resource_id')],
                  ['noisemat', noisemat],
                  ['co_f', f], ['npts', st[1].stats.npts],
                  ['rbaz', baz],
                  ['rdelta', distance],
                  ['rayp_s_deg', rayp_s_deg],
                  ['onset', first_arrival],
                  ['starttime', st[0].stats.starttime]]

    # Check if values are already in dict
    for key, value in append_inf:
        infodict.setdefault(key, []).append(value)

    infodict['dt'] = dt
    infodict['sampling_rate'] = sampling_f
    infodict['network'] = network
    infodict['station'] = station
    infodict['statlat'] = station_inv[0][0][0].latitude
    infodict['statlon'] = station_inv[0][0][0].longitude
    infodict['statel'] = station_inv[0][0][0].elevation

    logger.info("Stream accepted. Preprocessing successful")

    return st, crit, infodict


def __file_in_db(loc, filename):
    """Checks if file "filename" is already in location "loc"."""
    path = Path(loc+"/"+filename)
    if path.is_file():
        return True
    else:
        return False


def write_info(network, station, dictionary):
    """
    Writes information dictionary in shelve format in each of the station
    folders.

    Parameters
    ----------
    network : str
        Network Code.
    station : str
        Station code.
    dictionary : dict
        Dictionary containing the information.

    Returns
    -------
    None.

    """
    loc = os.path.join(config.outputloc, 'by_station', network, station)
    infof = os.path.join(loc, 'info')
    
    if not __file_in_db(loc, 'info.dat'):
        with shelve.open(os.path.join(loc, 'info'), writeback=True) as info:
            info.update(dictionary)
            info['num'] = len(info['ot_all'])
            if 'ot_ret' in info:
                info['numret'] = len(info['ot_ret'])
            info.sync()

    else:

        if 'ot_ret' in dictionary:
            append_inf = ['magnitude', 'magnitude_type', 'evtlat',
                          'evtlon', 'ot_ret', 'ot_all', 'evt_depth',
                          'evt_id', 'noisemat', 'co_f', 'npts', 'rbaz',
                          'rdelta', 'rayp_s_deg', 'onset', 'starttime']

            with shelve.open(infof, writeback=True) as info:
                for key in append_inf:
                    info.setdefault(key, []).extend(dictionary[key])
                info['num'] = len(info['ot_all'])
                info['numret'] = len(info['ot_ret'])
                info.sync()

        else:
            with shelve.open(infof, writeback=True) as info:
                info.setdefault('ot_all', []).extend(dictionary['ot_all'])
                info['num'] = len(info['ot_all'])
                info.sync()


# program-specific Exceptions
class SNRError(Exception):
    """raised when the SNR is too high"""
    # Constructor method

    def __init__(self, value):
        self.value = value
    # __str__ display function

    def __str__(self):
        return repr(self.value)
