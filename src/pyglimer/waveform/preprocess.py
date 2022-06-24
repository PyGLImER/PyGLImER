'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 19th May 2019 8:59:40 pm
Last Modified: Friday, 6th May 2022 11:58:54 am
'''

import fnmatch
import logging
import os
import shelve
import time
import itertools
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed, cpu_count
import obspy
from obspy import read, read_inventory, Stream, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from tqdm.std import tqdm

from pyglimer.utils.log import create_mpi_logger
from pyglimer.waveform.preprocessh5 import preprocessh5
from pyglimer.utils.signalproc import resample_or_decimate
from .errorhandler import redownload, redownload_statxml, \
    NoMatchingResponseHandler
from .qc import qcp, qcs
from .rotate import rotate_LQT_min, rotate_PSV
from ..rf.create import createRF
from ..utils.roundhalf import roundhalf
from ..utils.utils import dt_string, chunks


def preprocess(
    phase: str, rot: str, pol: str, taper_perc: float,
    event_cat: obspy.Catalog, model: obspy.taup.TauPyModel,
    taper_type: str, tz: int, ta: int, statloc: str, rawloc: str,
    preproloc: str, rfloc: str, deconmeth: str, hc_filt: float or None,
    saveasdf: bool = False, netrestr=None, statrestr=None,
        client: str = 'joblib', remove_response: bool = True):
    """
     Preprocesses waveforms to create receiver functions

        1. Clips waveform to the right length
        (tz before and ta after theorethical arrival.)
        2. Demean & Detrend
        3. Tapering
        4. Remove Instrument response, convert to velocity &
        simulate havard station.
        5. Rotation to NEZ and, subsequently, to RTZ.
        6. Compute SNR for highpass filtered waveforms
        (highpass f defined in qc.lowco).
        If SNR lower than in qc.SNR_criteria for all filters,
        rejects waveform.
        7. Write finished and filtered waveforms to folder
        specified in qc.outputloc.
        8. Write info file with shelf containing station,
        event and waveform information.

        Only starts after all waveforms of the event have been
        downloaded by download.py.
        (checked over the dynamic variables prepro_folder and tmp.folder)

        Saves preprocessed waveform files.
        Creates info file to save parameters.

    Parameters
    ----------
    phase : string
        "P" or "S"
    rot : string
        Coordinate system to cast seismogram in before deconvolution.
        Options are "RTZ", "LQT", or "PSS".
    pol : string
        "h" for Sh or "v" for Sv, only for PRFs.
    taper_perc : FLOAT
        Percentage to be tapered in beginning and at the end of waveforms.
    taper_type : STRING
        Taper type (see obspy documentation stream.taper).
    event_cat : event catalogue
        catalogue containing all events of waveforms.
    model : obspy.taup.TauPyModel
        1D velocity model to calculate arrival.
    tz : int
        time window before first arrival in seconds
    ta : int
        time window after first arrival in seconds
    logdir : string, optional
        Set the directory to where the download log is saved
    loglvl : int, optional
        Set logger to this level.

    Returns
    -------
    None.

    """
    ###########
    # logging
    logger = logging.getLogger("pyglimer.request.preprocess")
    logger.info('\n\n\n...Preprocessing initialiased...\n')

    # Logging for RF creation
    rflogger = logging.getLogger("pyglimer.request.preprocess.RF")

    #########

    if saveasdf:
        preprocessh5(
            phase, rot, pol, taper_perc, model, taper_type, tz, ta,
            rawloc, rfloc, deconmeth, hc_filt, netrestr,
            statrestr, logger, rflogger, client, event_cat, remove_response)
        return
    # else:
    # Here, we work with all available cores to speed things up
    # Split up event catalogue to mitigate the danger of data loss
    # Now the infodicts will be written in an even way
    # i.e. every 100 events

    # Number of cores is usally a power of 2 (therefore 128)
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

        if client.lower() == 'joblib':
            out = Parallel(n_jobs=-1, backend='multiprocessing')(
                delayed(__event_loop)(
                    phase, rot, pol, event, taper_perc,
                    taper_type, model, logger, rflogger, eh, tz,
                    ta, statloc, rawloc, preproloc, rfloc, deconmeth,
                    hc_filt, netrestr, statrestr, remove_response)
                for event in tqdm(evtcat))
        elif client.lower() == 'mpi':
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            psize = comm.Get_size()
            pmap = (np.arange(len(evtcat))*psize)/len(evtcat)
            pmap = pmap.astype(np.int32)
            ind = pmap == rank
            ind = np.arange(len(evtcat))[ind]

            # get new MPI compatible loggers
            logger = create_mpi_logger(logger, rank)
            rflogger = logging.getLogger("%s.RF" % logger.name)
            for ii in tqdm(ind):
                __event_loop(
                    phase, rot, pol, evtcat[ii], taper_perc, taper_type,
                    model, logger, rflogger, eh, tz, ta, statloc, rawloc,
                    preproloc, rfloc, deconmeth, hc_filt, netrestr, statrestr,
                    remove_response)

        # Use single core only
        elif client.lower() == 'single':
            out = []
            for event in tqdm(evtcat):
                out.append(
                    __event_loop(
                        phase, rot, pol, event, taper_perc,
                        taper_type, model, logger, rflogger, eh, tz,
                        ta, statloc, rawloc, preproloc, rfloc, deconmeth,
                        hc_filt, netrestr, statrestr, remove_response))

        else:
            raise NotImplementedError('Unknown client %s' % client)

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
                write_info(net, stat, masterdict[d], preproloc)
            except (ValueError, KeyError) as e:
                logger.exception([e, d])
                continue

    logger.info("Preprocessing finished.")


def __event_loop(
    phase: str, rot: str, pol: str, event: obspy.core.event.Event,
    taper_perc: float, taper_type: str, model: obspy.taup.TauPyModel,
    logger: logging.Logger, rflogger: logging.Logger, eh: bool, tz: float,
    ta: float, statloc: str, rawloc: str, preproloc: str, rfloc: str,
    deconmeth: str, hc_filt: float, netrestr: str, statrestr: str,
        remove_response: bool):
    """
    Loops over each event in the event catalogue
    """
    logger.info('Processing waveforms for event %s.' % event.resource_id)

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
        preproloc, 'by_event', '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))

    # make folder that will contain softlinks
    os.makedirs(by_event, exist_ok=True)

    # Folder, in which the preprocessing is actually happening
    prepro_folder = os.path.join(
        rawloc, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))

    try:  # If one event has no folder it interrupts else
        # Remove empty folders in the raw directory
        if not os.listdir(prepro_folder):
            os.rmdir(prepro_folder)
            return infolist  # It's important to return empty lists!
    except FileNotFoundError:
        # If we are not downloading that's entirely normal as
        # an earlier iteration just deletes empty directories
        pass
        return infolist

    # Preprocessing just for some stations?
    # Then skip files that should not be preprocessed
    if netrestr:
        pattern = '%s.%s*' % (netrestr, statrestr or '')
        files = fnmatch.filter(os.listdir(prepro_folder), pattern)
    else:
        files = os.listdir(prepro_folder)

    for filestr in files:
        try:
            info = __waveform_loop(
                phase, rot, pol, filestr, taper_perc, taper_type,
                model, origin_time, ot_fiss, evtlat, evtlon, depth,
                prepro_folder, event, logger, rflogger, by_event, eh, tz, ta,
                statloc, preproloc, rfloc, deconmeth, hc_filt, remove_response)
            infolist.append(info)
        except Exception as e:
            # Unhandled exceptions should not cause the loop to quit
            # processing one event
            logger.exception([filestr, e])
            continue

    return infolist


def __waveform_loop(
    phase: str, rot: str, pol: str, filestr: str, taper_perc: float,
    taper_type: str, model: obspy.taup.TauPyModel, origin_time: UTCDateTime,
    ot_fiss: str, evtlat: float, evtlon: float, depth: float,
    prepro_folder: str, event: obspy.core.event.event.Event,
    logger: logging.Logger, rflogger: logging.Logger, by_event, eh: bool,
    tz: float, ta: float, statloc: str, preproloc: str, rfloc: str,
        deconmeth: str, hc_filt: float, remove_response: bool):
    """
    Loops over each waveform for a specific event and a specific station
    """
    infodict = {}  # empty dictionary that will be dumped in a shelve file
    # at the end of the program

    start = time.time()
    # Open files that should be processed
    try:
        st = read(os.path.join(prepro_folder, filestr))
    except FileNotFoundError:  # file has not been downloaded yet
        return  # I will still want to have the RFs
    except Exception as e:  # Unknown erros
        logger.exception([prepro_folder, filestr, e])
        return
    station = st[0].stats.station
    network = st[0].stats.network

    # Location definitions
    outdir = os.path.join(preproloc, 'by_station', network, station)

    # Info file
    infof = os.path.join(outdir, 'info')

    ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]

    outf = os.path.join(outdir, '%s.%s.%s.mseed' % (network, station, ot_loc))

    statfile = os.path.join(statloc, '%s.%s.xml' % (network, station))

    # Create directory for preprocessed file
    os.makedirs(outdir, exist_ok=True)

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
                        "Station XML not available for station %s.%s" % (
                            network, station))

            # compute theoretical arrival
            distance, baz, _ = gps2dist_azimuth(
                station_inv[0][0].latitude, station_inv[0][0].longitude,
                evtlat, evtlon)
            distance = kilometer2degrees(distance/1000)

            # compute time of first arrival & ray parameter
            arrival = model.get_travel_times(
                source_depth_in_km=depth/1000, distance_in_degree=distance,
                phase_list=[phase])[0]
            rayp_s_deg = arrival.ray_param_sec_degree
            rayp = rayp_s_deg / 111319.9  # apparent slowness
            first_arrival = origin_time + arrival.time

            end = time.time()
            logger.debug("Before cut and resample")
            logger.debug(dt_string(end-start))

            # Check if step is already done
            if st[0].stats.sampling_rate != 10:
                st = __cut_resample(
                    st, logger, first_arrival, network, station, prepro_folder,
                    filestr, taper_perc, taper_type, eh, tz, ta)

            # Finalise preprocessing
            st, crit, infodict = __rotate_qc(
                phase, st, station_inv, network, station, baz,
                distance, outf, ot_fiss, event, evtlat, evtlon, depth,
                rayp_s_deg, first_arrival, infof, logger, infodict, by_event,
                eh, tz, statloc, remove_response)

        # Exceptions & logging

        except SNRError as e:  # QR rejections
            logger.debug(
                [filestr, "QC was not met, SNR ratios are", e])

            if __file_in_db(outdir, 'info.dat'):
                with shelve.open(infof, flag='r') as info:
                    if 'ot_all' not in info or ot_fiss not in info['ot_all']:
                        # Don't count rejected events twice
                        infodict.setdefault('ot_all', []).append(ot_fiss)

            else:
                infodict.setdefault('ot_all', []).append(ot_fiss)

            return infodict

        except StreamLengthError as e:
            logger.debug([filestr, e])
        # Everything else that might have gone wrong
        except Exception as e:
            logger.exception([prepro_folder, filestr, e])

        finally:
            end = time.time()
            logger.info(
                "File preprocessed.\n"
                + "Station %s.%s\n" % (network, station)
                + "Event %s" % event.resource_id)
            logger.debug(dt_string(end-start))

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
    if deconmeth and not __file_in_db(
        os.path.join(rfloc, network, station), '%s.%s.%s.sac' % (
            network, station, ot_loc)) and crit:

        # 21.04.2020 Second highcut filter
        if hc_filt:
            st.filter('lowpass', freq=hc_filt, zerophase=True, corners=2)

        start = time.time()

        ####
        try:
            # Rotate to LQT or PSS
            if rot == "LQT":
                st, ia = rotate_LQT_min(st, phase)
                # additional QC
                if ia < 5 or ia > 75:
                    crit = False
                    raise SNRError(
                        "The estimated incidence angle is "
                        + "unrealistic with %s degree." % str(ia))

            elif rot == "PSS":
                _, _, st = rotate_PSV(
                    station_inv[0][0][0].latitude,
                    station_inv[0][0][0].longitude,
                    rayp, st, phase)

            # Create RF object
            if phase[-1] == "S":
                trim = [40, 0]
                if distance >= 70:
                    trim[1] = ta - (-2*distance + 180)
                else:
                    trim[1] = ta - 40
            elif phase[-1] == "P":
                trim = False

            if not infodict:
                info = shelve.open(infof, flag='r')

                RF = createRF(
                    st, phase, pol=pol, info=info, trim=trim, method=deconmeth)

            else:
                RF = createRF(
                    st, phase, pol=pol, info=infodict, trim=trim,
                    method=deconmeth)

        # Write RF
            rfdir = os.path.join(rfloc, network, station)
            if pol == 'h' and phase == 'P':
                rfdir = rfdir + pol

            os.makedirs(rfdir, exist_ok=True)

            RF.write(os.path.join(
                rfdir, '%s.%s.%s.sac' % (network, station, ot_loc)),
                format='SAC')

            end = time.time()
            rflogger.info(
                "RF created.\n"
                + 'Station %s.%s\n' % (network, station)
                + 'Event: %s' % event.resource_id)
            rflogger.debug(dt_string(end-start))

        # Exception that occured in the RF creation
        # Usually don't happen
        except SNRError as e:
            rflogger.info(e)

        except Exception as e:
            print("RF creation failed")
            rflogger.exception([network, station, ot_loc, e])

        finally:
            if infodict:
                # Single-core case
                write_info(network, station, infodict, preproloc)

                # just to save RAM - not needed for single-core
                infodict = None
            else:  # The multicore case
                return infodict


def __cut_resample(
    st: obspy.Stream, logger: logging.Logger, first_arrival: UTCDateTime,
    network: str, station: str, prepro_folder: str, filestr: str,
    taper_perc: float, taper_type: str, eh: bool, tz: float,
        ta: float) -> obspy.Stream:
    """Cut and resample raw file. Will overwrite original raw"""

    start = time.time()

    # Trim TO RIGHT LENGTH BEFORE AND AFTER FIRST ARRIVAL  #
    # start and endtime of stream
    starttime = first_arrival - tz
    endtime = first_arrival + ta

    if st.count() < 3:
        if not eh:
            raise StreamLengthError(
                ["The stream contains less than three traces.", filestr])
        st = redownload(network, station, starttime, endtime, st)

    # Check one last time. If stream to short raise Exception
    if st.count() < 3:
        raise StreamLengthError(
            ["The stream contains less than 3 traces", filestr])

    # Change dtype
    for tr in st:
        np.require(tr.data, dtype=np.float64)
        tr.stats.mseed.encoding = 'FLOAT64'

    # trim to according length
    # Anti-Alias filtering is now done within the function below
    st = resample_or_decimate(st, 10)
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
        st.write(os.path.join(prepro_folder, filestr), format="MSEED")
    except ValueError:
        # Occurs for dtype=int32
        for tr in st:
            del tr.stats.mseed
        st.write(
            os.path.join(prepro_folder, filestr), format="MSEED")

    end = time.time()
    logger.debug("Unprocessed file rewritten")
    logger.debug(dt_string(end - start))

    return st


def __rotate_qc(
    phase: str, st: obspy.Stream, station_inv: obspy.Inventory, network: str,
    station: str, baz: float, distance: float, outf: str, ot_fiss: str,
    event: obspy.core.event.event.Event, evtlat: float, evtlon: float,
    depth: float, rayp_s_deg: float, first_arrival: UTCDateTime, infof: str,
    logger: logging.Logger, infodict: dict, by_event, eh: bool, tz: float,
        statloc: str, remove_response: bool) -> Tuple[Stream, bool, dict]:
    """REMOVE INSTRUMENT RESPONSE + convert to vel + SIMULATE
    Bugs occur here due to station inventories without response information
    Looks like the bulk downloader sometimes donwnloads
    station inventories without response files. I could fix that here by
    redownloading the response file (alike to the 3 traces problem)"""
    if remove_response:
        try:
            st.attach_response(station_inv)
            st.remove_response()
        except ValueError:
            # Occurs for "No matching response file found"

            if eh:
                station_inv, st = NoMatchingResponseHandler(
                    st, network, station, statloc)

            if not eh or not station_inv:
                raise ValueError(
                    ["No matching response file found for", network, station])

    st.rotate(method='->ZNE', inventory=station_inv)

    st.rotate(method='NE->RT', inventory=station_inv,
              back_azimuth=baz)
    st.normalize()

    # Sometimes streams contain more than 3 traces:
    if st.count() > 3:
        stream = {}
        for tr in st:
            stream[tr.stats.component] = tr
        if "Z" in stream:
            st = Stream([stream["Z"], stream["R"], stream["T"]])
        elif "3" in stream:
            st = Stream([stream["3"], stream["R"], stream["T"]])
        del stream

    # SNR CRITERIA
    dt = st[0].stats.delta  # sampling interval
    sampling_f = st[0].stats.sampling_rate

    if phase[-1] == "P":
        st, crit, f, noisemat = qcp(st, dt, sampling_f, tz)
        if not crit:
            infodict['dt'] = dt
            infodict['sampling_rate'] = sampling_f
            infodict['network'] = network
            infodict['station'] = station
            infodict['statlat'] = station_inv[0][0][0].latitude
            infodict['statlon'] = station_inv[0][0][0].longitude
            infodict['statel'] = station_inv[0][0][0].elevation
            raise SNRError('QC rejected %s' % np.array2string(noisemat))

    elif phase[-1] == "S":
        st, crit, f, noisemat = qcs(st, dt, sampling_f, tz)

        if not crit:
            infodict['dt'] = dt
            infodict['sampling_rate'] = sampling_f
            infodict['network'] = network
            infodict['station'] = station
            infodict['statlat'] = station_inv[0][0][0].latitude
            infodict['statlon'] = station_inv[0][0][0].longitude
            infodict['statel'] = station_inv[0][0][0].elevation
            raise SNRError('QC rejected %s' % np.array2string(noisemat))

    #      WRITE FILES     #
    try:
        st.write(outf, format="MSEED")
    except ValueError:
        # Occurs for weird mseed encodings
        for tr in st:
            del tr.stats.mseed
        st.write(outf, format="MSEED", encoding="ASCII")

    # create softlink
    try:
        os.symlink(outf, by_event)
    except FileExistsError:
        pass

    # WRITE AN INFO FILE
    # append_info: [key,value]
    append_inf = [
        ['magnitude', (
            event.preferred_magnitude() or event.magnitudes[0])['mag']],
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


def __file_in_db(loc: str, filename: str) -> bool:
    """Checks if file "filename" is already in location "loc"."""
    if os.path.isfile(os.path.join(loc, filename)):
        return True
    else:
        return False


def write_info(network: str, station: str, dictionary: dict, preproloc: str):
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
    loc = os.path.join(preproloc, 'by_station', network, station)
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


class StreamLengthError(Exception):
    """raised when stream has fewer than 3 components"""
    # Constructor method

    def __init__(self, value):
        self.value = value
    # __str__ display function

    def __str__(self):
        return repr(self.value)
