'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 19th May 2019 8:59:40 pm
Last Modified: Friday, 4th July 2025 01:40:49 pm
'''

import fnmatch
import logging
import os
import shelve
import time
import itertools
from typing import Tuple

import numpy as np
import h5py
from joblib import Parallel, delayed, cpu_count
import obspy
from obspy import read, read_inventory, Stream, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from tqdm.std import tqdm

from pyglimer import tmp
from pyglimer.database.rfh5 import RFDataBase, hierarchy, h5_FMTSTR
from pyglimer.utils.log import create_mpi_logger
from pyglimer.waveform.preprocessh5 import preprocessh5
from pyglimer.utils.signalproc import resample_or_decimate
from .errorhandler import redownload, redownload_statxml, \
    NoMatchingResponseHandler
from .preproc_utils import EventContext, PreprocessConfig, SNRError, \
    StreamLengthError, append_event_info, rotate_and_snr_qc, set_station_info
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
        client: str = 'joblib', remove_response: bool = True,
        clients=None, n_jobs: int = -1):
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
    clients : list of str, optional
        Names of the FDSN data centres that the waveforms were downloaded
        from (e.g. ``['IRIS', 'ORFEUS']``). The error handlers use this
        list to redownload incomplete waveforms. ``None`` (the default)
        keeps the previously registered clients, if any.
    n_jobs : int, optional
        Number of parallel jobs for the ``'joblib'`` backend (see
        :func:`joblib.Parallel`). ``-1`` (the default) uses all cores.
        Ignored for the ``'mpi'`` and ``'single'`` backends.

    Returns
    -------
    None.

    """
    ###########
    # logging
    logger = logging.getLogger("pyglimer.request.preprocess")
    logger.info('\n\n\n...Preprocessing initialiased...\n')

    tmp.set_clients(clients)


    # Logging for RF creation
    rflogger = logging.getLogger("pyglimer.request.preprocess.RF")

    #########

    if saveasdf:
        preprocessh5(
            phase, rot, pol, taper_perc, model, taper_type, tz, ta,
            rawloc, rfloc, deconmeth, hc_filt, netrestr,
            statrestr, logger, rflogger, client, event_cat, remove_response,
            n_jobs)
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

    # Static settings passed down through the event and waveform loops
    cfg = PreprocessConfig(
        phase=phase, rot=rot, pol=pol, taper_perc=taper_perc,
        taper_type=taper_type, model=model, logger=logger, rflogger=rflogger,
        eh=eh, tz=tz, ta=ta, statloc=statloc, rawloc=rawloc,
        preproloc=preproloc, rfloc=rfloc, deconmeth=deconmeth,
        hc_filt=hc_filt, netrestr=netrestr, statrestr=statrestr,
        remove_response=remove_response)

    # Returns generator object with evtcats with each 100 events
    evtcats = chunks(event_cat, n_split)
    for evtcat in evtcats:

        if client.lower() == 'joblib':
            logger.debug('USING JOBLIB AS BACKEND')
            out = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                delayed(__event_loop)(
                    cfg, event, single_core=False)
                for event in tqdm(evtcat))
        elif client.lower() == 'mpi':
            logger.debug('USING MPI AS BACKEND')
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
            # rflogger.debug(f'Rank/Size: {rank}/{psize}')
            cfg.logger = logger
            cfg.rflogger = rflogger
            out = []
            for ii in tqdm(ind):
                out.append(__event_loop(cfg, evtcat[ii], single_core=False))

        # Use single core only
        elif client.lower() == 'single':
            logger.debug('USING SINGLE CORE BACKEND')
            out = []
            for event in tqdm(evtcat):
                out.append(
                    __event_loop(cfg, event, single_core=True))

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
                logger.exception('Malformed dictionary %s: %s', d, e)
                continue
            if key in masterdict:
                for k in d:
                    if isinstance(d[k], list):
                        masterdict[key].setdefault(k, []).extend(d[k])
            else:
                masterdict[key] = d

        # Write dictionaries in the folder
        for d in masterdict:
            try:
                net, stat = d.split('.')
                write_info(net, stat, masterdict[d], preproloc)
            except (ValueError, KeyError) as e:
                logger.exception('Malformed dictionary %s: %s', d, e)
                continue

    logger.info("Preprocessing finished.")


def __event_loop(
    cfg: PreprocessConfig, event: obspy.core.event.Event,
        single_core: bool):
    """
    Loops over each event in the event catalogue
    """
    # Shorthand for the config settings used in this loop (the rest is
    # passed down via cfg)
    logger = cfg.logger
    preproloc = cfg.preproloc
    rawloc = cfg.rawloc
    netrestr = cfg.netrestr
    statrestr = cfg.statrestr

    logger.info('Processing waveforms for event %s.' % event.resource_id)

    # create list for what will later be the info files
    infolist = []

    # fetch event-data
    origin = (event.preferred_origin() or event.origins[0])
    origin_time = origin.time
    ot_fiss = UTCDateTime(origin_time).format_fissures()
    evtlat = origin.latitude
    evtlon = origin.longitude
    depth = origin.depth or 25000  # For very old events this info is empty

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

    evt = EventContext(
        event=event, origin_time=origin_time, ot_fiss=ot_fiss,
        evtlat=evtlat, evtlon=evtlon, depth=depth,
        prepro_folder=prepro_folder, by_event=by_event)

    for filestr in files:
        try:
            info = __waveform_loop(cfg, evt, filestr, single_core)
            infolist.append(info)
        except Exception as e:
            # Unhandled exceptions should not cause the loop to quit
            # processing one event
            logger.exception('Error processing %s: %s', filestr, e)
            continue

    return infolist


def __waveform_loop(
    cfg: PreprocessConfig, evt: EventContext, filestr: str,
        single_core: bool):
    """
    Loops over each waveform for a specific event and a specific station
    """
    # Shorthand for the config settings (the loop bodies predate the
    # config object)
    phase = cfg.phase
    rot = cfg.rot
    pol = cfg.pol
    taper_perc = cfg.taper_perc
    taper_type = cfg.taper_type
    model = cfg.model
    logger = cfg.logger
    rflogger = cfg.rflogger
    eh = cfg.eh
    tz = cfg.tz
    ta = cfg.ta
    statloc = cfg.statloc
    preproloc = cfg.preproloc
    rfloc = cfg.rfloc
    deconmeth = cfg.deconmeth
    hc_filt = cfg.hc_filt
    remove_response = cfg.remove_response

    event = evt.event
    origin_time = evt.origin_time
    ot_fiss = evt.ot_fiss
    evtlat = evt.evtlat
    evtlon = evt.evtlon
    depth = evt.depth
    prepro_folder = evt.prepro_folder
    by_event = evt.by_event

    infodict = {}  # empty dictionary that will be dumped in a shelve file
    # at the end of the program

    start = time.time()
    # Open files that should be processed
    try:
        st = read(os.path.join(prepro_folder, filestr))
    except FileNotFoundError:  # file has not been downloaded yet
        return  # I will still want to have the RFs
    except Exception as e:  # Unknown erros
        logger.exception('Error reading %s in %s: %s', filestr,
                         prepro_folder, e)
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
    already_done = False
    if __file_in_db(outdir, 'info.dat'):
        with shelve.open(infof, flag='r') as info_db:
            already_done = ot_fiss in info_db['ot_all']
    if not already_done:
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
            infodict['dt'] = st[0].stats.delta
            infodict['sampling_rate'] = st[0].stats.sampling_rate
            infodict['network'] = network
            infodict['station'] = station
            infodict['statlat'] = station_inv[0][0][0].latitude
            infodict['statlon'] = station_inv[0][0][0].longitude
            infodict['statel'] = station_inv[0][0][0].elevation

            if __file_in_db(outdir, 'info.dat'):
                with shelve.open(infof, flag='r') as info:
                    if 'ot_all' not in info or ot_fiss not in info['ot_all']:
                        # Don't count rejected events twice
                        infodict.setdefault('ot_all', []).append(ot_fiss)

            else:
                infodict.setdefault('ot_all', []).append(ot_fiss)

            if single_core:
                write_info(network, station, infodict, preproloc)
                return
            return infodict

        except StreamLengthError as e:
            logger.debug('%s: %s', filestr, e)
        # Everything else that might have gone wrong
        except Exception as e:
            logger.exception('Error processing %s in %s: %s', filestr,
                             prepro_folder, e)

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
    if deconmeth and not __rf_in_db(
        rfloc, network, station, ot_fiss, phase, pol) and crit:

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
                with shelve.open(infof, flag='r') as info:
                    RF = createRF(
                        st, phase, pol=pol, info=info, trim=trim,
                        method=deconmeth)

            else:
                RF = createRF(
                    st, phase, pol=pol, info=infodict, trim=trim,
                    method=deconmeth)

        # Write RF
            rf_filename = h5_FMTSTR.format(
                dir=rfloc, network=network, station=station)
            os.makedirs(rfloc, exist_ok=True)
            with RFDataBase(rf_filename) as rfdb:
                rfdb.add_rf(RF)

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
            rflogger.exception('RF creation failed for %s.%s at %s: %s',
                               network, station, ot_loc, e)

        finally:
            if single_core:
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
        tr.data = np.require(tr.data, dtype=np.float64)
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

    st, crit, f, noisemat, dt, sampling_f = rotate_and_snr_qc(
        phase, st, station_inv, baz, tz)

    set_station_info(infodict, network, station, station_inv, dt, sampling_f)

    if not crit:
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
    append_event_info(
        infodict, event, ot_fiss, evtlat, evtlon, depth, noisemat, f, st,
        baz, distance, rayp_s_deg, first_arrival)

    logger.info("Stream accepted. Preprocessing successful")

    return st, crit, infodict


def __file_in_db(loc: str, filename: str) -> bool:
    """Checks if file "filename" is already in location "loc"."""
    if os.path.isfile(os.path.join(loc, filename)):
        return True
    else:
        return False


def __rf_in_db(
        rfloc: str, network: str, station: str, ot_fiss: str, phase: str,
        pol: str) -> bool:
    """Checks if the receiver function for the given event is already
    saved in the station's hdf5 database file."""
    rf_file = h5_FMTSTR.format(dir=rfloc, network=network, station=station)
    if not os.path.isfile(rf_file):
        return False
    path = hierarchy.format(
        tag='rf', network=network, station=station, phase=phase, pol=pol,
        evt_time=ot_fiss)
    with h5py.File(rf_file, 'r') as f:
        return path in f


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
