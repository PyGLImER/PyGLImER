#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tue May 26 2019 13:31:30
Last Modified: Thursday, 8th September 2022 04:11:27 pm
'''

import typing as tp
import fnmatch
from http.client import IncompleteRead
import logging
import os
import sys
from functools import partial
from itertools import compress
from tqdm import tqdm
from pprint import pprint

from joblib import Parallel, delayed
import psutil
from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from obspy.core.event.catalog import Catalog
from obspy.core.inventory.inventory import Inventory
from obspy.core.inventory.station import Station
from obspy.core.inventory.network import Network
from obspy.core.inventory.channel import Channel
from obspy.taup import TauPyModel
from pyasdf import ASDFDataSet

from pyglimer.database.raw import RawDatabase, mseed_to_hdf5
from pyglimer import tmp
from pyglimer.utils.roundhalf import roundhalf
from pyglimer.utils import utils as pu
from pyglimer.utils.utils import download_full_inventory
from pyglimer.waveform.preprocessh5 import compute_toa


def __check_times_small_db_sub(
        event_cat: Catalog,
        rawloc: str, tz: float, ta: float, phase: str, model: TauPyModel,
        logger: logging.Logger, saveh5: bool, mintime: float, maxtime: float,
        inv: Inventory, net: str, stat: str, channels: tp.List[Channel]) -> dict:

    d = {'event': [], 'startt': [], 'endt': [], 'net': [], 'stat': [], 'chan': []}

    # Information on available data for hdf5
    global av_data
    av_data = {}

    print(f'--> Enter TOA event loop for station {net}.{stat}')
    for evt in event_cat:


        try:
            logger.debug(f"Computing TOA for {evt.resource_id} and {net}.{stat}")

            # Get origin
            o = (evt.preferred_origin() or evt.origins[0])

            # Get origin dependent station
            STA = inv.select(
                network=net, station=stat,
                starttime=o.time+mintime, endtime=o.time+maxtime)[0][0]

            # Compute the Time of arrival
            toa, _, _, _, delta = compute_toa(
                evt, STA.latitude, STA.longitude, phase, model)

        except (IndexError, ValueError):

            # occurs when there is no arrival of the phase at stat
            logger.debug(
                'No valid arrival found for station %s,' % stat
                + 'event %s, and phase %s' % (evt.resource_id, phase))
            continue

        # Already in DB?
        if saveh5:
            # Check if all expected channels are in saved already
            checklist = []
            for _channel in channels:
                checklist.append(wav_in_hdf5(rawloc, net, stat, '*', _channel, toa-tz, toa+ta))

            if all(checklist):

                continue
        else:
            o = (evt.preferred_origin() or evt.origins[0])
            ot_loc = UTCDateTime(
                o.time, precision=-1).format_fissures()[:-6]
            evtlat_loc = str(roundhalf(o.latitude))
            evtlon_loc = str(roundhalf(o.longitude))
            tmp.folder = os.path.join(
                rawloc, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))

            # create folder for each event
            os.makedirs(tmp.folder, exist_ok=True)

            # Check if all expected channels are in saved already
            checklist = []
            for _channel in channels:
                checklist.append(wav_in_db(net, stat, '*', _channel, toa-tz, toa+ta))

            if all(checklist):
                logger.info(f'Already in database: {net}.{stat} event={str(evt.resource_id).split("=")[-1]}')
                continue

        # It's new data, so add to request!
        d['event'].append(evt)
        d['startt'].append(toa-tz)
        d['endt'].append(toa+ta)
        d['net'].append(net)
        d['stat'].append(stat)

        # Add a list of channels
        addchannels = []
        for _channel, _check in zip(channels, checklist):
            if not _check:
                addchannels.append(_channel)

        d['chan'].append(addchannels)

    # Check the dictionary for overlaps
    # Filtering overlaps
    dd = filter_overlapping_times(d)
    logger.info(f'Found {len(d["net"])-len(dd["net"])} overlaps '
                f'at {net}.{stat}')

    return dd


def filter_overlapping_times(d):
    """Filters the found dictionary for overlapping windows.
    Dictionary looks as follows:
    ``d = {'event': [], 'startt': [], 'endt': [], 'net': [], 'stat': []}``.
    """

    # Check overlapping windows and set checkovelap to False if there is
    # overlap
    checkoverlap = pu.check_UTC_overlap(d['startt'], d['endt'])

    # Create filtered dictionary
    dfilt = dict()
    dfilt['event'] = list(compress(d['event'], checkoverlap))
    dfilt['net'] = list(compress(d['net'], checkoverlap))
    dfilt['stat'] = list(compress(d['stat'], checkoverlap))
    dfilt['chan'] = list(compress(d['chan'], checkoverlap))
    dfilt['startt'] = list(compress(d['startt'], checkoverlap))
    dfilt['endt'] = list(compress(d['endt'], checkoverlap))

    return dfilt


def download_small_db(
    phase: str, min_epid: float, max_epid: float, model: TauPyModel,
    event_cat: Catalog, tz: float, ta: float, statloc: str,
    rawloc: str, clients: list, network: str | tp.List[str],
    station: str | tp.List[str], channel: str,
        saveh5: bool):
    """
    see corresponding method :meth:`~pyglimer.waveform.request.Request.\
    download_waveforms_small_db`
    """

    # Calculate the min and max theoretical arrival time after event time
    # according to minimum and maximum epicentral distance
    mintime = model.get_travel_times(source_depth_in_km=500,
                                      distance_in_degree=min_epid,
                                      phase_list=[phase])[0].time - tz

    maxtime = model.get_travel_times(source_depth_in_km=0.001,
                                      distance_in_degree=max_epid,
                                      phase_list=[phase])[0].time + ta

    # logging
    logger = logging.getLogger('pyglimer.request')

    # If station and network are None
    station = station or '*'
    network = network or '*'
    # First we download the stations to subsequently compute the times of
    # theoretical arrival
    clients = pu.get_multiple_fdsn_clients(clients)

    logger.info('Requesting data from the following FDSN servers:\n %s' % str(
        clients))

    bulk_stat = pu.create_bulk_str(network, station, '*', channel, '*', '*')

    logger.info('Bulk_stat parameter created.')
    logger.debug('Bulk stat parameters: %s' % str(bulk_stat))
    logger.info('Initialising station response download.')

    # Create Station Output folder
    os.makedirs(statloc, exist_ok=True)

    # Get number of cores available
    NCPU = psutil.cpu_count(logical = False) - 2
    NCPU = 1 if NCPU < 1 else NCPU

    # Run parallel station loop.
    if len(clients) == 1:
        logger.debug('Running single thread station loop')
        inv = pu.__client__loop__(clients[0], statloc, bulk_stat)

    else:
        logger.debug('Running parallel station loop')
        out = Parallel(n_jobs=NCPU, prefer='multiprocessing')(
            delayed(pu.__client__loop__)(client, statloc, bulk_stat)
            for client in clients)
        inv = pu.join_inv([inv for inv in out])

    # Create a dictionary of unique networks/stations
    dinv = dict()
    for net in inv:
        # If not in dictionary add network to dictionary
        if net.code in dinv:
            pass
        else:
            dinv[net.code] = dict()

        # If not in dictionary add station to dictionary
        for stat in net:

            if stat.code in dinv[net.code]:
                pass
            else:
                dinv[net.code][stat.code] = set()

            # If not in dictionary add station to dictionary
            for _channel in stat:
                dinv[net.code][stat.code].add(_channel.code)

    # Create lists for parallel computation
    networks, stations, channels, subinvs = [], [], [], []
    netsta_str = []

    for _net, _stations in dinv.items():
        for _sta, _channels in _stations.items():

            # Get actual network and station from inventory
            subinv = inv.select(network=_net, station=_sta)

            # Add stuff to computations lists
            networks.append(_net)
            stations.append(_sta)
            netsta_str.append(f'{_net}.{_sta}')
            channels.append(list(_channels))

            # The inventory now only includes the relevant station entries
            # It is important to KEEP ALL STATION ENTRIES because the location
            # can change and time the inventory was online.
            # We divide into subinvs so that parallel I/O is faster.
            subinvs.append(subinv)

            # Print debugging message
            logger.debug(f" --- {_net}.{_sta}")
            for _cha in _channels:
                logger.debug(f"  |---- {_cha}")

    # Multiple stations
    MULT = True if len(set(netsta_str)) > 1 else False

    # Get bulk string for a single station
    if len(networks) == 1:

        logger.info('Computing TOA and checking available data single threaded')

        # Empty network dict
        d = dict()
        d['net'], d['stat'], d['chan'], d['startt'], d['endt'] = [],[],[],[],[]

        for _chan in channels[0]:
            _d = __check_times_small_db_sub(
                event_cat, rawloc, tz, ta, phase, model,
                logger, saveh5, mintime, maxtime, subinvs[0], networks[0], stations[0], _chan)

            d['net'].extend(_d['net'])
            d['stat'].extend(_d['stat'])
            d['chan'].extend(_d['chan'])

        # Hi bulk
        netsta_bulk = []
        for _net, _sta, _cha, _st, _et in zip(d['net'], d['stat'], d['chan'], d['startt'], d['endt']):
            netsta_bulk.extend(pu.create_bulk_str(_net, _sta, '*', _cha, _st, _et))

        # Sort bulk request
        netsta_bulk.sort()

        # Create lists
        netsta_bulk = [netsta_bulk, ]
        netsta_d = [d, ]

    # Get bulk string for a one station at a time in parallel
    else:
        logger.info('Computing TOA and checking available data in parallel')

        # Create partial function
        dsub = partial(
            __check_times_small_db_sub,
            event_cat, rawloc, tz, ta, phase, model,
            logger, saveh5, mintime, maxtime)

        #
        logger.debug('Launch Joblib')

        # Run partial function in parallel, resulting in one
        # dictionary per net/sta combo
        netsta_d = Parallel(n_jobs=NCPU, backend='multiprocessing')(
            delayed(dsub)(_subinv, _net, _sta, _channels)
            for _subinv, _net, _sta, _channels in
            zip(subinvs, networks, stations, channels))

        # Create one request per station over all events.
        # This makes writing an asdf file per station much easier
        netsta_bulk = []

        # Fix the lists to check to add request per channel
        for _d in netsta_d:
            mini_bulk = []
            nets, stats, chans, st, et = [],[],[],[],[]
            # Get bulk stuff
            for _net, _sta, _channels, _st, _et in zip(_d['net'], _d['stat'], _d['chan'], _d['startt'], _d['endt']):
                for _cha in _channels:
                    nets.append(_net)
                    stats.append(_sta)
                    chans.append(_cha)
                    st.append(_st)
                    et.append(_et)

            for _nets, _stats, _chans, _st, _et \
                in zip(nets, stats, chans, st, et):

                mini_bulk.extend(pu.create_bulk_str(
                    _nets, _stats, '*', _chans, _st, _et))

            # Sort bulk request
            mini_bulk.sort()

            # Add minibulk to overall request
            netsta_bulk.append(mini_bulk)

    if len(netsta_bulk) == 0:
        logger.info('No new data found.')
        return

    # This does almost certainly need to be split up, so we don't overload the
    # RAM with the downloaded mseeds
    logger.info('Initialising waveform download.')
    logger.debug('The request string looks like this:')
    for _bulkd in netsta_bulk:
        for _bw in _bulkd:
            logger.debug(f"{_bw}")

    # Create waveform directories
    os.makedirs(rawloc, exist_ok=True)

    if len(clients) == 1:

        if MULT is False:
            # Length of request
            N = len(netsta_bulk)

            # Number of digits
            Nd = len(str(N))

            # Download station chunks chunks
            for _i, (_subinv, _net, _sta, _bulk, _netsta_d) in enumerate(
                zip(subinvs, networks, stations, netsta_bulk, netsta_d)):

                # Provide status
                logger.info(f"Downloading ... {_i:{Nd}d}/{N:d}")

                # Download
                pu.__client__loop_wav__(
                    clients[0], rawloc, _bulk, _netsta_d, saveh5,
                    _subinv, network=_net, station=_sta)

                logger.info(f"Downloaded {_i:{Nd}d}/{N:d}")

        else:

            # Download multiple stations in parallel
            Parallel(n_jobs=NCPU, backend='multiprocessing')(
                delayed(pu.__client__loop_wav__)(
                clients[0], rawloc, _bulk, _netsta_d, saveh5,
                _subinv, network=_net, station=_sta)
                for _subinv, _net, _sta, _bulk, _netsta_d \
                    in zip(subinvs, networks, stations, netsta_bulk, netsta_d))

    else:
        # Length of requestf
        N = len(netsta_bulk)

        # Number of digits
        Nd = len(str(N))

        # Download station chunks chunks
        for _i, (_subinv, _net, _sta, _bulk, _netsta_d) in enumerate(
                zip(subinvs, networks, stations, netsta_bulk, netsta_d)):
            # Provide status
            logger.info(f"Downloading ... {_i:{Nd}d}/{N:d}")

            # Download
            Parallel(n_jobs=NCPU, backend='multiprocessing')(
                delayed(pu.__client__loop_wav__)(
                    client, rawloc, _bulk, _netsta_d, saveh5, _subinv,
                    network=_net.code,
                    station=_sta.code) for client in clients)

            logger.info(f"Downloaded {_i:{Nd}d}/{N:d}")


def downloadwav(
    phase: str, min_epid: float, max_epid: float, model: TauPyModel,
    event_cat: Catalog, tz: float, ta: float, statloc: str,
    rawloc: str, clients: list, evtfile: str, network: str = None,
    station: str = None, saveasdf: bool = False,
    log_fh: logging.FileHandler = None, loglvl: int = logging.WARNING,
        verbose: bool = False, fast_redownload: bool = False):
    """
    Downloads the waveforms for all events in the catalogue
     for a circular domain around the epicentre with defined epicentral
     distances from Clients defined in clients. Also Station
     xmls for corresponding stations are downloaded.

    Parameters
    ----------
    phase : string
        Arrival phase to be used. P, S, SKS, or ScS.
    min_epid : float
        Minimal epicentral distance to be downloaded.
    max_epid : float
        Maxmimal epicentral distance to be downloaded.
    model : obspy.taup.TauPyModel
        1D velocity model to calculate arrival.
    event_cat : Obspy event catalog
        Catalog containing all events, for which waveforms should be
        downloaded.
    tz : int
        time window before first arrival to download (seconds)
    ta : int
        time window after first arrival to download (seconds)
    statloc : string
        Directory containing the station xmls.
    rawloc : string
        Directory containing the raw seismograms.
    clients : list
        List of FDSN servers. See obspy.Client documentation for acronyms.
    network : string or list, optional
        Network restrictions. Only download from these networks, wildcards
        allowed. The default is None.
    station : string or list, optional
        Only allowed if network != None. Station restrictions.
        Only download from these stations, wildcards are allowed.
        The default is None.
    saveasdf : bool, optional
        Save the dataset as Adaptable Seismic Data Format (asdf; recommended).
        Else, one will be left with .mseeds.
    log_fh : logging.FileHandler, optional
        file handler to be used for the massdownloader logger.
    loglvl : int, optional
        Use this logging level.
    verbose: Bool, optional
        Set True, when experiencing issues with download. Output of
        obspy MassDownloader will be logged in download.log.

    Returns
    -------
    None

    """

    # needed to check whether data is already in the asdf
    global asdfsave
    asdfsave = saveasdf

    global av_data
    # Dictionary holding available data - needed in download functions
    # Keys are {network}{station}
    av_data = {}

    # Calculate the min and max theoretical arrival time after event time
    # according to minimum and maximum epicentral distance
    min_time = model.get_travel_times(source_depth_in_km=500,
                                      distance_in_degree=min_epid,
                                      phase_list=[phase])[0].time

    max_time = model.get_travel_times(source_depth_in_km=0.001,
                                      distance_in_degree=max_epid,
                                      phase_list=[phase])[0].time

    mdl = MassDownloader(providers=clients)

    ###########
    # logging for the download
    fdsn_mass_logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
    fdsn_mass_logger.setLevel(loglvl)

    # # Create handler to the log
    if log_fh is None:
        fh = logging.FileHandler(os.path.join('logs', 'download.log'))
        fh.setLevel(logging.INFO)
        fh.setLevel(loglvl)
        # Create Formatter
        fmt = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
    else:
        fh = log_fh

    fdsn_mass_logger.addHandler(fh)

    ####
    # counter how many events are written only for h5
    jj = 0
    # Loop over each event
    global event
    for ii, event in enumerate(tqdm(event_cat)):
        # fetch event-data
        origin_time = event.origins[0].time
        ot_fiss = UTCDateTime(origin_time).format_fissures()
        fdsn_mass_logger.info('Downloading event: '+ot_fiss)
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude

        # Download location
        ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]
        evtlat_loc = str(roundhalf(evtlat))
        evtlon_loc = str(roundhalf(evtlon))
        tmp.folder = os.path.join(
            rawloc, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))

        # create folder for each event
        os.makedirs(tmp.folder, exist_ok=True)

        # Circular domain around the epicenter. This module also offers
        # rectangular and global domains. More complex domains can be
        # defined by inheriting from the Domain class.

        domain = CircularDomain(latitude=evtlat, longitude=evtlon,
                                minradius=min_epid, maxradius=max_epid)

        restrictions = Restrictions(
            # Get data from sufficient time before earliest arrival
            # and after the latest arrival
            # Note: All the traces will still have the same length
            starttime=origin_time + min_time - tz,
            endtime=origin_time + max_time + ta,
            network=network, station=station,
            # You might not want to deal with gaps in the data.
            # If this setting is
            # True, any trace with a gap/overlap will be discarded.
            # This will delete streams with several traces!
            reject_channels_with_gaps=False,
            # And you might only want waveforms that have data for at least 95%
            # of the requested time span. Any trace that is shorter than 95% of
            # the desired total duration will be discarded.
            minimum_length=0.95,  # For 1.00 it will always delete the waveform
            # No two stations should be closer than 1 km to each other. This is
            # useful to for example filter out stations that are part of
            # different networks but at the same physical station. Settings
            # this option to zero or None will disable that filtering.
            # Guard against the same station having different names.
            minimum_interstation_distance_in_m=100.0,
            # Only HH or BH channels. If a station has BH channels, those will
            # be downloaded, otherwise the HH. Nothing will be downloaded if it
            # has neither.
            channel_priorities=["BH[ZNE12]", "HH[ZNE12]"],
            # Location codes are arbitrary and there is no rule as to which
            # location is best. Same logic as for the previous setting.
            # location_priorities=["", "00", "10"],
            sanitize=False
            # discards all mseeds for which no station information is available
            # I changed it too False because else it will redownload over and
            # over and slow down the script
        )

        # The data will be downloaded to the ``./waveforms/`` and
        # ``./stations/`` folders with automatically chosen file names.
        incomplete = True
        while incomplete:
            try:
                mdl.download(
                    domain, restrictions,
                    mseed_storage=get_mseed_storage,
                    stationxml_storage=statloc,
                    threads_per_client=3, download_chunk_size_in_mb=50)
                incomplete = False
            except IncompleteRead:
                continue  # Just retry for poor connection
            except Exception:
                incomplete = False  # Any other error: continue

        # 2021.02.15 Here, we write everything to hdf5
        # this should not be done every single event, but perhaps every 100th
        # event or so.
        if saveasdf and ii-jj > 99:
            fdsn_mass_logger.info('Rewriting mseed to hdf5.....')
            mseed_to_hdf5(rawloc, False)
            jj = ii
            fdsn_mass_logger.info('...Done')
        if fast_redownload:
            event_cat[ii:].write(evtfile, format="QUAKEML")

    # Always issues with the massdownlaoder inventory, so lets fix it here
    fdsn_mass_logger.info('Downloading all response files.....')
    download_full_inventory(statloc, clients)
    if saveasdf:
        fdsn_mass_logger.info('Rewriting mseed and xmls to hdf5.....')
        mseed_to_hdf5(rawloc, save_statxml=True, statloc=statloc)
        fdsn_mass_logger.info('...Done')
    tmp.folder = "finished"  # removes the restriction for preprocess.py


def get_mseed_storage(
    network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime) -> str:
    """Stores the files and checks if files are already downloaded"""
    # Returning True means that neither the data nor the StationXML file
    # will be downloaded.

    if asdfsave:
        if wav_in_hdf5(
                os.path.join(tmp.folder, os.pardir),
                network, station, location, channel, starttime, endtime):
            return True
    else:
        if wav_in_db(network, station, location, channel, starttime, endtime):
            return True

    # If a string is returned the file will be saved in that location.
    return os.path.join(tmp.folder, "%s.%s.mseed" % (network, station))


def get_stationxml_storage(network: str, station: str, statloc: str):

    filename = os.path.join(statloc, "%s.%s.xml" % (network, station))

    return {

        "available_channels": [],

        "missing_channels": '*',

        "filename": filename}


def wav_in_db(
    network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime) -> bool:

    """Checks if waveform is already downloaded."""
    path = os.path.join(tmp.folder, "%s.%s.mseed" % (network, station))

    if os.path.isfile(path):
        st = read(path)
        # '.' + location +
    else:
        return False
    if len(st) == 3:
        return True  # All three channels are downloaded
    # In case, channels are missing
    elif len(st) == 2:
        if st[0].stats.channel == channel or st[1].stats.channel == channel:
            return True
    elif st[0].stats.channel == channel:
        return True
    else:
        return False


def wav_in_asdf(
    network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime) -> bool:
    """Is the waveform already in the asdf database?"""
    asdf_file = os.path.join(tmp.folder, os.pardir, '%s.%s.h5' % (
        network, station))

    if not os.path.isfile(asdf_file):
        logging.debug(f'{asdf_file} not found')
        return False

    # Change precision of start and endtime
    # Pyasdf rounds with a precision of 1 for the starttime and 0 for endtime
    starttime = UTCDateTime(
        starttime, precision=1).format_iris_web_service()[:-6]
    endtime = endtime.format_iris_web_service()[:-6]
    # make them patterns for the cases where it downloads a slightly different
    # time window
    starttime += '??'
    endtime += '??'

    # Waveforms are saved in pyasdf with filenames akin to:
    nametag = "%s.%s.%s.%s__%s__%s__raw_recording"\
        % (network, station, location, channel, starttime, endtime)

    with ASDFDataSet(asdf_file, mode='r') as ds:
        # Note .list only checks the names not the actual traces and is
        # therefore way faster!
        try:
            exists = len(fnmatch.filter(ds.waveforms[
                '%s.%s' % (network, station)].list(), nametag)) > 0
            # exists = nametag in ds.waveforms[
            #     '%s.%s' % (network, station)].list()
            if exists:
                logging.debug(f'{nametag} already exists!')
            return exists
        except KeyError:
            return False

def wav_in_hdf5(
    rawloc: str, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime) -> bool:
    """Is the waveform already in the Raw hdf5 database?"""

    # H5 file location
    h5_file = os.path.join(rawloc, '%s.%s.h5' % (
        network, station))

    t = starttime.format_fissures()[:-4]

    # First check dictionary
    try:
        if t in av_data[network][station][channel]:
            return True
        else:
            return False
    except KeyError:
        pass

    # Check whether there is data from this station at all
    av_data.setdefault(network, {})
    av_data[network].setdefault(station, {})

    if not os.path.isfile(h5_file):
        logging.debug(f'{h5_file} not found')
        av_data[network][station][channel] = []
        return False

    # The file exists, so we will have to open it and get the dictionary
    with RawDatabase(h5_file) as rdb:
        av_data[network][station] = rdb._get_table_of_contents()

        # Safety net for when the channel list is empty for some reason.
        if not av_data[network][station]:
            av_data[network][station][channel] = []

    # execute this again to check
    return wav_in_hdf5(rawloc, network, station, location, channel, starttime, endtime)