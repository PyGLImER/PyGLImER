#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tue May 26 2019 13:31:30
Last Modified: Friday, 20th January 2023 03:50:29 pm
'''

from multiprocessing import Event
import typing as tp
from http.client import IncompleteRead
import logging
import os
from functools import partial
from itertools import compress
from tqdm import tqdm

from joblib import Parallel, delayed
import psutil
from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from obspy.core.event.catalog import Catalog
from obspy.core.inventory.inventory import Inventory
from obspy.taup import TauPyModel

from pyglimer.database.raw import RawDatabase, mseed_to_hdf5
from pyglimer import tmp
from pyglimer.utils.roundhalf import roundhalf
from pyglimer.utils import utils as pu
from pyglimer.utils.utils import download_full_inventory
from pyglimer.waveform.preprocessh5 import compute_toa


def ____check_times_small_db_event(
        rawloc: str, tz: float, ta: float, phase: str, model: TauPyModel,
        logger: logging.Logger, saveh5: bool, mintime: float, maxtime: float,
        inv: Inventory, net: str, stat: str, channels: tp.List[str],
        evt: Event, av_data_manual: tp.Union[dict, None] = None) \
        -> tp.Tuple[UTCDateTime, tp.Union[tp.List[str]], bool]:

    """Checks whether event already in database and if not compute toa and
    return missing channels.

    Parameters
    ----------
    rawloc : str
        path to waveform folder in the database
    tz : float
        padding before arrival
    ta : float
        padding after arrival
    phase : str
        phases to check for
    model : TauPyModel
        model, from obspy
    logger : logging.Logger
        logger for log messages
    saveh5 : bool
        whether to check in the hdf5 database or the miniseed database
    mintime : float
        mintime
    maxtime : float
        maxtime
    inv : Inventory
        station inventory
    net : str
        network name
    stat : str
        station name
    channels : tp.List[str]
        channels
    evt : Event
        event to check arrivals from
    av_data_manual : dict | None, optional
        if not None the events are checked in parallel, by default None

    Returns
    -------
    tp.Tuple[UTCDateTime, tp.List[str]] | bool
        toa, list of channels to request

    """

    # Get origin
    o = (evt.preferred_origin() or evt.origins[0])

    # Make the evt id a global variable for the sub functions if the stations
    # are parallelized
    if not av_data_manual:
        global evt_id

    # Get evt_id
    evt_id = pu.utc_save_str(o.time)

    # Channel fix, not all channels are equally present at in a given time
    # range
    channels_fix = []
    for _cha in channels:
        try:
            STA = inv.select(
                network=net, station=stat, channel=_cha,
                starttime=o.time+mintime-tz, endtime=o.time+maxtime+ta)[0][0]
            channels_fix.append(_cha)
        except Exception:
            logger.debug(f'Channel {net}.{stat}..{_cha} not available for '
                         f'event {str(evt.resource_id).split("=")[-1]}')

    # Already in DB?
    if saveh5:
        # Check if all expected channels are in saved already
        checklist = []
        for _channel in channels_fix:

            # For event parallelism to separate functions.
            if av_data_manual is None:
                indb = wav_in_hdf5(rawloc, net, stat, '*', _channel)
            else:
                indb = evt_id in av_data_manual[net][stat][_channel]

            logger.debug(
                f'||----> {net}.{stat}..{_channel} - {evt_id} in file: {indb}')

            # Add to checklist whether a channel is in the database
            checklist.append(indb)

        if all(checklist):
            logger.info(f'Already in database: {net}.{stat} '
                        f'event={str(evt.resource_id).split("=")[-1]}')
            return False

    # If dataformat is mseed
    else:
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

        for _channel in channels_fix:
            # Here we don't need a separate function because each event just
            # checks the existence of a file, which requires
            # no parallel access.
            checklist.append(wav_in_db(net, stat, '*', _channel))

        if all(checklist):
            logger.info(
                f'Already in database: {net}.{stat} '
                f'event={str(evt.resource_id).split("=")[-1]}')
            return False

    # Get required channels
    # Add a list of channels
    addchannels = []
    for _channel, _check in zip(channels_fix, checklist):
        if not _check:
            addchannels.append(_channel)

    # If the file is not in the database compute the TOA
    try:

        logger.debug(f"Computing TOA for {evt.resource_id} and {net}.{stat}")

        # Get origin dependent station
        STA = inv.select(
            network=net, station=stat,
            starttime=o.time+mintime-tz, endtime=o.time+maxtime+ta)[0][0]

        # Compute the Time of arrival
        toa, _, _, _, _ = compute_toa(
            evt, STA.latitude, STA.longitude, phase, model)

    except (IndexError, ValueError):

        # occurs when there is no arrival of the phase at stat
        logger.debug(
            'No valid arrival found for station %s,' % stat
            + 'event %s, and phase %s' % (evt.resource_id, phase))
        return False

    # Finally return the time of arrival and the channels to be downloaded
    return toa, addchannels


def __check_times_small_db_sub(
        event_cat: Catalog,
        rawloc: str, tz: float, ta: float, phase: str, model: TauPyModel,
        logger: logging.Logger, saveh5: bool, mintime: float, maxtime: float,
        inv: Inventory, net: str, stat: str, channels: tp.List[str],
        parallel: bool = False) -> dict:
    """loop over events

    Parameters
    ----------
    event_cat : Catalog
        catalog of events
    rawloc : str
        path to raw waveform folder in database
    tz : float
        padding before
    ta : float
        padding after
    phase : str
        phase to requested
    model : TauPyModel
        obspy 1d traveltime model
    logger : logging.Logger
        logger
    saveh5 : bool
        whether to save/check the saveh5
    mintime : float
        minimum time after an event
    maxtime : float
        maximum time after an event
    inv : Inventory
        station inventory
    net : str
        network string
    stat : str
        station string
    channels : tp.List[str]
        list of channels for net.sta
    parallel : bool, optional
        if true parallelize event loop, if false iterate over events,
        by default False

    Returns
    -------
    dict
        _description_
    """

    d = {
        'event': [],
        'startt': [], 'endt': [],
        'net': [], 'stat': [], 'chan': []}

    # Get origin times
    # otimes = [_o.time _o in [evt.preferred_origin() or evt.origins[0]
    # for evt in event_cat.events]]

    print(f'--> Enter TOA event loop for station {net}.{stat}')

    # This parallelizes the event loop, which is only use if we downloading
    # data for a single station
    if parallel:
        # Get HDF5 content

        # H5 file location
        h5_file = os.path.join(rawloc, '%s.%s.h5' % (
            net, stat))

        # Check available data from this station
        av_data_manual = {}
        av_data_manual.setdefault(net, {})
        av_data_manual[net].setdefault(stat, {})

        if not os.path.isfile(h5_file):
            logging.debug(f'{h5_file} not found')
            for _cha in channels:
                av_data_manual[net][stat][_cha] = list()
        else:
            # The file exists, so we will have to open it and get
            # the dictionary
            with RawDatabase(h5_file) as rdb:
                av_data_manual[net][stat] = rdb._get_table_of_contents()

                # Safety net for when the channel list is empty for some
                # reason.
                if not av_data_manual[net][stat]:
                    for _cha in channels:
                        av_data_manual[net][stat][_cha] = list()

        # Get number of cores available
        NCPU = psutil.cpu_count(logical=False) - 2
        NCPU = 1 if NCPU < 1 else NCPU

        # Create partial function
        dsub = partial(
            ____check_times_small_db_event,
            rawloc, tz, ta, phase, model, logger, saveh5, mintime, maxtime,
            inv, net, stat, channels)

        # Run parallel event loop
        out = Parallel(n_jobs=NCPU, backend='multiprocessing')(
            delayed(dsub)(_evt, av_data_manual)
            for _evt in event_cat)

        for evt, _out in zip(event_cat, out):

            if not _out:
                continue
            else:
                toa, addchannels = _out

            # It's new data, so add to request!
            d['event'].append(evt)
            d['startt'].append(toa-tz)
            d['endt'].append(toa+ta)
            d['net'].append(net)
            d['stat'].append(stat)
            d['chan'].append(addchannels)

    # Here the event loop is serial, because the stations are parallelized
    else:

        # Information on available data for hdf5
        global av_data
        av_data = {}

        # Loop over events (probably because station loop is parallelized.)
        for evt in event_cat:

            out = ____check_times_small_db_event(
                rawloc, tz, ta, phase, model, logger, saveh5, mintime, maxtime,
                inv, net, stat, channels, evt)

            if not out:
                continue
            else:
                toa, addchannels = out

            # It's new data, so add to request!
            d['event'].append(evt)
            d['startt'].append(toa-tz)
            d['endt'].append(toa+ta)
            d['net'].append(net)
            d['stat'].append(stat)
            d['chan'].append(addchannels)

    def printd(d: dict, logger):
        keys = list(d.keys())
        lists = (v for v in d.values())

        net_idx = keys.index('net')
        sta_idx = keys.index('stat')
        cha_idx = keys.index('chan')
        st_idx = keys.index('startt')
        et_idx = keys.index('endt')

        logger.debug("====================================================")
        for _d in zip(*lists):
            logger.debug(f"||  {_d[net_idx]}.{_d[sta_idx]}..{_d[cha_idx]}: "
                         f"{_d[st_idx]} -- {_d[et_idx]}")
        logger.debug("====================================================")

    # Print TOAs before check overlap
    logger.debug(f'Final list of found arrival times for station {net}.{stat}')
    if logger.level < 20:
        printd(d, logger)

    # Filtering overlaps
    dd = filter_overlapping_times(d)

    # Print TOAs after checking overlaps
    logger.debug('After overlap')
    if logger.level < 20:
        printd(dd, logger)

    # Logging the events found by overlap
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


def inv2uniqlists(inv: Inventory):
    """Creates a list of unique station and channel lists from an inventory."""

    # logging
    logger = logging.getLogger('pyglimer.request')

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

    return networks, stations, channels, subinvs, netsta_str


def create_bulk_list(netsta_d: tp.List[dict]):
    """Takes in a list of dictionaries, to create a bulk request list
    of dictionaries that contain station info and bulk request strings."""

    # Fix the lists to check to add request per channel
    bulk_list = []
    fullbulk = []

    # Loop over station dictionaries
    for _d in netsta_d:

        # Create new set of lists
        nets, stats, chans, st, et, evts = [], [], [], [], [], []

        # Loop over stations
        for _net, _sta, _channels, _st, _et, _evt in zip(
                _d['net'], _d['stat'], _d['chan'],
                _d['startt'], _d['endt'], _d['event']):

            # Loop over channels
            for _cha in _channels:
                nets.append(_net)
                stats.append(_sta)
                chans.append(_cha)
                st.append(_st)
                et.append(_et)
                evts.append(_evt)

        # Create new dictionary with channelwise bulk requests
        bulk_dict = dict()
        bulk_dict['net'] = nets
        bulk_dict['stat'] = stats
        bulk_dict['chan'] = chans
        bulk_dict['startt'] = st
        bulk_dict['endt'] = et
        bulk_dict['event'] = evts

        # one list per net.sta.cha start/endtime
        mini_bulk_list = []
        for _nets, _stats, _chans, _st, _et \
                in zip(nets, stats, chans, st, et):

            # Get list of bulk strings associated with net.sta.... combo
            subbulk = pu.create_bulk_str(
                _nets, _stats, '*', _chans, _st, _et)

            # Add list to list of bulk associated with corresponding
            # net.sta... combo
            mini_bulk_list.append(subbulk)

            # Get full number of bulk requests by extending a list of all
            # requests.
            fullbulk.extend(subbulk)

        # Add list to bulk
        bulk_dict['bulk'] = mini_bulk_list

        # Append the new dictionary to the bulk_list
        bulk_list.append(bulk_dict)

    # The full number of bulk requests
    Nbulk = len(fullbulk)

    return bulk_list, Nbulk


def download_small_db(
    phase: str, min_epid: float, max_epid: float, model: TauPyModel,
    event_cat: Catalog, tz: float, ta: float, statloc: str,
    rawloc: str, clients: list, network: tp.Union[str, tp.List[str]],
    station: tp.Union[str, tp.List[str]], channel: str,
        saveh5: bool):
    """
    see corresponding method :meth:`~pyglimer.waveform.request.Request.\
    download_waveforms_small_db`
    """

    # Sort event catalog after origin time
    idx_list = sorted(
        range(len(event_cat)),
        key=lambda k: (
            event_cat[k].preferred_origin()
            or event_cat[k].origins[0]).time.format_fissures())

    # Remake event catalog into sorted event catalog
    event_cat = Catalog([event_cat[idx] for idx in idx_list])

    # Calculate the min and max theoretical arrival time after event time
    # according to minimum and maximum epicentral distance
    mintime = model.get_travel_times(
        source_depth_in_km=500,
        distance_in_degree=min_epid,
        phase_list=[phase])[0].time - tz

    maxtime = model.get_travel_times(
        source_depth_in_km=0.001,
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

    # Find earliest start and end times
    otimes = [evt.preferred_origin().time for evt in event_cat.events]

    bulk_stat = pu.create_bulk_str(
        network, station, '*', channel, min(otimes), max(otimes))

    logger.info('Bulk_stat parameter created.')
    logger.debug('Bulk stat parameters: %s' % str(bulk_stat))
    logger.info('Initialising station response download.')

    # Create Station Output folder
    os.makedirs(statloc, exist_ok=True)

    # Get number of cores available
    NCPU = psutil.cpu_count(logical=False) - 2
    NCPU = 1 if NCPU < 1 else NCPU

    # Run single thread station
    if len(clients) == 1:
        logger.debug('Running single thread station loop')
        inv = pu.__client__loop__(clients[0], statloc, bulk_stat)

    # or run parallel station loop.
    else:
        logger.debug('Running parallel station loop')
        out = Parallel(n_jobs=NCPU, prefer='multiprocessing')(
            delayed(pu.__client__loop__)(client, statloc, bulk_stat)
            for client in clients)
        inv = pu.join_inv([inv for inv in out])

    # Create list of unique
    networks, stations, channels, subinvs, netsta_str = inv2uniqlists(inv)

    # Multiple stations
    MULT = True if len(set(netsta_str)) > 1 else False

    # Get bulk string for a single station
    if not MULT:

        logger.info(
            'Computing TOA and checking available data single threaded')

        # Empty network dict
        d = dict()
        d['net'], d['stat'], d['chan'], d['startt'], d['endt'] \
            = [], [], [], [], []

        # Since we only have a single station we can loop over the events
        d = __check_times_small_db_sub(
            event_cat, rawloc, tz, ta, phase, model,
            logger, saveh5, mintime, maxtime, subinvs[0], networks[0],
            stations[0], channels[0], parallel=True)

        # Since theres only a single station, there's also only a single
        # dictionary but we make it into a list to conform with the rest of the
        # code
        netsta_d = [d, ]

    # Get bulk string for a one station at a time in parallel
    else:
        logger.info('Computing TOA and checking available data in parallel')

        # Create partial function with the input that is the same for all
        # function calls
        dsub = partial(
            __check_times_small_db_sub,
            event_cat, rawloc, tz, ta, phase, model,
            logger, saveh5, mintime, maxtime)

        # Run partial function in parallel, resulting in one
        # dictionary per net/sta combo
        netsta_d = Parallel(n_jobs=NCPU, backend='multiprocessing')(
            delayed(dsub)(_subinv, _net, _sta, _channels)
            for _subinv, _net, _sta, _channels in
            zip(subinvs, networks, stations, channels))

    # Transform the list of network.station dictionaries containing lists
    # of channels to a list of network.station dictionaries that lists channels
    # individually and contain the bulk request parameters.
    bulk_list, Nbulk = create_bulk_list(netsta_d)

    # End here if there aren't any bulk requests.
    if Nbulk == 0:
        logger.info('No new data found.')
        return

    # Else log the number of total requests to be made
    else:
        logger.info(f'A total of {Nbulk} requests will be made.')

    # Log the final bulk strings
    if logger.level < 20:
        logger.debug('The request string looks like this:')
        for _d in bulk_list:
            for _bulk in _d['bulk']:
                for _bw in _bulk:
                    logger.debug(f"{_bw}")

    # This does almost certainly need to be split up, so we don't overload the
    # RAM with the downloaded mseeds
    logger.info('Initialising waveform download.')

    # Create waveform directories
    os.makedirs(rawloc, exist_ok=True)

    # If the number of clients is 1, parallelize either stations, or events if
    # we download a single station
    if len(clients) == 1:

        # Single Station, parallellize events -> kwarg parallel=True
        if not MULT:

            pu.__client__loop_wav__(
                clients[0], rawloc, bulk_list[0], saveh5,
                subinvs[0], network=networks[0], station=stations[0],
                parallel=True)

        # Multiple Stations, parallellize stations directly
        else:

            # Download multiple stations in parallel
            Parallel(n_jobs=NCPU, backend='multiprocessing')(
                delayed(pu.__client__loop_wav__)(
                    clients[0], rawloc, _bulk_dict, saveh5,
                    _subinv, network=_net, station=_sta)
                for _subinv, _net, _sta, _bulk_dict
                in zip(subinvs, networks, stations, bulk_list))

    # If there are multiple clients, parallelize the clients
    else:
        # Length of request
        N = len(bulk_list)

        # Number of digits
        Nd = len(str(N))

        # Download station chunks chunks
        for _i, (_subinv, _net, _sta, _bulk_dict) in enumerate(
                zip(subinvs, networks, stations, bulk_list)):
            # Provide status
            logger.info(f"Downloading ... {_i:{Nd}d}/{N:d}")

            # Download
            Parallel(n_jobs=NCPU, backend='multiprocessing')(
                delayed(pu.__client__loop_wav__)(
                    client, rawloc, _bulk_dict, saveh5, _subinv,
                    network=_net.code,
                    station=_sta.code) for client in clients)

            logger.info(f"Downloaded {_i:{Nd}d}/{N:d}")


def downloadwav(
    phase: str, min_epid: float, max_epid: float, model: TauPyModel,
    event_cat: Catalog, tz: float, ta: float, statloc: str,
    rawloc: str, clients: list, evtfile: str, network: str = None,
    station: str = None, inventory_restriction: Inventory = None,
    saveasdf: bool = False,
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
    inventory_restriction : Inventory, optional
        If not None the provided inventory will be used to restrict the
        retrieval of station data and waveforms to the provided inventory.
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
        fh.setLevel(logging.info)
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
    global evt_id

    for ii, event in enumerate(tqdm(event_cat)):
        # fetch event-data
        origin_time = (event.preferred_origin() or event.origins[0]).time
        ot_fiss = UTCDateTime(origin_time).format_fissures()
        fdsn_mass_logger.info('Downloading event: '+ot_fiss)
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude

        evt_id = pu.utc_save_str(origin_time)

        # Download location
        tmp.folder = os.path.join(rawloc, f'{evt_id}')

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
            sanitize=False,
            # discards all mseeds for which no station information is available
            # I changed it too False because else it will redownload over and
            # over and slow down the script
            # Should restrict the only to stations with the provided inventory
            # if From my first test it does not look like it.
            limit_stations_to_inventory=inventory_restriction
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
                    threads_per_client=3, download_chunk_size_in_mb=500)
                incomplete = False
            except IncompleteRead:
                continue  # Just retry for poor connection
            except Exception as e:
                fdsn_mass_logger.info(e)
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
                network, station, location, channel):
            return True
    else:
        if wav_in_db(network, station, location, channel):
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
        network: str, station: str, location: str, channel: str) -> bool:

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


def wav_in_hdf5(
        rawloc: str, network: str, station: str, location: str,
        channel: str) -> bool:
    """Is the waveform already in the Raw hdf5 database?"""

    # H5 file location
    h5_file = os.path.join(rawloc, '%s.%s.h5' % (
        network, station))

    # First check dictionary
    try:
        if evt_id in av_data[network][station][channel]:
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
    return wav_in_hdf5(
        rawloc, network, station, location, channel)


def wav_in_hdf5_no_global(
        av_data: dict, network: str, station: str, channel: str,
        evt_id) -> bool:
    """Is the waveform already in the Raw hdf5 database?"""

    if channel not in av_data[network][station]:
        return False

    if evt_id in av_data[network][station][channel]:
        return True
    else:
        return False
