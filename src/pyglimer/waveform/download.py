'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tue May 26 2019 13:31:30
Last Modified: Friday, 8th April 2022 01:28:54 pm
'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import fnmatch
from http.client import IncompleteRead
import logging
import os
import shutil
from tqdm import tqdm

from joblib import Parallel, delayed
from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from obspy.core.event.catalog import Catalog
from obspy.taup import TauPyModel
from pyasdf import ASDFDataSet

from pyglimer.database.asdf import writeraw
from pyglimer import tmp
from pyglimer.utils.roundhalf import roundhalf
from pyglimer.utils import utils as pu
from pyglimer.utils.utils import download_full_inventory
from pyglimer.waveform.preprocessh5 import compute_toa


def download_small_db(
    phase: str, min_epid: float, max_epid: float, model: TauPyModel,
    event_cat: Catalog, tz: float, ta: float, statloc: str,
    rawloc: str, clients: list, network: str, station: str, channel: str,
        saveasdf: bool):
    """
    see corresponding method :meth:`~pyglimer.waveform.request.Request.\
    download_waveforms_small_db`
    """

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
    # create output folder
    os.makedirs(statloc, exist_ok=True)
    out = Parallel(n_jobs=-1, prefer='threads')(
        delayed(pu.__client__loop__)(client, statloc, bulk_stat)
        for client in clients)
    inv = pu.join_inv([inv for inv in out])

    logger.info(
        'Computing theoretical times of arrival and checking available data.')

    # Now we compute the theoretical arrivals using the events and the station
    # information
    # We make a list of dicts akin to
    d = {'event': [], 'startt': [], 'endt': [], 'net': [], 'stat': []}
    for net in inv:
        for stat in net:
            logger.info(f"Checking {net.code}.{stat.code}")
            for evt in event_cat:
                try:
                    toa, _, _, _, delta = compute_toa(
                        evt, stat.latitude, stat.longitude, phase, model)
                except (IndexError, ValueError):
                    # occurs when there is no arrival of the phase at stat
                    logger.debug(
                        'No valid arrival found for station %s,' % stat.code
                        + 'event %s, and phase %s' % (evt.resource_id, phase))
                    continue
                # We only do that if the epicentral distances are correct
                # This is already done in compute_toa
                # if delta < min_epid or delta > max_epid:
                #     logger.debug(
                #         'No valid arrival found for station %s, ' % stat.code
                #         + 'event %s, and phase %s' % (evt.resource_id, phase))
                #     continue
                # Already in DB?
                if saveasdf:
                    if wav_in_asdf(net, stat, '*', channel, toa-tz, toa+ta):
                        logger.info(
                            'File already in database. %s ' % stat.code
                            + 'Event: %s' % evt.resource_id)
                        continue
                else:
                    o = (evt.preferred_origin() or evt.origins[0])
                    ot_loc = UTCDateTime(
                        o.time, precision=-1).format_fissures()[:-6]
                    evtlat_loc = str(roundhalf(o.latitude))
                    evtlon_loc = str(roundhalf(o.longitude))
                    folder = os.path.join(
                        rawloc, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))
                    fn = os.path.join(folder, '%s.%s.mseed' % (net, stat))
                    if os.path.isfile(fn):
                        logger.info(
                            'File already in database. %s ' % stat.code
                            + 'Event: %s' % evt.resource_id)
                        continue
                # It's new data, so add to request!
                d['event'].append(evt)
                d['startt'].append(toa-tz)
                d['endt'].append(toa+ta)
                d['net'].append(net.code)
                d['stat'].append(stat.code)

    # Create waveform download bulk list
    bulk_wav = pu.create_bulk_str(
        d['net'], d['stat'], '*', channel, d['startt'], d['endt'])

    if len(bulk_wav) == 0:
        logger.info('No new data found.')
        return

    # This does almost certainly need to be split up, so we don't overload the
    # RAM with the downloaded mseeds
    logger.info('Initialising waveform download.')
    logger.debug(f'The request string looks like this:\n\n{bulk_wav}\n')

    Parallel(n_jobs=1, prefer='threads')(
        delayed(pu.__client__loop_wav__)(
            client, rawloc, bulk_wav, d, saveasdf, inv)
        for client in clients)


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

        # 2021.02.15 Here, we write everything to asdf
        if saveasdf:
            writeraw(event, tmp.folder, statloc, verbose, True)
            # If that works, we will be deleting the cached mseeds here
            try:
                shutil.rmtree(tmp.folder)
            except FileNotFoundError:
                # This does not make much sense, but for some reason it occurs
                # even if the folder exists? However, we will not want the
                # whole process to stop because of this
                pass
        if fast_redownload:
            event_cat[ii:].write(evtfile, format="QUAKEML")

    if not saveasdf:
        download_full_inventory(statloc, clients)
    tmp.folder = "finished"  # removes the restriction for preprocess.py


def get_mseed_storage(
    network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime) -> str:
    """Stores the files and checks if files are already downloaded"""
    # Returning True means that neither the data nor the StationXML file
    # will be downloaded.

    if asdfsave:
        if wav_in_asdf(
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
