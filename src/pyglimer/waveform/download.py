'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tue May 26 2019 13:31:30
Last Modified: Thursday, 25th March 2021 04:02:08 pm
'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


from http.client import IncompleteRead
import logging
import os
import shutil
from tqdm import tqdm

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from pathlib import Path
from pyasdf import ASDFDataSet

from pyglimer.database.asdf import writeraw
from pyglimer import tmp
from pyglimer.utils.roundhalf import roundhalf


def downloadwav(phase, min_epid, max_epid, model, event_cat, tz, ta, statloc,
                rawloc, clients, network: str = None, station: str = None,
                saveasdf: bool = False, logdir: str = None, debug: bool = False,
                verbose: bool = False):
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
    logdir : string, optional
        Set the directory to where the download log is saved
    debug : Bool, optional
        All loggers go to debug mode.
    verbose: Bool, optional
        Set True, when experiencing issues with download. Output of
        obspy MassDownloader will be logged in download.log.

    Returns
    -------
    None

    """

    if saveasdf:
        raise NotImplementedError("Will be implemented in a future version.")
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
    fdsn_mass_logger.setLevel(logging.WARNING)
    if debug:
        fdsn_mass_logger.setLevel(logging.DEBUG)

    # Create handler to the log
    if logdir is None:
        fh = logging.FileHandler(os.path.join('logs', 'download.log'))
    else:
        fh = logging.FileHandler(os.path.join(logdir, 'download.log'))

    fh.setLevel(logging.INFO)
    if debug:
        fh.setLevel(logging.DEBUG)
    fdsn_mass_logger.addHandler(fh)

    if not verbose and not debug:
        fdsn_mass_logger.propagate = False

    # Create Formatter
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)

    ####
    # Loop over each event
    for event in tqdm(event_cat):
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
                    # get_stationxml_storage(network, station, statloc),
                    threads_per_client=3, download_chunk_size_in_mb=50)
                incomplete = False
            except IncompleteRead:
                continue  # Just retry for poor connection
            except Exception:
                incomplete = False  # Any other error: continue

        # 2021.02.15 Here, we write everything to asdf
        if saveasdf:
            writeraw(event, tmp.folder, statloc, verbose)

            # If that works, we will be deleting the cached mseeds here
            shutil.rmtree(tmp.folder)

    tmp.folder = "finished"  # removes the restriction for preprocess.py


def get_mseed_storage(network, station, location, channel, starttime, endtime):
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


def get_stationxml_storage(network, station, statloc):

    # available_channels = []

    # missing_channels = []

    # path = Path(statloc, "%s.%s.xml" % (network, station))

    filename = os.path.join(statloc, "%s.%s.xml" % (network, station))

    return {

        "available_channels": [],

        "missing_channels": '*',

        "filename": filename}


def wav_in_db(network, station, location, channel, starttime, endtime):
    """Checks if waveform is already downloaded."""
    path = Path(tmp.folder, "%s.%s.mseed" % (network, station))

    if path.is_file():
        st = read(os.path.join(tmp.folder, network + "." + station + '.mseed'))
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


def wav_in_asdf(network, station, location, channel, starttime, endtime):
    """Is the waveform already in the asdf database?"""
    asdf_file = os.path.join(tmp.folder, os.pardir, 'raw.h5')

    # Change precision of start and endtime
    # Pyasdf rounds with a precision of 1 for the starttime and 0 for endtime..
    # ... I think
    starttime = UTCDateTime(
        starttime, precision=1).format_iris_web_service()[:-4]
    endtime = endtime.format_iris_web_service()[:-4]

    # Waveforms are saved in pyasdf with filenames akin to:
    nametag = "%s.%s.%s.%s__%s__%s__raw_recording"\
        % (network, station, location, channel, starttime, endtime)

    with ASDFDataSet(asdf_file) as ds:
        if nametag in ds.waveforms['%s.%s' % (network, station)]:
            return True
        else:
            return False
