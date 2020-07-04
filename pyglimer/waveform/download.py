'''
Author: Peter Makus (peter.makus@student.uib.no
Created: Tue May 26 2019 13:31:30
Last Modified: Friday, 3rd July 2020 11:14:15 am
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from http.client import IncompleteRead
import logging
import os
import subprocess

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from pathlib import Path

from .. import tmp
from ..utils.roundhalf import roundhalf


def downloadwav(phase, min_epid, max_epid, model, event_cat, tz, ta, statloc,
                rawloc, clients, network=None, station=None, 
                logdir=None, debug=False):
    """
    Downloads the waveforms for all events in the catalogue
     for a circular domain around the epicentre with defined epicentral
     distances from Clients defined in clients. Also Station
     xmls for corresponding stations are downloaded.

    Parameters
    ----------
    phase : string
        P or S for P or SRFS.
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
    debug : Bool, optional
        All loggers go to debug mode.
    
    Returns
    -------
    None

    """

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
        fh = logging.FileHandler('logs/download.log')
    else:
        fh = logging.FileHandler(os.path.join(logdir, 'download.log'))

    fh.setLevel(logging.INFO)
    if debug:
        fh.setLevel(logging.DEBUG)
    fdsn_mass_logger.addHandler(fh)

    fdsn_mass_logger.propagate = False

    # Create Formatter
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)

    ####
    # Loop over each event
    for event in event_cat:
        fdsn_mass_logger.warning('Downloading event: ')
        # fetch event-data
        origin_time = event.origins[0].time
        ot_fiss = UTCDateTime(origin_time).format_fissures()
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude

        # Download location
        ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]
        evtlat_loc = str(roundhalf(evtlat))
        evtlon_loc = str(roundhalf(evtlon))
        tmp.folder = os.path.join(rawloc, ot_loc + '_'
                                     + evtlat_loc + "_" + evtlon_loc)

        # create folder for each event
        if not Path(tmp.folder).is_dir():
            subprocess.call(["mkdir", "-p", tmp.folder])

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
            # And you might only want waveforms that have data for at least 95 % of
            # the requested time span. Any trace that is shorter than 95 % of the
            # desired total duration will be discarded.
            minimum_length=0.95,  # For 1.00 it will always delete the waveform
            # No two stations should be closer than 1 km to each other. This is
            # useful to for example filter out stations that are part of different
            # networks but at the same physical station. Settings this option to
            # zero or None will disable that filtering.
            # Guard against the same station having different names.
            minimum_interstation_distance_in_m=100.0,
            # Only HH or BH channels. If a station has BH channels, those will be
            # downloaded, otherwise the HH. Nothing will be downloaded if it has
            # neither.
            channel_priorities=["BH[ZNE12]", "HH[ZNE12]"],
            # Location codes are arbitrary and there is no rule as to which
            # location is best. Same logic as for the previous setting.
            # location_priorities=["", "00", "10"],
            sanitize=False
            # discards all mseeds for which no station information is available
            # I changed it too False because else it will redownload over and
            # over and slow down the script
        )

        # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
        # folders with automatically chosen file names.
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

    tmp.folder = "finished"  # removes the restriction for preprocess.py


def get_mseed_storage(network, station, location, channel, starttime, endtime):
    """Stores the files and checks if files are already downloaded"""
    # Returning True means that neither the data nor the StationXML file
    # will be downloaded.

    if wav_in_db(network, station, location, channel, starttime, endtime):
        return True

    # If a string is returned the file will be saved in that location.
    return os.path.join(tmp.folder, "%s.%s.mseed" % (network, station))


def wav_in_db(network, station, location, channel, starttime, endtime):
    """Checks if waveform is already downloaded."""
    path = Path(tmp.folder, "%s.%s.mseed" % (network, station))
                                                   #location))
    if path.is_file():
        st = read(tmp.folder + '/' + network + "." + station + '.mseed')
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
