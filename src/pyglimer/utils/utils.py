'''

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Lucas Sawade (lsawade@princeton.edu)
    Peter Makus (makus@gfz-potsdam.de)


Created: Tue May 26 2019 13:31:30
Last Modified: Wednesday, 7th September 2022 01:27:30 pm
'''

import logging
import os
from obspy.core.event.event import Event
from typing import List, Tuple, Optional
from warnings import warn
import psutil
import numpy as np
from joblib import Parallel, delayed
from obspy.clients.fdsn import Client, header
from obspy.clients.fdsn.header import URL_MAPPINGS
from obspy.core.inventory.inventory import Inventory
from obspy.core.stream import Stream, Trace
from obspy.core.utcdatetime import UTCDateTime

from pyglimer.database.asdf import save_raw_DB_single_station, write_st
from pyglimer.rf.create import RFStream, RFTrace
# from pyglimer.database.asdf import save_raw_single_station_asdf

from .roundhalf import roundhalf


log_lvl = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR}


def utc_save_str(utc: UTCDateTime):
    return UTCDateTime(round(utc.timestamp)).format_fissures()[:-6]


def dt_string(dt: float) -> str:
    """Returns Time elapsed string depending on how much time has passed.
    After a certain amount of seconds it returns minutes, and after a certain
    amount of minutes it returns the elapsed time in hours."""

    if dt > 500:
        dt = dt / 60

        if dt > 120:
            dt = dt / 60
            tstring = "   Time elapsed: %3.1f h" % dt
        else:
            tstring = "   Time elapsed: %3.1f min" % dt
    else:
        tstring = "   Time elapsed: %3.1f s" % dt

    return tstring


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst. Useful for multi-threading"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def download_full_inventory(statloc: str, fdsn_client: list):
    """
    This utility loops through statloc and redownloads the whole response
    (i.e., all channels and all times) for every station. Thus, overwriting
    the old xml file.

    :param statloc: Folder in which the old stationxmls are saved
    :type statloc: str
    :param fdsn_client: List of FDSN providers that should be queried.
    :type fdsn_client: list
    """
    bulk = []
    for fi in os.listdir(statloc):
        f = fi.split('.')
        if f[-1].lower() != 'xml':
            continue
        bulk.append((f[0], f[1], '*', '*', '*', '*'))
    fdsn_client = get_multiple_fdsn_clients(fdsn_client)

    _ = Parallel(n_jobs=-1, prefer='threads')(
        delayed(__client__loop__)(client, statloc, bulk)
        for client in fdsn_client)


def join_inv(invlist=List[Inventory]) -> Inventory:
    inv = invlist.pop(0)
    for ii in invlist:
        for net in ii:
            inv.extend([net])
    return inv


def check_UTC_overlap(
        start: List[UTCDateTime], end: List[UTCDateTime]) -> List[bool]:
    """Checks a list of starttimes and endtimes for overlap

    Parameters
    ----------
    start : List[UTCDateTime]
        List of UTCDatetime starttimes
    end : List[UTCDateTime]
        List of UTCDateTime endtimes

    Returns
    -------
    List[bool]
        List of booleans. If True no overlap, if False overlap
    """

    # Initiate list saying there is no overlap
    check = len(start)*[True]

    for _i, (_start, _end) in enumerate(zip(start, end)):

        # Loop over same array to check whether there is overlap in any of
        # the windows.
        for _j, (_startc, _endc) in enumerate(zip(start, end)):

            # Don't want to compute overlap of the same window
            if _j == _i:
                continue

            # Check whether start or endtime of another window is in the range
            if (_start < _startc and _startc < _end) \
                    or (_start < _endc and _endc < _end):

                check[_i] = False

    return check


def __client__loop__(
        client: str or Client, statloc: str, bulk: list) -> Inventory:
    """
    Download station information from specified client and for the
    specified bulk list.

    :param client: FDSN client to use
    :type client: str or obspy.fdsn.clients.Client
    :param statloc: Location in which the station xmls are to be saved.
    :type statloc: str
    :param bulk: Bulk list to described the queried download.
    :type bulk: list
    :return: The inventory for all downloaded stations
    :rtype: obspy.Inventory
    """
    logger = logging.getLogger('pyglimer.request')

    try:
        if not isinstance(client, Client):
            client = Client(client)
        stat_inv = client.get_stations_bulk(
            bulk, level='response')
    except (header.FDSNNoDataException, header.FDSNException, ValueError) as e:
        logger.warning(str(e))
        logger.debug(f"{bulk}")
        return  # wrong client
        # ValueError is raised for querying a client without station service
    for network in stat_inv:
        netcode = network.code
        for station in network:
            statcode = station.code
            logger.debug(f"{netcode}.{statcode}")
            out = os.path.join(statloc, '%s.%s.xml' % (netcode, statcode))
            stat_inv.select(network=netcode, station=statcode).write(
                out, format="STATIONXML")
    return stat_inv


def chunkdict(d: dict, chunksize) -> list:
    """Chunks upd dictionary with values into list of dictionaries where
    each dictionary has chunksize or less entries."""

    #
    tempd = dict()
    for k, v in d.items():
        tempd[k] = list(chunks(v, chunksize))

    key = list(d.keys())[0]
    Nchunk = len(tempd[key])

    out = []

    for i in range(Nchunk):
        subd = dict()
        for k, v in tempd.items():
            subd[k] = v[i]
        out.append(subd)

    return out


def __download_sub__(
        client: str, rawloc: str, saved: dict, saveasdf: bool,
        inv: Inventory, network: Optional[str] = None,
        station: Optional[str] = None):

    logger = logging.getLogger('pyglimer.request')

    # Get bulk and flatten
    bulk = []

    for _b in saved['bulk']:
        bulk.extend(_b)

    ## The Block here is to debug the actual request made #####
    #  But for code and output purposes it is unecessary
    for _net, _sta, _cha, _st, _et, _event, _b in zip(
            saved['net'], saved['stat'], saved['chan'],
            saved['startt'], saved['endt'], saved['event'], saved['bulk']):

        print(f"{_net}.{_sta}..{_cha}: {_st} -- {_et} >> {str(_event.resource_id).split('=')[-1]}")

        for __b in _b:
            print(f"^-->{__b}")
    ##############################################################

    # Sort bulk request.
    bulk.sort()

    try:
        if not isinstance(client, Client):
            client = Client(client)

        st = client.get_waveforms_bulk(bulk)

        logger.info(f'Downloaded {st.count()} traces from Client {str(client)}.')

        return st

    except (header.FDSNNoDataException, header.FDSNException, ValueError) as e:
        print(str(e))
        if 'HTTP Status code: 204' in str(e):
            logger.debug('--------- NO DATA FOR REQUESTS: ----------------')
            for __bulk in bulk:
                logger.debug(f"||    {__bulk}")
            logger.debug('------------------------------------------------')
            return Stream()
        else:
            print(str(e))
            raise ValueError('See Error above.')
            # ValueError is raised for querying a client without station service


def __client__loop_wav__(
    client: str, rawloc: str, saved: dict, saveasdf: bool,
        inv: Inventory, network: Optional[str] = None,
        station: Optional[str] = None, parallel: bool = False):
    """
    Download waveforms from specified client and for the
    specified bulk list.

    :param client: FDSN client to use
    :type client: str or obspy.fdsn.clients.Client
    :param rawloc: Location in which the waveforms are to be saved
    :type rawloc: str
    :param bulk: Bulk list to described the queried download.
    :type bulk: list
    :param saved: Dictionary holding information about each request and
        their associated events.
    :type saved: dict
    :param saveasdf: True if the desired output format will be asdf.
    :type saveasdf: bool
    :param inv: The inventory for all downloaded stations
    :type inv: obspy.Inventory
    """
    logger = logging.getLogger('pyglimer.request')

    # Making sure that all chunksizes are the same
    chunksize = 100

    # Get one key
    key = list(saved.keys())[0]
    N0 = len(saved[key])

    # Chunkify data and bulkstrings
    if N0 > chunksize:
        saved = chunkdict(saved, chunksize)
    else:
        saved = [saved, ]

        # Request too small to parallelize
        parallel = False

    N = len(saved)
    logger.debug(f"Downloading ...")

    # Get number of cores available
    if parallel:

        NCPU = psutil.cpu_count(logical=False) - 2

        # Turn off parallelism if c=number of available cores is 1
        if NCPU <= 1:
            parallel = False

        elif NCPU > 4:
            NCPU = 4

    if parallel:

        print('saved length ', len(saved))
        print(NCPU)

        # Chunk the chunks
        saved = list(chunks(saved, NCPU))

        counter = 0
        logger.debug(f"    Downloading {N} chunks with each chunk  <= 100 requests dowloaded in Parallel")

        for _i, _saved in enumerate(saved):

            counter += len(_saved)
            logger.debug(f"    Downloading {counter}/{N} chunks")

            streams = Parallel(n_jobs=NCPU, backend='multiprocessing')(
                delayed(__download_sub__)(
                    client, rawloc, __saved, saveasdf, inv, network, station)
                for __saved in _saved)

            for st, __saved in zip(streams, _saved):

                if len(st) == 0:
                    continue

                if saveasdf:
                    # Make sure ASDF file is only opened once
                    if (network is not None) and (station is not None):
                        save_raw_DB_single_station(
                            network, station, __saved, st, rawloc, inv)

                    # Opens and closes ASDF file for each trace. It's more versatile, but
                    # less efficient
                    else:
                        save_raw(__saved, st, rawloc, inv, False)
                else:
                    save_raw(__saved, st, rawloc, inv, False)

    else:

        streams = []
        for i, _saved in enumerate(saved):

            logger.debug(f"    --> Bulk chunk {i}/{N}")

            st = __download_sub__(
                client, rawloc, _saved, saveasdf, inv, network, station)

            if len(st) == 0:
                continue

            if saveasdf:
                # Make sure ASDF file is only opened once
                if (network is not None) and (station is not None):
                    save_raw_DB_single_station(
                        network, station, _saved, st, rawloc, inv)

                # Opens and closes ASDF file for each trace. It's more versatile, but
                # less efficient
                else:
                    save_raw(_saved, st, rawloc, inv, False)
            else:
                save_raw(_saved, st, rawloc, inv, False)

def save_raw(
        saved: dict, st: Stream, rawloc: str, inv: Inventory, saveasdf: bool):
    """
    Save the raw waveform data in the desired format.
    The point of this function is mainly that the waveforms will be saved
    with the correct associations and at the correct locations.

    :param saved: Dictionary holding information about the original streams
        to identify them afterwards.
    :type saved: dict
    :param st: obspy stream holding all data (from various stations)
    :type st: Stream
    :param rawloc: Parental directory (with phase) to save the files in.
    :type rawloc: str
    :param inv: The inventory holding all the station information
    :type inv: Inventory
    :param saveasdf: If True the data will be saved in asdf format.
    :type saveasdf: bool
    """
    # Just use the same name
    for evt, startt, endt, net, stat in zip(
        saved['event'], saved['startt'], saved['endt'], saved['net'],
            saved['stat']):
        # earlier we downloaded all locations, but we don't really want
        # to have several, so let's just keep one
        try:
            sst = st.select(network=net, station=stat)
            # This might actually be empty if so, let's just skip
            if sst.count() == 0:
                logging.debug(f'No trace of {net}.{stat} in Stream.')
                continue
            slst = sst.slice(startt, endt)
            # Only write the prevelant location
            locs = [tr.stats.location for tr in sst]
            filtloc = max(set(locs), key=locs.count)
            sslst = slst.select(location=filtloc)
            if saveasdf:
                sinv = inv.select(net, stat, starttime=startt, endtime=endt)
                write_st(sslst, evt, rawloc, sinv)
            else:
                save_raw_mseed(evt, sslst, rawloc, net, stat)
        except Exception as e:
            logging.error(e)


def save_raw_mseed(evt: Event, slst: Stream, rawloc: str, net: str, stat: str):
    """
    Saves input stream in the correct location in mini seed format.

    :param evt: event that is associated to the recording.
    :type evt: Event
    :param slst: The selected stream. Should contain all components of one
        recording of a teleseismic event.
    :type slst: Stream
    :param rawloc: Location (with phase) to save the miniseed files in (will
        also create subfolders).
    :type rawloc: str
    :param net: Network code
    :type net: str
    :param stat: Station Code
    :type stat: str
    """
    o = (evt.preferred_origin() or evt.origins[0])
    ot_loc = UTCDateTime(o.time, precision=-1).format_fissures()[:-6]
    evtlat_loc = str(roundhalf(o.latitude))
    evtlon_loc = str(roundhalf(o.longitude))
    folder = os.path.join(
        rawloc, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, '%s.%s.mseed' % (net, stat))
    slst.write(fn, fmt='mseed')


def get_multiple_fdsn_clients(
        clients: List[str] or str or None) -> Tuple[Client]:
    """
    Returns a tuple holding all the queried fdsn providers. Also finds
    all available fdsn providers if input is None. Will also sort the clients
    so the big providers (IRIS and ORFEUS) come last.

    Just a modified version of a code in the obspy mass downloader.

    :param clients: List of strings, each describing one client.
        Put None if you want to use all available.
    :type clients: List[str] or str or None
    :return: Tuple of the fdsn Client objects.
    :rtype: Tuple[Client]
    """
    # That bit is stolen from the massdownloader
    if isinstance(clients, str):
        clients = [clients]
    # That bit is stolen from the massdownloader
    elif clients is None:
        providers = dict(URL_MAPPINGS.items())
        _p = []

        if "RASPISHAKE" in providers:
            # exclude RASPISHAKE by default
            del providers["RASPISHAKE"]

        if "IRIS" in providers:
            has_iris = True
            del providers["IRIS"]
        else:
            has_iris = False

        if "ODC" in providers:
            providers["ORFEUS"] = providers["ODC"]
            del providers["ODC"]

        if "ORFEUS" in providers:
            has_orfeus = True
            del providers["ORFEUS"]
        else:
            has_orfeus = False

        _p = sorted(providers)
        if has_orfeus:
            _p.append("ORFEUS")
        if has_iris:
            _p.append("IRIS")

        providers = _p

        clients = tuple(providers)
    return clients


def create_bulk_str(
    networks: str | List[str], stations: str | List[str], location: str,
    channel: str| List[str], t0: UTCDateTime or str or List[UTCDateTime],
        t1: UTCDateTime or str or List[UTCDateTime]) -> List[tuple]:
    """
    Function to generate the input for the obspy functions:
    get_stations_bulk() and get_waveforms_bulk().

    :param networks: The requested networks, can be str or List.
    :type networks: str or List[str]
    :param stations: The requested stations, can be str or List.
    :type stations: str or List[str]
    :param location: Location string
    :type location: str
    :param channel: Channel string
    :type channel: str
    :param t0: starttimes
    :type t0: UTCDateTime or str or List[UTCDateTime]
    :param t1: endtimes
    :type t1: UTCDateTime or str or List[UTCDateTime]
    :raises ValueError: For invalid input types or inputs
    :return: The List to be used as input with the aforementioned functions.
    :rtype: List[tuple]

    .. note::

        All parameters accept wildcards.
    """
    # request object
    bulk = []
    if isinstance(t0, str) and t0 != '*':
        t0 = UTCDateTime(t0)
    elif isinstance(t0, list):
        t0 = [UTCDateTime(t) for t in t0]
    if isinstance(t1, str) and t1 != '*':
        t1 = UTCDateTime(t1)
    elif isinstance(t1, list):
        t1 = [UTCDateTime(t) for t in t1]

    if isinstance(networks, list) and isinstance(stations, list):
        if len(networks) != len(stations):
            raise ValueError(
                'If network and station are provided as lists, they have to\
 have the same length!')
        if (isinstance(t1, list) and isinstance(t0, list)) \
                or type(t1) != type(t0):
            if len(stations) != len(t1) or len(t1) != len(t0) \
                    or type(t1) != type(t0):
                raise ValueError('Time Lists have to have same length!')
            for net, stat, st, et in zip(networks, stations, t0, t1):
                bulk.append((net, stat, location, channel, st, et))
            return bulk
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            for net, stat in zip(networks, stations):
                bulk.append((net, stat, location, channel, t0, t1))
    elif isinstance(networks, list) and stations == '*':
        if (isinstance(t1, list) and isinstance(t0, list)) \
                or type(t1) != type(t0):
            if len(networks) != len(t1) or len(t1) != len(t0) \
                    or type(t1) != type(t0):
                raise ValueError('Time Lists have to have same length!')
            for net, st, et in zip(networks, t0, t1):
                bulk.append((net, stations, location, channel, st, et))
            return bulk
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            for net in networks:
                bulk.append((net, stations, location, channel, t0, t1))
    elif isinstance(stations, list) and isinstance(networks, str):
        if (isinstance(t1, list) and isinstance(t0, list)) \
                or type(t0) != type(t1):
            if len(stations) != len(t1) or len(t1) != len(t0) \
                    or type(t0) != type(t1):
                raise ValueError('Time Lists have to have same length!')
            for stat, st, et in zip(stations, t0, t1):
                bulk.append((networks, stat, location, channel, st, et))
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            for stat in stations:
                bulk.append((networks, stat, location, channel, t0, t1))
    elif isinstance(stations, str) and isinstance(networks, str):
        if (isinstance(t1, list) and isinstance(t0, list)) \
                or type(t1) != type(t0):
            if len(t1) != len(t0) or type(t1) != type(t0):
                raise ValueError('Time Lists have to have same length!')
            for st, et in zip(t0, t1):
                bulk.append((networks, stations, location, channel, st, et))
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            bulk.append((networks, stations, location, channel, t0, t1))
    else:
        raise ValueError('Invalid combination of input types or input length.\
\nCheck the following:\n\t1. If all inputs are lists, do they have the same \
length?\n\t2. If stations is a string and not a wildcard (i.e., *), networks \
has to be a string as well.')
    return bulk


def cos_taper_st(
    st: Stream, taper_len: float, taper_at_masked: bool,
        side: str = 'both') -> Stream:
    """
    Applies a cosine taper to the input Stream.

    :param tr: Input Stream
    :type tr: :class:`~obspy.core.stream.Stream`
    :param taper_len: Length of the taper per side
    :type taper_len: float
    :param taper_at_masked: applies a split to each trace and merges again
        afterwards
    :type taper_at_masked: bool
    :return: Tapered Stream
    :rtype: :class:`~obspy.core.stream.Stream`

    .. note::
        This action is performed in place. If you want to keep the
        original data use :func:`~obspy.core.stream.Stream.copy`.
    """
    if isinstance(st, Trace):
        st = Stream([st])
    elif isinstance(st, RFTrace):
        st = RFStream([st])
    for ii, _ in enumerate(st):
        try:
            st[ii] = cos_taper(st[ii], taper_len, taper_at_masked, side)
        except ValueError as e:
            warn('%s, corresponding trace not tapered.' % e)
    return st


def cos_taper(
    tr: Trace, taper_len: float, taper_at_masked: bool,
        side: str = 'both') -> Trace:
    """
    Applies a cosine taper to the input trace.

    :param tr: Input Trace
    :type tr: Trace
    :param taper_len: Length of the taper per side in seconds
    :type taper_len: float
    :param taper_at_masked: applies a split to each trace and merges again
        afterwards
    :type taper_at_masked: bool
    :return: Tapered Trace
    :rtype: Trace

    .. note::
        This action is performed in place. If you want to keep the
        original data use :func:`~obspy.core.trace.Trace.copy`.
    """
    if taper_len <= 0:
        raise ValueError('Taper length must be larger than 0 s')
    if taper_at_masked:
        st = tr.split()
        st = cos_taper_st(st, taper_len, False)
        st = st.merge()
        if st.count():
            tr.data = st[0].data
            return tr
        else:
            raise ValueError('Taper length must be larger than 0 s')
    taper = np.ones_like(tr.data)
    tl_n = round(taper_len*tr.stats.sampling_rate)
    if tl_n * 2 > tr.stats.npts:
        raise ValueError(
            'Taper Length * 2 has to be smaller or equal to trace\'s length.')
    tap = np.sin(np.linspace(0, np.pi, tl_n*2))
    if side in ['left', 'both']:
        taper[:tl_n] = tap[:tl_n]
    if side in ['right', 'both']:
        taper[-tl_n:] = tap[-tl_n:]
    tr.data = np.multiply(tr.data, taper)
    return tr
