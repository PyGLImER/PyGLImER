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
Last Modified: Friday, 27th August 2021 04:07:01 pm

'''

from logging import warn
import logging
import os
from threading import Event
from typing import List

from joblib import Parallel, delayed
from obspy.clients.fdsn import Client, header
from obspy.clients.fdsn.header import URL_MAPPINGS
from obspy.core.inventory.inventory import Inventory
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime

from pyglimer.database.asdf import write_st

from .roundhalf import roundhalf


log_lvl = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR}


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
            inv.extend(net)
    return inv


def __client__loop__(client: str, statloc: str, bulk: list):
    try:
        if not isinstance(client, Client):
            client = Client(client)
        stat_inv = client.get_stations_bulk(
            bulk, level='response')
    except (header.FDSNNoDataException, header.FDSNException, ValueError) as e:
        print(e)
        warn(e)
        return  # wrong client
        # ValueError is raised for querying a client without station service
    for network in stat_inv:
        netcode = network.code
        for station in network:
            statcode = station.code
            out = os.path.join(statloc, '%s.%s.xml' % (netcode, statcode))
            stat_inv.select(network=netcode, station=statcode).write(
                out, format="STATIONXML")
    return stat_inv


def __client__loop_wav__(
    client: str, rawloc: str, bulk: list, saved: dict, saveasdf: bool,
        inv: Inventory):
    try:
        if not isinstance(client, Client):
            client = Client(client)
        st = client.get_waveforms_bulk(bulk)
    except (header.FDSNNoDataException, header.FDSNException, ValueError) as e:
        print(e)
        warn(e)
        return  # wrong client
        # ValueError is raised for querying a client without station service
    save_raw(saved, st, rawloc, inv, saveasdf)


def save_raw(
        saved: dict, st: Stream, rawloc: str, inv: Inventory, saveasdf: bool):
    # Just use the same name
    for evt, startt, endt, net, stat in zip(
        saved['event'], saved['startt'], saved['endt'], saved['net'],
            saved['stat']):
        # earlier we downloaded all locations, but we don't really want
        # to have several, so let's just keep one
        try:
            sst = st.select(network=net, station=stat)
            ssst = Stream()
            ii = 0
            while ssst.count() < 3:
                ssst = sst.select(location=sst[ii].stats.location)
            slst = ssst.slice(startt, endt)
            if saveasdf:
                sinv = inv.select(net, stat, starttime=startt, endtime=endt)
                write_st(slst, evt, rawloc, sinv)
            else:
                save_raw_mseed(evt, slst, rawloc, net, stat)
        except Exception as e:
            logging.error(e)


def save_raw_mseed(evt: Event, slst: Stream, rawloc: str, net: str, stat: str):
    o = (evt.preferred_origin() or evt.origins[0])
    ot_loc = UTCDateTime(o.time, precision=-1).format_fissures()[:-6]
    evtlat_loc = str(roundhalf(o.latitude))
    evtlon_loc = str(roundhalf(o.longitude))
    folder = os.path.join(
        rawloc, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, '%s.%s.mseed' % (net, stat))
    slst.write(fn, fmt='mseed')


def get_multiple_fdsn_clients(clients: List[str] or str or None):
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
    networks: str or List[str], stations: str or List[str], location: str,
    channel: str, t0: UTCDateTime or str or List[UTCDateTime],
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
    if isinstance(networks, list) and isinstance(stations, list):
        if len(networks) != len(stations):
            raise ValueError(
                'If network and station are provided as lists, they have to\
 have the same length!')
        if isinstance(t1, list) and isinstance(t0, list):
            if len(networks) != len(t1) or len(t1) != len(t0):
                raise ValueError('Time Lists have to have same length!')
            for net, stat, st, et in zip(networks, stations, t0, t1):
                bulk.append((net, stat, location, channel, st, et))
            return bulk
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            for net, stat in zip(networks, stations):
                bulk.append((net, stat, location, channel, t0, t1))
    if isinstance(networks, list) and stations == '*':
        if isinstance(t1, list) and isinstance(t0, list):
            if len(networks) != len(t1) or len(t1) != len(t0):
                raise ValueError('Time Lists have to have same length!')
            for net, st, et in zip(networks, t0, t1):
                bulk.append((net, stations, location, channel, st, et))
            return bulk
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            for net in networks:
                bulk.append((net, stations, location, channel, t0, t1))
    elif isinstance(stations, list) and isinstance(networks, str):
        if isinstance(t1, list) and isinstance(t0, list):
            if len(stations) != len(t1) or len(t1) != len(t0):
                raise ValueError('Time Lists have to have same length!')
            for stat, st, et in zip(stations, t0, t1):
                bulk.append((networks, stat, location, channel, st, et))
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            for stat in stations:
                bulk.append((networks, stat, location, channel, t0, t1))
    elif isinstance(stations, str) and isinstance(networks, str):
        if isinstance(t1, list) and isinstance(t0, list):
            if len(t1) != len(t0):
                raise ValueError('Time Lists have to have same length!')
            for st, et in zip(t0, t1):
                bulk.append((networks, stations, location, channel, st, et))
        elif isinstance(t0, (str, UTCDateTime)) and isinstance(
                t1, (str, UTCDateTime)):
            bulk.append((networks, stations, location, channel, t0, t1))
    else:
        raise ValueError('Invalid comobination of input types or input length.\
\nCheck the following:\n\t1. If all inputs are lists, do they have the same \
length?\n\t2. If stations is a string and not a wildcard (i.e., *), networks \
has to be a string as well.')
    return bulk
