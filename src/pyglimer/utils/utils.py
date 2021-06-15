"""

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Lucas Sawade (lsawade@princeton.edu)


Last Update: November 2019

"""

import os

from joblib import Parallel, delayed
from obspy.clients.fdsn import Client, header


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
        bulk.append((f[0], f[1], '--', '*', '*', '*'))
    if isinstance(fdsn_client, str):
        fdsn_client = [fdsn_client]
    _ = Parallel(n_jobs=-1)(
        delayed(__client__loop__)(client, statloc, bulk)
        for client in fdsn_client)


def __client__loop__(client: str, statloc: str, bulk: list):
    client = Client(client)
    try:
        stat_inv = client.get_stations_bulk(
            bulk, level='response')
    except (header.FDSNNoDataException, header.FDSNException):
        return  # wrong client
    for network in stat_inv:
        netcode = network.code
        for station in network:
            statcode = station.code
            out = os.path.join(statloc, '%s.%s.xml' % (netcode, statcode))
            stat_inv.select(network=netcode, station=statcode).write(
                out, format="STATIONXML")

