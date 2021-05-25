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

from obspy import read_inventory, Inventory
from obspy.core.inventory.station import Station
from obspy.core.inventory.network import Network


def dt_string(dt):
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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Useful for multi-threading"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def add_statxml_to_existing(stat: Station, netcode: str, dir: str):
    net = Network(netcode, stations=[stat])
    inv = Inventory(networks=[net])
    station = inv[0][0].code
    outfile = os.path.join(dir, '%s.%s.xml' % (netcode, station))
    try:
        old_inv = read_inventory(outfile)
    except FileNotFoundError:
        old_inv = Inventory()
    old_inv.extend(inv)
    # Write again to the same file
    old_inv.write(outfile, 'STATIONXML')
