'''
Module that handles loading local seismic data.
TO use this feed in a function that yields obspy streams.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 8th April 2022 02:27:30 pm
Last Modified: Friday, 6th May 2022 02:28:59 pm
'''

import logging
import os
from typing import Iterable

from obspy import Stream, Inventory
from obspy.core.event.catalog import Catalog
from obspy.taup import TauPyModel

from pyglimer.utils import utils as pu
from pyglimer.waveform.preprocessh5 import compute_toa


def import_database(
    phase: str, model: TauPyModel, event_cat: Catalog, tz: float, ta: float,
    statloc: str, rawloc: str, saveasdf: bool, yield_inv: Iterable[Inventory],
        yield_st: Iterable[Stream]):
    """
    see corresponding method
    :meth:`~pyglimer.waveform.request.Request.import_database`
    """

    # Create Station Output folder
    if not saveasdf:
        os.makedirs(statloc, exist_ok=True)

    # logging
    logger = logging.getLogger('pyglimer.request')

    logger.info('Computing theoretical times of arrival. Save Inventories')

    # Now we compute the theoretical arrivals using the events and the station
    # information
    # We make a list of dicts akin to
    d = {'event': [], 'startt': [], 'endt': [], 'net': [], 'stat': []}
    # terrible triple loop, but well
    for inv in yield_inv():
        for net in inv:
            for stat in net:
                logger.info(f"Checking {net.code}.{stat.code}")

                # Write inventory to db
                logger.debug(f"{net.code}.{stat.code}")
                if not saveasdf:
                    out = os.path.join(
                        statloc, '%s.%s.xml' % (net.code, stat.code))
                    inv.select(network=net.code, station=stat.code).write(
                        out, format="STATIONXML")

                for evt in event_cat:
                    try:
                        toa, _, _, _, _ = compute_toa(
                            evt, stat.latitude, stat.longitude, phase, model)
                    except (IndexError, ValueError):
                        # occurs when there is no arrival of the phase at stat
                        logger.debug(
                            f'No valid arrival found for station {stat.code},'
                            + f'event {evt.resource_id}, and phase {phase}')
                        continue

                    # Append the information for processing
                    d['event'].append(evt)
                    d['startt'].append(toa-tz)
                    d['endt'].append(toa+ta)
                    d['net'].append(net.code)
                    d['stat'].append(stat.code)

    logger.info('Slice data in chunks and save them into PyGLImER database.')
    os.makedirs(rawloc, exist_ok=True)
    for st, inv in (yield_st(), yield_inv()):
        pu.save_raw(d, st, rawloc, inv, saveasdf)
