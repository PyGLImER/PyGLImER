'''
A module to handle and rewrite the PyGLiMER database to and in Adaptable
Seismic Format (asdf).

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 12th February 2021 03:24:30 pm

Last Modified: Friday, 9th September 2022 04:28:15 pm
'''

import logging
import os

import obspy
from obspy import read, read_inventory
from obspy.core.event.catalog import Event
from obspy.core.inventory.inventory import Inventory
from obspy.core.stream import Stream
from pyasdf import ASDFDataSet
from pyglimer.database import raw

from pyglimer.utils.signalproc import resample_or_decimate


def writeraw(
    event: obspy.core.event.event.Event, rawfolder: str, statloc: str,
        verbose: bool, resample: bool):
    """
    Write the downloaded miniseed, event, and stationxmls to a single asdf
    file.

    :param event: Event that all the waveforms are associated to.
    :type event: obspy.core.event.event.Event
    :param rawfolder: Folder to save the .h5 file to.
    :type rawfolder: str
    :param statloc: Folder, in which the station xmls can be found
    :type statloc: str
    :param verbose: show warnings?
    :type verbose: bool
    :param resample: Resample/Decimate the waveforms to 10Hz before writing.
        Includes an AA-filter.
    :type resample: bool
    """
    # Folder to save asdf to
    outfolder = os.path.join(rawfolder, os.pardir)

    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")

    # 2021/08/03
    # Let's create one file per station
    files = os.listdir(rawfolder)

    for fi in files:
        code = '.'.join(fi.split('.')[:-1])

        try:
            statxml = read_inventory(os.path.join(statloc, '%s.xml' % code))
            st = read(os.path.join(rawfolder, fi))
        # Start out by adding the event, which later will be associated to
        # each of the waveforms
            write_st(st, event, outfolder, statxml, resample)
        except Exception as e:
            logger.error(e)


def write_st(
    st: Stream, event: Event, outfolder: str, statxml: Inventory,
        resample: bool = True):
    """
    Write raw waveform data to an asdf file. This includes the corresponding
    (teleseismic) event and the station inventory (i.e., response information).

    :param st: The stream holding the raw waveform data.
    :type st: Stream
    :param event: The seismic event associated to the recorded data.
    :type event: Event
    :param outfolder: Output folder to write the asdf file to.
    :type outfolder: str
    :param statxml: The station inventory
    :type statxml: Inventory
    :param resample: Resample the data to 10Hz sampling rate? Defaults to True.
    :type resample: bool, optional
    """
    fname = '%s.%s.h5' % (st[0].stats.network, st[0].stats.station)
    if resample:
        st.filter('lowpass_cheby_2', freq=4, maxorder=12)
        st = resample_or_decimate(st, 10, filter=False)
    with ASDFDataSet(os.path.join(outfolder, fname)) as ds:
        # Events should not be added because it will read the whole
        # catalogue every single time!
        ds.add_waveforms(st, tag='raw_recording')
        ds.add_stationxml(statxml)  # If there are still problems, we will have
        # to check whether they are similar probelms to add event


def save_raw_DB_single_station(
        network: str, station: str, saved: dict,
        st: Stream, rawloc: str, inv: Inventory):
    """
    A variation of the above function that will open the ASDF file once and
    write
    all traces and then close it afterwards
    Save the raw waveform data in the desired format.
    The point of this function is mainly that the waveforms will be saved
    with the correct associations and at the correct locations.

    W are specifically writing event by event streams, so that we don't loose
    parts that are already downloaded!

    :param saved: Dictionary holding information about the original streams
        to identify them afterwards.
    :type saved: dict
    :param st: obspy stream holding all data (from various stations)
    :type st: Stream
    :param rawloc: Parental directory (with phase) to save the files in.
    :type rawloc: str
    :param inv: The inventory holding all the station information
    :type inv: Inventory
    """
    # Get logger
    logger = logging.getLogger('pyglimer.request')

    # Filename of the station to be opened.
    fname = '%s.%s.h5' % (network, station)

    # Just use the same name
    with raw.RawDatabase(os.path.join(rawloc, fname)) as ds:

        # Inventory should be saved once to the the station file
        sinv = inv.select(network=network, station=station)
        ds.add_response(sinv)

        # Number of events
        N = len(saved['event'])
        Ns = len(str(N))
        # Events should not be added because it will read the whole
        # catalogue every single time!
        outst = Stream()
        for _i, (evt, startt, endt, net, stat, chan) in enumerate(zip(
            saved['event'], saved['startt'], saved['endt'], saved['net'],
                saved['stat'], saved['chan'])):
            logger.debug(f'{net}.{stat}..{chan}: Processing #{_i+1:>{Ns}d}/{N}')
            # earlier we downloaded all locations, but we don't really want
            # to have several, so let's just keep one
            try:
                # Grab only single station from stream (should be only one...)
                sst = st.select(network=net, station=stat, channel=chan)

                # This might actually be empty if so, let's just skip
                if sst.count() == 0:
                    logger.debug(f'No trace of {net}.{stat} in Stream.')
                    continue

                # This must assume that there is no overlap
                slst = sst.slice(startt, endt)

                if slst.count() == 0:
                    print(f"No data for {net}.{stat} and event {evt.resource_id}")
                    continue

                # Only write the prevelant location
                locs = [tr.stats.location for tr in slst]
                filtloc = max(set(locs), key=locs.count)
                sslst = slst.select(location=filtloc)
                for tr in sslst:
                    print(tr)
                if sslst.count() == 0:
                    print(f"No data for {net}.{stat} and event {evt.resource_id}")
                    continue

                write_st_to_ds(ds, sslst)

                # Add substream to stream for content update
                outst += sslst

            except Exception as e:
                logger.error(e)

        # Create table of new contents
        new_cont = {}
        for tr in outst:
            new_cont.setdefault(tr.stats.channel, [])
            new_cont[tr.stats.channel].append(
                tr.stats.starttime.format_fissures()[:-4])

        # Get old table of contents
        old_cont = ds._get_table_of_contents()

        # Extend the old table of contents
        for k, v in old_cont.items():
            try:
                new_cont[k].extend(v)
            except KeyError:
                new_cont[k] = v

        # Redefine the table of contents.
        ds._define_content(new_cont)


def write_st_to_ds(
    ds: raw.DBHandler, st: Stream,
        resample: bool = True):
    """
    Write raw waveform data to an asdf file. This includes the corresponding
    (teleseismic) event and the station inventory (i.e., response information).

    :param st: The stream holding the raw waveform data.
    :type st: Stream
    :param event: The seismic event associated to the recorded data.
    :type event: Event
    :param outfolder: Output folder to write the asdf file to.
    :type outfolder: str
    :param statxml: The station inventory
    :type statxml: Inventory
    :param resample: Resample the data to 10Hz sampling rate? Defaults to True.
    :type resample: bool, optional
    """
    if resample:
        st.filter('lowpass_cheby_2', freq=4, maxorder=12)
        st = resample_or_decimate(st, 10, filter=False)

    # Add waveforms and stationxml
    ds.add_waveform(st, tag='raw_recording')

