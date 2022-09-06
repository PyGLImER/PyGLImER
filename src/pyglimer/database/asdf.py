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

Last Modified: Friday, 6th May 2022 02:28:13 pm
'''

import logging
import os
import shutil

import obspy
from obspy import read, read_inventory, UTCDateTime
from obspy.core.event.catalog import read_events, Event
from obspy.core.inventory.inventory import Inventory
from obspy.core.stream import Stream
from pyasdf import ASDFDataSet

from pyglimer.utils.signalproc import resample_or_decimate
from pyglimer.utils.roundhalf import roundhalf


def rewrite_to_hdf5(catfile: str, rawfolder: str, statloc: str):
    """
    Converts an existing miniseed waveform database to hierachal data format
    (hdf5).

    :param catfile: The pat hto the event catalogue that was used to download
        the raw data. Will be altered during the process (removes already used
        ones).
    :type catfile: path to obspy.Catalog (str)
    :param rawfolder: The folder that the raw data is saved in - ending with
        the phase code (i.e., waveforms/raw/P)
    :type rawfolder: str
    :param statloc: Location that the station xmls are saved in.
    :type statloc: str
    """
    # Create backup of original catalog
    shutil.copyfile(catfile, '%s_bac' % catfile)
    cat = read_events(catfile)
    while cat.count():
        event = cat[0]
        origin_time = event.origins[0].time
        ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude
        evtlat_loc = str(roundhalf(evtlat))
        evtlon_loc = str(roundhalf(evtlon))
        evtdir = os.path.join(
            rawfolder, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))
        if not os.path.isdir(evtdir):
            pass
        elif not os.listdir(evtdir):
            os.rmdir(evtdir)
        else:
            writeraw(event, evtdir, statloc, False, True)
        logging.warning('removing event...')
        del cat[0]
        # Overwrite old catalog, so we don't have to restart the whole
        # process over again afterwards
        cat.write(catfile, format="QUAKEML")


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


def save_raw_single_station_asdf(
        network: str, station: str, saved: dict,
        st: Stream, rawloc: str, inv: Inventory):
    """
    A variation of the above function that will open the ASDF file once and write
    all traces and then close it afterwards
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

    # Filename of the station to be opened.
    fname = '%s.%s.h5' % (network, station)

    # Just use the same name
    with ASDFDataSet(os.path.join(rawloc, fname)) as ds:
        # Events should not be added because it will read the whole
        # catalogue every single time!
        N = len(saved['event'])
        for _i, (evt, startt, endt, net, stat) in enumerate(zip(
            saved['event'], saved['startt'], saved['endt'], saved['net'],
                saved['stat'])):
            logging.debug(f'{net}.{stat}: Processing #{_i}/N')
            # earlier we downloaded all locations, but we don't really want
            # to have several, so let's just keep one
            try:
                sst = st.select(network=net, station=stat)
                # This might actually be empty if so, let's just skip
                if sst.count() == 0:
                    logging.debug(f'No trace of {net}.{stat} in Stream.')
                    continue

                # This must assume that there is no overlap
                slst = sst.slice(startt, endt)

                # Only write the prevelant location
                locs = [tr.stats.location for tr in sst]
                filtloc = max(set(locs), key=locs.count)
                sslst = slst.select(location=filtloc)

                # Check whether channels overlap
                sinv = inv.select(net, stat, starttime=startt, endtime=endt)

                logging.debug(f'{net}.{stat} writing event #{_i}/N')
                write_st_to_ds(ds, sslst, evt, sinv)

            except Exception as e:
                logging.error(e)


def write_st_to_ds(
    ds: ASDFDataSet, st: Stream, outfolder: str, statxml: Inventory,
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

    # Add waveforms and stationxml
    ds.add_waveforms(st, tag='raw_recording')
    ds.add_stationxml(statxml)
    # If there are still problems, we will have
    # to check whether they are similar probelms to add event