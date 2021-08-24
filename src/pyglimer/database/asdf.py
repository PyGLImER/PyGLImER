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
Last Modified: Tuesday, 24th August 2021 10:38:28 am
'''


import os
from warnings import warn

import obspy
from obspy import read, read_inventory, UTCDateTime
from pyasdf import ASDFDataSet

from pyglimer.utils.signalproc import resample_or_decimate
from pyglimer.utils.roundhalf import roundhalf


def rewrite_to_hdf5(cat: obspy.Catalog, rawfolder: str, statloc: str):
    """
    Converts an existing miniseed waveform database to hierachal data format
    (hdf5).

    :param cat: The event catalogue that was used to download the raw data.
    :type cat: obspy.Catalog
    :param rawfolder: The folder that the raw data is saved in - ending with
        the phase code (i.e., waveforms/raw/P)
    :type rawfolder: str
    :param statloc: Location that the station xmls are saved in.
    :type statloc: str
    """
    for event in cat:
        origin_time = event.origins[0].time
        ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]
        evtlat = event.origins[0].latitude
        evtlon = event.origins[0].longitude
        evtlat_loc = str(roundhalf(evtlat))
        evtlon_loc = str(roundhalf(evtlon))
        evtdir = os.path.join(
            rawfolder, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))
        if not os.path.isdir(evtdir):
            continue
        if not os.listdir(evtdir):
            os.rmdir(evtdir)
            continue
        writeraw(event, evtdir, statloc, False, True)


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

    # 2021/08/03
    # Let's create one file per station
    for fi in os.listdir(rawfolder):
        code = '.'.join(fi.split('.')[:-1])
        fname = code + '.h5'
        statxml = read_inventory(os.path.join(statloc, '%s.xml' % code))
        st = read(os.path.join(rawfolder, fi))
    # Start out by adding the event, which later will be associated to
    # each of the waveforms
        if resample:
            try:
                st.filter('lowpass_cheby_2', freq=4, maxorder=12)
                st = resample_or_decimate(st, 10, filter=False)
            except ValueError as e:
                # Corrupt data
                print(e)
                continue

        with ASDFDataSet(os.path.join(outfolder, fname)) as ds:
            # Retrieve eventid - not the most elgant way, but works
            evtid = event.resource_id
            try:
                if st.count() >= 3:
                    ds.add_quakeml(event)
            except ValueError:
                if verbose:
                    warn(
                        'Event with event-id %s already in DB, skipping...'
                        % str(evtid), UserWarning)
                else:
                    pass
            ds.add_waveforms(st, tag='raw_recording', event_id=evtid)
            ds.add_stationxml(statxml)
