'''
A module to handle and rewrite the PyGLiMER database to and in Adaptable
Seismic Format (asdf).

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 12th February 2021 03:24:30 pm
Last Modified: Thursday, 25th March 2021 03:17:08 pm
'''


import os
from warnings import warn

import obspy
from obspy import read, read_inventory
from pyasdf import ASDFDataSet

# def rewrite(folder:str, outputfile:str):


def writeraw(
    event: obspy.core.event.event.Event, rawfolder: str, statloc: str,
        verbose: bool):
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
    """
    # Folder to save asdf to
    outfolder = os.path.join(rawfolder, os.pardir)

    # Start out by adding the event, which later will be associated to
    # each of the waveforms
    with ASDFDataSet(os.path.join(outfolder, 'raw.h5')) as ds:
        # Retrieve eventid - not the most elgant way, but works
        evtid = event.resource_id
        try:
            ds.add_quakeml(event)
        except ValueError:
            if verbose:
                warn(
                    'Event with event-id %s already in DB, skipping...'
                    % str(evtid), UserWarning)
            else:
                pass

    # Read all the waveforms associated to this event
    try:
        st = read(os.path.join(rawfolder, '*.mseed'))
        # Write the waveforms to the asdf
        with ASDFDataSet(os.path.join(outfolder, 'raw.h5')) as ds:
            ds.add_waveforms(st, tag='raw_recording', event_id=evtid)

        # Lastly, we will want to save the stationxmls
        statxml = read_inventory(os.path.join(statloc, '*.xml'))
        with ASDFDataSet(os.path.join(outfolder, 'raw.h5')) as ds:
            ds.add_stationxml(statxml)
    except Exception:
        # For some cases, there will be events without
        # waveforms associated to them
        pass