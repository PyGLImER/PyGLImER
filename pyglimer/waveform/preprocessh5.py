'''
This is a newer version of preprocess.py meant to be used with pyasdf.
Now, we will have to work in a very different manner than for .mseed files
and process files station wise rather than event wise.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 18th February 2021 02:26:03 pm
Last Modified: Thursday, 25th March 2021 03:18:28 pm
'''

'''
import fnmatch
import logging
import os
import shelve
import subprocess
import time
import itertools
import warnings

import numpy as np
from joblib import Parallel, delayed, cpu_count
from obspy import read, read_inventory, Stream, UTCDateTime
from obspy.clients.iris import Client
from obspy.clients.fdsn import Client as Webclient
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from pathlib import Path
from pyasdf import ASDFDataSet

from pyglimer.waveform.preprocess import StreamLengthError
from .. import tmp
from .errorhandler import redownload, redownload_statxml, \
    NoMatchingResponseHandler, NotLinearlyIndependentHandler
from ..constants import DEG2KM
from .qc import qcp, qcs
from .rotate import rotate_LQT_min, rotate_PSV, rotate_LQT
from ..rf.create import createRF
from ..utils.roundhalf import roundhalf
from ..utils.utils import dt_string, chunks

def preprocessh5(
    phase, rot, pol, taper_perc, event_cat, model, taper_type, tz, ta,
    rawloc, preproloc, rfloc, deconmeth, hc_filt, netrestr,
    statrestr, logger, rflogger, debug):

    # Open ds
    with ASDFDataSet(os.path.join(rawloc, 'raw.h5'), mode='r') as ds:
        

def process(
    st, inv, phase, rot, pol, taper_perc, event_cat, model, taper_type,
    tz, ta, hc_filt, netrestr, statrest, logger, rflogger):
    
    
    # Starting out with the usual preprocessing
    if inv is None:
        msg = 'No station inventory found for station ' + st[0].stats.network \
        + '.' + st[0].stats.station
        logger.exception(msg)
    
    try:
        __station_process__(st, inv)
    except ValueError:
        msg = 'No response information found for station ' + st[0].stats.network \
        + '.' + st[0].stats.station + '.' + st[0].stats.starttime
        logger.exception(msg)
    
    # Now, we do things base on event


def __station_process__(st, inv):
    """
    Processing that is equal for each waveform recorded on one station
    """
    # Change dtype
    for tr in st:
        np.require(tr.data, dtype=np.float64)
        tr.stats.mseed.encoding = 'FLOAT64'
    
    # Anti-Alias
    st.filter(type="lowpass", freq=4.95, zerophase=True, corners=2)
    st.resample(10)  # resample streams with 10Hz sampling rate

    # Remove repsonse
    st.attach_response(inv)
    st.remove_response()

    # DEMEAN AND DETREND #
    st.detrend(type='demean')

    # TAPER #
    st.taper(max_percentage=taper_perc, type=taper_type,
             max_length=None, side='both')
    
    return st


def __cut_resample__(st, logger, first_arrival, network, station,
                   prepro_folder, filestr, taper_perc, taper_type, eh, tz, ta):
    """Cut and resample raw file. Will overwrite original raw"""

    start = time.time()

    # Trim TO RIGHT LENGTH BEFORE AND AFTER FIRST ARRIVAL  #
    # start and endtime of stream
    starttime = first_arrival - tz
    endtime = first_arrival + ta

    if st.count() < 3:
        if not eh:
            raise StreamLengthError(
                ["The stream contains less than three traces.", filestr])
        st = redownload(network, station, starttime, endtime, st)

    # Check one last time. If stream to short raise Exception
    if st.count() < 3:
        raise StreamLengthError(
            ["The stream contains less than 3 traces", filestr])
    
    # trim to according length
    st.trim(starttime=starttime, endtime=endtime)
    # After trimming length has to be checked again (recording may
    # be empty now)
    if st.count() < 3:
        raise Exception("The stream contains less than 3 traces")

    # Write trimmed and resampled files into raw-file folder
    # to save space
    try:
        st.write(os.path.join(prepro_folder, filestr), format="MSEED")
    except ValueError:
        # Occurs for dtype=int32
        for tr in st:
            del tr.stats.mseed
        st.write(os.path.join(prepro_folder, filestr),
                  format="MSEED")


    end = time.time()
    logger.info("Unprocessed file rewritten")
    logger.info(dt_string(end - start))

    return st
'''