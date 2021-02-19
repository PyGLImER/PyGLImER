'''
This is a newer version of preprocess.py meant to be used with pyasdf.
Now, we will have to work in a very different manner than for .mseed files
and process files station wise rather than event wise.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 18th February 2021 02:26:03 pm
Last Modified: Friday, 19th February 2021 09:40:41 am
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
        

def process(st, inv):
    