#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:33:41 2019

Script to start the data accumulation for automatic RF processing.

IMPORTANT:
!!All configuration is done in config.py!!

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""
import multiprocessing
from joblib import Parallel, delayed
from obspy import UTCDateTime

import config
from src import tmp
from src.waveform.request import Request


##############################################################################
##############################################################################

event_coords = (None, None, None, None)

request = Request(
    config.phase, config.rot, config.evtloc, config.statloc, config.waveform, config.outputloc,
    config.RF, config.decon_meth,
    config.starttime, config.endtime, pol='v', wavdownload=config.wavdownload, event_coords=event_coords,
    network=config.network, station=config.station,
    waveform_client=config.waveform_client, re_client=config.re_clients,
    evtcat=config.evtcat, minmag=config.minmag)

if config.wavdownload:
    tmp.folder = "not_started"  # resetting momentary event download folder
    # multi-threading doesn't work anymore this way on python 3.8?
    # if __name__ == '__main__':
    #     multiprocessing.Process(target=request.download_waveforms,
    #                             args=()).start()

    #     multiprocessing.Process(target=request.preprocess, args=()).start()
    request.download_waveforms()
    config.wavdownload = False  # Sop it utilises all cores
    request.preprocess()

else:  # Only preprocess files in waveform location
    tmp.folder = "finished"
    request.preprocess()
