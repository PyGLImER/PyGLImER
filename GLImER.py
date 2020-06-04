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

import config
from src.waveform.request import Request


# !values are located in subfunctions/config.py!

##############################################################################
##############################################################################

event_coords = (config.eMINLAT, config.eMAXLAT, config.eMINLON, config.eMAXLON)

request = Request(
    config.phase, config.starttime, config.endtime, event_coords=event_coords,
    network=config.network, station=config.station,
    waveform_client=config.waveform_client, re_client=config.re_clients,
    evtcat=config.evtcat, minmag=5.5)

if config.wavdownload:
    config.folder = "not_started"  # resetting momentary event download folder
    # multi-threading doesn't work anymore this way on python 3.8?
    # if __name__ == '__main__':
    #     multiprocessing.Process(target=request.download_waveforms,
    #                             args=()).start()

    #     multiprocessing.Process(target=request.preprocess, args=()).start()
    request.download_waveforms()
    config.wavdownload = False  # Sop it utilises all cores
    request.preprocess()

else:  # Only preprocess files in waveform location
    config.folder = "finished"
    request.preprocess()
