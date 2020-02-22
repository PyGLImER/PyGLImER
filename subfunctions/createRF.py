#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:10 2020

@author: pm
"""

import numpy as np
from subfunctions import config
from subfunctions.deconvolve import it, damped
from obspy import read


def createRF(st, dt):
    RF = st.copy()
    # delete the old traces
    while RF.count() > 1:
        del RF[1]
    RF[0].stats.channel = config.phase + "RF"
    stream = {}
    for tr in st:
        stream[tr.stats.channel[2]] = tr.data
    # define denominator v and enumerator u
    if config.phase == "P" and config.rot == "RTZ":
        if "Z" in stream:
            v = stream["Z"]
        elif "3" in stream:
            v = stream["3"]
        u = stream["R"]
    elif config.phase == "P" and config.rot == "LQT":
        v = stream["L"]
        u = stream["Q"]
    elif config.phase == "P" and config.rot == "PSS":
        v = stream["P"]
        u = stream["V"]
    elif config.phase == "S" and config.rot == "RTZ":
        if "Z" in stream:
            u = stream["Z"]
        elif "3" in stream:
            u = stream["3"]
        v = stream["R"]
    elif config.phase == "S" and config.rot == "LQT":
        u = stream["L"]
        v = stream["Q"]
    elif config.phase == "S" and config.rot == "PSS":
        u = stream["P"]
        v = stream["V"]
    if config.decon_meth == "it":
        RF[0].data = it(v, u, dt, shift=config.tz)[0]
    elif config.decon_meth == "dampedf":
        RF[0].data = damped(v, u)
    return RF