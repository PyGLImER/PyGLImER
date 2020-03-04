#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:00:05 2020

@author: pm
"""
from subfunctions import config
from obspy import read
from obspy.core import stream
import shelve
import os
import numpy as np
import matplotlib.pyplot as plt
from subfunctions.createRF import stackRF

ioloc = config.RF +"/IU/HRV/4/"
traces = []
info = shelve.open(ioloc+"info")
for file in os.listdir(ioloc):
    if file[:4] == "info":
        continue
    try:
        st = read(ioloc + file)
    except IsADirectoryError:
        continue
    jj = info["starttime"].index(st[0].stats.starttime)
    st.normalize()
    st[0].stats.coordinates = {}
    st[0].stats["coordinates"]["latitude"] = info["evtlat"][jj]
    st[0].stats["coordinates"]["longitude"] = info["evtlon"][jj]
    st[0].stats.delta = .1
    st[0].data = -np.flip(st[0].data)[round(st[0].stats.npts/2):round(5*st[0].stats.npts/8)]
    del st[0].stats.starttime  # They all have to start at same time
    traces.append(st[0])
statlat = info["statlat"]
statlon = info["statlon"]
info.close()
st = stream.Stream(traces=traces)

# z, stack, RF_mo, raw = stackRF("IU", "HRV/PSS/4")
st2 = st.copy()
# for ii, tr in enumerate(st2):
#     tr.data = RF_mo[ii]
# plt.figure()
# fig = st.plot(type='section', dist_degree=True, ev_coord=(statlat,statlon), scale=2, time_down=True)
st2.plot(type='section', dist_degree=True, ev_coord=(statlat,statlon), scale=2, time_down=True,linewidth=1.5)