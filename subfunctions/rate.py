#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:28:38 2020

@author: pm

"""
import os
from obspy import read
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import matplotlib as mpl
from pathlib import Path
from subfunctions import config
import subprocess
import shelve
import time

rating = {}  # mutable object


def rate(network, station, onset=config.tz, phase="S", review=False):
    """"Module to rate and review the quality of waveform from the given
    station"""
    inloc = config.outputloc[:-1] + phase + "/by_station/" + network +\
        '/' + station + '/'
    for file in os.listdir(inloc):
        if file[:4] == "info":   # Skip the info files
            continue
        try:
            st = read(inloc + file)
        except IsADirectoryError as e:
            print(e)
            continue
        y = []
        ch = []
        starttime = str(st[0].stats.starttime)
        old = r_file(network, station, starttime)  # old
        if old and not review:  # skip already rated files
            continue
        elif review and not old:  # skip unrated files
            continue
        for tr in st:
            y.append(tr.data)
            ch.append(tr.stats.channel[2])

        # shorten vector
        # create time vector
        t = np.linspace(0-onset, tr.stats.npts*tr.stats.delta-onset, len(y[0]))
        plt.close('all')
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[2].set_xlabel('time in s')
        ax[1].set_ylabel(starttime)
        for ii, x in enumerate(ax):
            x.plot(t, y[ii][:], color="k")
            # x.axvline(color='r')
            x.set_title(ch[ii])
            x.yaxis.set_major_formatter(ScalarFormatter())
            x.yaxis.major.formatter._useMathText = True
            x.xaxis.set_major_formatter(ScalarFormatter())
            x.xaxis.major.formatter._useMathText = True
            x.yaxis.set_minor_locator(AutoMinorLocator(5))
            x.xaxis.set_minor_locator(AutoMinorLocator(5))
            # x.yaxis.set_label_coords(0.63, 1.01)
            x.yaxis.tick_right()
            x.grid(color='c', which='both', linewidth=.25)
        if old:
            ax[2].text(0, 1, old,
                        bbox=dict(facecolor='green', alpha=0.5))
            ax[2].set_xlabel(['time in s, old rating', old])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        print("1: discard, 2: ok, 3: good, 4: very good, 5: keep old rating",
              "quit with q")
        fig.canvas.mpl_connect('key_press_event', ontype)
        while not plt.waitforbuttonpress(30):
            time.sleep(0.25)
        # fig.canvas.mpl_connect('key_press_event', ontype)
        if rating["k"] == "q":
            break
        elif "k" in rating:
            w_file(network, station, starttime)


def ontype(event):
    """Deals with keyboard input"""
    rating["k"] = event.key


def w_file(network, station, starttime):
    try:
        int(rating["k"])
    except:
        return False
    with shelve.open(network + "." + station + "rating") as f:
        if int(rating["k"]) != 5:
            f[starttime] = rating["k"]
            print("You've rated the stream", rating["k"])

def r_file(network, station, starttime):
    with shelve.open(network + "." + station + "rating") as f:
        if starttime in f:
            old = f[starttime]
        else:
            old = False
    return old


def sort_rated(network, station, phase=config.phase):
    """Functions that sorts waveforms in 4 folders,
    depending on their rating"""
    inloc = config.outputloc[:-1] + phase + "/by_station/" + network +\
        '/' + station + '/'
    for n in range(1, 5):
        subprocess.call(["mkdir", "-p", inloc + str(n)])
    dic = shelve.open(network + "." + station + "rating")
    for file in os.listdir(inloc):
        if file[:4] == "info":   # Skip the info files
            continue
        try:
            st = read(inloc + file)
        except IsADirectoryError as e:
            print(e)
            continue
        starttime = str(st[0].stats.starttime)
        if starttime in dic:
            subprocess.call(["cp", inloc + file, inloc + dic[starttime] + '/'
                             + file])
