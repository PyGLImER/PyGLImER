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
from subfunctions.preprocess import QC_S
from subfunctions.createRF import createRF
from subfunctions.plot import plot_stack
from subfunctions.SignalProcessingTB import rotate_PSV, rotate_LQT

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


def automatic_rate(network, station, phase="S"):
    """Check the automatic QC criteria for SRF waveforms."""
    inloc = config.outputloc[:-1] + phase + "/by_station/" + network +\
        '/' + station + '/'
    diff = 0
    ret = 0
    sts = []
    crits = []
    for file in os.listdir(inloc):
        if file[:4] == "info":   # Skip the info files
            continue
        try:
            st = read(inloc + file)
        except IsADirectoryError as e:
            print(e)
            continue
        starttime = str(st[0].stats.starttime)
        st, crit, hf, noisemat = QC_S(st, st[0].stats.delta,
                                      st[0].stats.sampling_rate)
        with shelve.open(network + "." + station + "rating") as f:
            f[starttime + "_auto"] = crit
            if int(f[starttime]) < 3 and crit:
                diff = diff + 1
            if crit:
                ret = ret + 1
        sts.append(st)
        crits.append(crit)
    return diff, ret, sts, crits


def sort_auto_rated(network, station, phase=config.phase):
    """Puts all retained RF and waveforms in a folder"""
    inloc = config.outputloc[:-1] + phase + "/by_station/" + network +\
        '/' + station + '/'
    inloc_RF = config.RF[:-1] + phase + '/' + network + '/' + station + '/'
    subprocess.call(["mkdir", "-p", inloc + "ret"])
    subprocess.call(["mkdir", "-p", inloc_RF + "ret"])
    dic = shelve.open(network + "." + station + "rating")
    for file in os.listdir(inloc):
        if file[:4] == "info":   # Skip the info files
            continue
        try:
            st = read(inloc + file)
        except IsADirectoryError as e:
            print(e)
            continue
        starttime = str(st[0].stats.starttime) + "_auto"
        if starttime in dic and dic[starttime]:
            subprocess.call(["cp", inloc + file, inloc + 'ret/' + file])
            subprocess.call(["cp", inloc + file, inloc_RF + 'ret/' + file])


def auto_rate_stack(network, station, phase=config.phase):
    """Does a quality control on the downloaded files and shows a plot of the
    stacked receiver function. Rotation into PSS. Just meant to facilitate
    the decision for QC parameters (config.SNR_criteria).
    Returns:
        diff: number of files that were retained but not rated 3 or 4 in
        manual QC.
        ret: number of retained files"""
    diff, ret, sts, crits = automatic_rate(network, station, phase)
    sort_auto_rated(network, station, phase)
    info_file = config.outputloc[:-1] + phase + '/by_station/' + network + \
        '/' + station + '/info'
    with shelve.open(info_file) as info:
        statlat = info["statlat"]
        statlon = info["statlon"]
    outloc = config.RF[:-1] + phase + '/' + network + '/' + station + '/test/'
    if Path(outloc).is_dir():
        subprocess.call(['rm', '-rf', outloc])
    subprocess.call(['mkdir', '-p', outloc])
    for jj, st in enumerate(sts):
        if not crits[jj]:
            continue
        dt = st[0].stats.delta
        with shelve.open(info_file) as info:
            ii = info["starttime"].index(st[0].stats.starttime)
            rayp = info["rayp_s_deg"][ii]/111319.9
            ot = info["ot_ret"][ii]
        # _, _, st = rotate_PSV(statlat, statlon, rayp, st)
        st, rat = rotate_LQT(st)
        if rat > 0.5 and rat < 2:  # The PCA did not work properly, L & Q too similar
            ret = ret - 1
            continue
        RF = createRF(st, dt)
        RF.write(outloc + ot + '.mseed', format="MSEED")
    print("N files that were not 3/4: ", diff, "N retained: ", ret)
    subprocess.call(['cp', info_file+'.bak', outloc])
    subprocess.call(['cp', info_file+'.dir', outloc])
    subprocess.call(['cp', info_file+'.dat', outloc])
    sta = station + "/test"
    plot_stack(network, sta, phase=phase)
    return diff, ret
