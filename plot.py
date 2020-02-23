#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:50:58 2020

Module to plot waveforms from receiver functions
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

# plotting properties
# dirFile = os.path.dirname('main.py')

plt.style.use('PaperDoubleFig.mplstyle')
# Make some style choices for plotting
colourWheel = ['#329932', '#ff6961', 'b', '#6a3d9a', '#fb9a99', '#e31a1c',
               '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99',
               '#b15928', '#67001f', '#b2182b', '#d6604d', '#f4a582',
               '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3',
               '#2166ac', '#053061']
dashesStyles = [[3, 1], [1000, 1], [2, 1, 10, 1], [4, 1, 1, 1, 1, 1]]


def plot_all_RF(phase, network, station, TAT=config.tz, dpi=300):
    """Plots all receiver functions in config.outputloc until 50s after
    first arrival.
    INPUT:
        network
        station
        phase: "S" or "P"
        TAT: theoretical arrival time
        dpi: figure Resolution (int)
    """
    # make output folder
    outloc = "figures/RF/" + phase + '/'  # location for figures
    inloc = config.RF[:-1] + phase + '/' + network + '/' + station + '/'
    if not Path(outloc).is_dir():
        subprocess.call(["mkdir", "-p", outloc])
    for file in os.listdir(inloc):
        if file[:4] == "info":   # Skip the info files, they are none for RF
            # but doesn't hurt
            continue
        RF = read(inloc + file)
        dt = RF[0].stats.delta
        N = RF[0].stats.npts
        y = RF[0].data
        if phase == "S":  # flip trace
            y = np.flip(y)
            y = -y  # polarity
            TATp = (N-1)*dt - TAT
        else:
            TATp = TAT
        # shorten vector
        y = y[:round((TATp+50)/dt)]
        # create time vector
        t = np.linspace(0-TATp, 50, len(y))
        plt.close('all')
        fig = plt.figure()
        fig, ax = plt.subplots(1, 1, sharex=True)
        ax.axvline(color='r')  # should plot at x = 0
        ax.plot(t, y, color="k")
        ax.set_ylabel('normalised Amplitude')
        ax.set_xlabel('time in s')
        x = ax
        x.set_xlim(-20, 50)
        x.yaxis.set_major_formatter(ScalarFormatter())
        x.yaxis.major.formatter._useMathText = True
        x.xaxis.set_major_formatter(ScalarFormatter())
        x.xaxis.major.formatter._useMathText = True
        x.yaxis.set_minor_locator(AutoMinorLocator(5))
        x.xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(color='c', which="both", linewidth=.25)
        # x.yaxis.set_label_coords(0.63, 1.01)
        x.yaxis.tick_right()
        # ax.legend(frameon=False, loc='upper left', ncol=2, handlelength=4)
        # plt.legend(['it_max=4000', 'it_max=400'])
        plt.savefig(os.path.join(outloc + file[:-5] + "pdf"), dpi=dpi)


def plot_all_wav(phase, network, station, TAT=config.tz, dpi=300):
    """Plots all waveforms in config.outputloc until 50s after
    first arrival.
    INPUT:
        network
        station
        phase: "S" or "P"
        TAT: theoretical arrival time
        dpi: figure Resolution (int)
    """
    # make output folder
    outloc = "figures/waveforms/" + phase + '/'  # location for figures
    inloc = config.outputloc[:-1] + phase + "/by_station/" + network +\
        '/' + station + '/'
    if not Path(outloc).is_dir():
        subprocess.call(["mkdir", "-p", outloc])
    for file in os.listdir(inloc):
        if file[:4] == "info":  # Skip the info files
            continue
        st = read(inloc + file)
        dt = st[0].stats.delta
        N = st[0].stats.npts
        y = []
        ch = []  # Channel information
        for tr in st:
            y.append(tr.data)
            ch.append(tr.stats.channel[2])
        if phase == "S":  # flip trace
            y = np.flip(y)
            # y = -y  # polarity
            TATp = (N-1)*dt - TAT
        else:
            TATp = TAT
        # shorten vector
        y = y[:][:round((TATp+50)/dt)]
        # create time vector
        t = np.linspace(0-TATp, 50, len(y[0]))
        plt.close('all')
        fig = plt.figure()
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[2].set_xlabel('time in s')
        ax[1].set_ylabel('normalised Amplitude')
        for ii, x in enumerate(ax):
            x.plot(t, y[ii][:], color="k")
            x.axvline(color='r')
            x.set_xlim(-20, 50)
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
        # ax.legend(frameon=False, loc='upper left', ncol=2, handlelength=4)
        # plt.legend(['it_max=4000', 'it_max=400'])
        plt.savefig(os.path.join(outloc + file[:-5] + "pdf"), dpi=dpi)