#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:28:38 2020
Contains various functions used to evaluate the quality of RF and waveforms
@author: pm

"""
import os
from obspy import read
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from pathlib import Path
from subfunctions import config
import subprocess
import shelve
import time
from subfunctions.preprocess import QC_S
from subfunctions.createRF import createRF
from subfunctions.plot import plot_stack
from subfunctions.rotate import rotate_PSV, rotate_LQT,\
                                            rotate_LQT_min
from obspy.taup import TauPyModel
import time

rating = {}  # mutable object


def rate(network, station, onset=config.tz, phase="S", review=False,
         retained=False, decon_meth=config.decon_meth):
    """"Module to rate and review the quality of waveform from the given
    station. Shows the automatic rating from QC_S
    INPUT:
        network: network code
        station: station code
        phase: phase (only coded for S)
        review: review already manually rated, False, True, or rating (1,2,3,4)
        retained: show only automatically retained, Bool"""
    inloc = config.outputloc[:-1] + phase + "/by_station/" + network +\
        '/' + station + '/'
    # For TauPy lookup
    model = TauPyModel()
    for file in os.listdir(inloc):
        if file[:4] == "info":   # Skip the info files
            continue
        try:
            st = read(inloc + file)
        except IsADirectoryError as e:
            print(e)
            continue

        # List for taper coordinates
        if "taper" in rating:
            del rating["taper"]
        rating["taper"] = []
        st.normalize()
        y = []
        ch = []
        starttime = str(st[0].stats.starttime)
        dt = st[0].stats.delta
        sampling_f = st[0].stats.sampling_rate
        old = r_file(network, station, starttime)  # old
        # read info file
        with shelve.open(inloc+"info") as info:
            ii = info["starttime"].index(st[0].stats.starttime)
            rdelta = info["rdelta"][ii]  # epicentral distance
            mag = info["magnitude"][ii]
            statlat = info["statlat"]
            statlon = info["statlon"]
            rayp = info["rayp_s_deg"][ii]/111319.9
            evt_depth = info["evt_depth"][ii]/1000
        if old and not review:  # skip already rated files
            continue
        if type(review) == int:
            if review > 4:
                raise Exception("review has to be between 0 and 4")
            if review != int(old):
                continue

        # check automatic rating
        _, crit, hf, noisemat = QC_S(st, dt, sampling_f)
        if retained and not crit:  # skip files that have not been retained
            continue

        # create RF
        st_f = st.filter('bandpass', freqmin=.03, freqmax=hf, zerophase=True)
        _, _, PSV = rotate_PSV(statlat, statlon, rayp, st_f)
        RF = createRF(PSV, dt, phase, method=decon_meth)

        # TauPy lookup
        arrivals = model.get_travel_times(evt_depth,
                                          distance_in_degree=rdelta)
        primary_time = model.get_travel_times(evt_depth,
                                              distance_in_degree=rdelta,
                                              phase_list=phase)[0].time
        ph_name = []
        ph_time = []
        for arr in arrivals:
            if arr.time < primary_time-onset or arr.time >\
                    primary_time+config.ta or arr.name == phase:
                continue
            ph_name.append(arr.name)
            ph_time.append(arr.time-primary_time)

        # waveform data
        st.sort()
        for tr in st:
            y.append(tr.data)
            ch.append(tr.stats.channel[2])

        # shorten vector
        # create time vector
        t = np.linspace(0-onset, tr.stats.npts*tr.stats.delta-onset, len(y[0]))

        # plot
        fig, ax = draw_plot(starttime, t, y, ph_time, ph_name, ch, noisemat,
                            RF, old, rdelta, mag, crit)
        while not plt.waitforbuttonpress(30):
            # Taper when input is there
            if len(rating["taper"]) == 2:
                if rating["taper"][0] < rating["taper"][1]:
                    trim = [rating["taper"][0], rating["taper"][1]]
                else:
                    trim = [rating["taper"][1], rating["taper"][0]]
                trim[0] = trim[0] - t[0]
                trim[1] = t[-1] - trim[1]
                RF = createRF(PSV, dt, phase, method=decon_meth, trim=trim)
                draw_plot(starttime, t, y, ph_time, ph_name, ch, noisemat, RF,
                          old, rdelta, mag, crit)
                rating["taper"].clear()
        # fig.canvas.mpl_connect('key_press_event', ontype)
        if rating["k"] == "q":
            break
        elif "k" in rating:
            w_file(network, station, starttime)


def draw_plot(starttime, t, y, ph_time, ph_name, ch, noisemat, RF, old, rdelta,
              mag, crit):
    """Draws the plot for the rate function"""
    plt.close('all')
    fig, ax = plt.subplots(4, 1, sharex=False)
    ax[1].set_ylabel(starttime)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '.75', '.5', '.25',
              '#FF6666', '#FF6633', '#FFFF66', '#66FF66', '#66CCCC',
              '#00FFFF', '#3399FF', '#9966FF', '#FF66FF']
    for ii, x in enumerate(ax[:3]):
        x.plot(t, y[ii][:], color="k")
        for jj, t2 in enumerate(ph_time):
            x.axvline(x=t2, linewidth=0.35, label=ph_name[jj],
                      color=colors[jj])
        x.set_title(ch[ii])
        x.legend(loc='upper right', prop={"size": 8})
        x.yaxis.set_major_formatter(ScalarFormatter())
        x.yaxis.major.formatter._useMathText = True
        x.xaxis.set_major_formatter(ScalarFormatter())
        x.xaxis.major.formatter._useMathText = True
        x.yaxis.set_minor_locator(AutoMinorLocator(5))
        x.xaxis.set_minor_locator(AutoMinorLocator(5))
        # x.yaxis.set_label_coords(0.63, 1.01)
        x.yaxis.tick_right()
        # x.grid(color='c', which='both', linewidth=.25)
        x.set_ylabel('Amplitude')
    # if old:
    #     ax[2].text(0, 1, old,
    #                bbox=dict(facecolor='green', alpha=0.5))
    #     ax[2].set_xlabel(['time in s, old rating', old])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[2].text(0.05, 0.95, str(noisemat), transform=ax[2].transAxes,
               fontsize=10, verticalalignment='top', bbox=props)
    # show RF
    ax[3].plot(t, -np.flip(RF[0].data), color='k')
    ax[3].set_xlabel('time in s')
    ax[3].set_xlim(-5, 40)
    # textbox
    if old:
        textstr = '\n'.join((
            r'$\Delta=%.2f$' % (rdelta, ),
            r'$\mathrm{mag}=%.2f$' % (mag, ),
            r'$\mathrm{rat_{man}}=%.2f$' % (int(old), ),
            r'$\mathrm{rat_{auto}}=%.2f$' % (crit, )))
    else:
        textstr = '\n'.join((
            r'$\Delta=%.2f$' % (rdelta, ),
            r'$\mathrm{mag}=%.2f$' % (mag, ),
            r'$\mathrm{rat_{auto}}=%.2f$' % (crit, )))
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[3].text(0.05, 0.95, textstr, transform=ax[3].transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    ax[3].yaxis.set_major_formatter(ScalarFormatter())
    ax[3].yaxis.major.formatter._useMathText = True
    ax[3].xaxis.set_major_formatter(ScalarFormatter())
    ax[3].xaxis.major.formatter._useMathText = True
    ax[3].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[3].xaxis.set_minor_locator(AutoMinorLocator(5))
    # x.yaxis.set_label_coords(0.63, 1.01)
    ax[3].yaxis.tick_right()
    # x.grid(color='c', which='both', linewidth=.25)
    ax[3].set_ylabel('Amplitude')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.canvas.mpl_connect('key_press_event', ontype)
    fig.canvas.mpl_connect('button_press_event', onclick)
    print("1: discard, 2: ok, 3: good, 4: very good, 5: keep old rating",
          "quit with q. Click on figure to taper (first left then right)")
    return fig, ax


def ontype(event):
    """Deals with keyboard input"""
    rating["k"] = event.key


def onclick(event):
    """Deals with mouse input"""
    rating["taper"].append(event.xdata)


def w_file(network, station, starttime):
    try:
        int(rating["k"])
    except ValueError:
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
        # crit, hf, noisemat = None, None, None
        with shelve.open(network + "." + station + "rating") as f:
            f[starttime + "_auto"] = crit
            if starttime in f and int(f[starttime]) < 3 and crit:
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


def auto_rate_stack(network, station, phase=config.phase,
                    decon_meth=config.decon_meth, rot="LQT_min"):
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
    subprocess.call(['cp', info_file+'.bak', outloc])
    subprocess.call(['cp', info_file+'.dir', outloc])
    subprocess.call(['cp', info_file+'.dat', outloc])
    RFs = []
    for jj, st in enumerate(sts):
        if not crits[jj]:
            continue
        dt = st[0].stats.delta
        with shelve.open(info_file) as info:
            ii = info["starttime"].index(st[0].stats.starttime)
            rayp = info["rayp_s_deg"][ii]/111319.9
            ot = info["ot_ret"][ii]
            evt_depth = info["evt_depth"][ii]
        # to reproduce Rychert et al (2007)
        if int(ot[:4]) > 2002 or evt_depth > 1.5e5:
            ret = ret-1
            continue
        if rot == "PSS":
            _, _, st = rotate_PSV(statlat, statlon, rayp, st)
        elif rot == "LQT":
            st, rat = rotate_LQT(st)
            if rat > 0.5 and rat < 2:
                # The PCA did not work properly, L & Q too similar
                ret = ret - 1
                continue
        elif rot == "LQT_min":
            st, ia = rotate_LQT_min(st)
            if ia > 70 or ia < 5:
                # Unrealistic incidence angle
                ret = ret - 1
                continue
        else:
            raise Exception("Unknown rotation method rot=,", rot, """"Use
                            either 'PSS', 'LQT', or 'LQT_min'.""")
        # noise RF test
        trim = [40, 90]
        # st.trim(st[0].stats.starttime+trim[0], st[0].stats.endtime-trim[1])
        st.taper(0.5, side='left', max_length=trim[0])
        st.taper(0.5, side='right', max_length=trim[1])
        RF = createRF(st, dt, phase=phase, method=decon_meth,
                      shift=config.tz)
        RFs.append(RF[0].data)
        RF.write(outloc + ot + '.mseed', format="MSEED")
        # with shelve.open(outloc+"info", writeback=True) as info:
        #     info["starttime"][ii] = st[0].stats.starttime
        #     info.sync()
    print("N files that were not 3/4: ", diff, "N retained: ", ret)
    sta = station + "/test"
    plot_stack(network, sta, phase=phase)
    # plot unmigrated stack
    # RF_av = np.average(RFs, axis=0)
    # RF_av = -np.flip(RF_av)
    # t = np.linspace(-15,15, len(RF_av))
    # plt.close('all')
    # fig = plt.figure()
    # fig, ax = plt.subplots(1, 1, sharex=True)
    # # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    # ax.spines['bottom'].set_position("center")
    # # ax.spines['left'].set_visible(False)
    
    # # Eliminate upper and right axes
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    
    # # Show ticks in the left and lower axes only
    # ax.xaxis.set_ticks_position('bottom')
    # # ax.axvline(color='r')  # should plot at x = 0
    # ax.plot(t, RF_av, color="k", linewidth=1.5)
    # ax.set_ylabel('Amplitude')
    # ax.set_xlabel('time in s')
    # # ax.set_position([.5, .5, .1, .1])
    # x = ax
    # x.set_xlim(-15, 15)
    # x.set_ylim(-.5, .5)
    # x.xaxis.set_major_formatter(ScalarFormatter())
    # x.yaxis.major.formatter._useMathText = True
    # # x.yaxis.set_major_formatter(plt.NullFormatter())
    # x.xaxis.major.formatter._useMathText = True
    # x.yaxis.set_minor_locator(AutoMinorLocator(5))
    # x.xaxis.set_minor_locator(AutoMinorLocator(5))
    # return t, RF_av
