#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:28:38 2020
Contains various functions used to evaluate the quality of RF and waveforms
@author: pm

"""
import os
from pathlib import Path
import shelve
import subprocess
import fnmatch

# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import numpy as np
from obspy import read, UTCDateTime
from obspy.taup import TauPyModel
from obspy.clients.iris import Client
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

import config
from ..rf.create import createRF, RFStream
from ..waveform.qc import qcs, qcp
from ..waveform.rotate import rotate_PSV, rotate_LQT, rotate_LQT_min

rating = {}  # mutable object


def rate(network, station=None, onset=config.tz, phase=config.phase,
         review=False, retained=False, decon_meth=config.decon_meth,
         test_tt_calculation=False):
    """
    Module to rate and review the quality of Sp or Ps waveforms from the given
    station. Shows the automatic rating from. Tapers can be controlled with
    the mouse. Rating is done with the keyboard.

    Parameters
    ----------
    network : STRING
        Network code (two letters).
    station : str or list, optional
        Station code (three letters).
    onset : float, optional
        Difference between starttime and theoretical first arrival.
        The default is config.tz.
    phase : STRING, optional
        Either "P" for Ps or "S" Sp. The default is "S".
    review : INTEGER, optional
        If true, already rated waveforms are shown.
        Can also be an integer between 1 and 4. Then, only waveforms rated
        with the respected rating are shown. The default is False.
    retained : Bool, optional
        Show only waveforms retained by qcp or qcs. The default is False.
    decon_meth : STRING, optional
        Deconvolution method, "waterlevel", 'dampedf' for constant
        damping level, 'it' for iterative time domain deconvoltuion, 'multit'
        for multitaper or 'fqd' for frequency dependently damped spectral
        division. The default is config.decon_meth.
        The default is config.decon_meth.

    Raises
    ------
    Exception
        For Typing mistakes.

    Returns
    -------
    None.

    """

    inloc = os.path.join(config.outputloc[:-1], phase, 'by_station', network)

    infiles = []  # List of all files in folder
    pattern = []  # List of input constraints
    streams = []  # List of files filtered for input criteria

    for root, dirs, files in os.walk(inloc):
        for name in files:
            infiles.append(os.path.join(root, name))

    # Set filter patterns

    if station:
        if type(station) == str:
            pattern.append('*%s*%s*.mseed' % (network, station))
        elif type(station) == list:
            for stat in station:
                pattern.append('*%s*%s*.mseed' % (network, stat))
    else:
        pattern.append('*%s*.mseed' % (network))

    # Do filtering
    for pat in pattern:
        streams.extend(fnmatch.filter(infiles, pat))

    # clear memory
    del pattern, infiles

    # For TauPy lookup
    model = TauPyModel()

    if test_tt_calculation:
        client = Client()

    for f in streams:
        # if file[:4] == "info":  # Skip the info files
        #     continue
        # try:
        #     st = read(inloc + file)
        # except IsADirectoryError as e:
        #     print(e)
        #     continue

        st = read(f)
        # List for taper coordinates
        if "taper" in rating:
            del rating["taper"]
        rating["taper"] = []
        st.normalize()

        # Additional filter "test"
        if phase == "S":
            st.filter("lowpass", freq=1.0, zerophase=True, corners=2)
        elif phase == "P":
            st.filter("lowpass", freq=1.0, zerophase=True, corners=2)

        y = []
        ch = []
        starttime = str(st[0].stats.starttime)
        dt = st[0].stats.delta
        sampling_f = st[0].stats.sampling_rate
        old = __r_file(network, st[0].stats.station, starttime)  # old

        # read info file
        # location info file
        infof = os.path.join(inloc, st[0].stats.station, 'info')
        with shelve.open(infof) as info:
            ii = info["starttime"].index(st[0].stats.starttime)
            rdelta = info["rdelta"][ii]  # epicentral distance
            mag = info["magnitude"][ii]
            statlat = info["statlat"]
            statlon = info["statlon"]
            rayp = info["rayp_s_deg"][ii] / 111319.9
            evt_depth = info["evt_depth"][ii] / 1000
            ot = info["ot_ret"][ii]
            evtlat = info['evtlat'][ii]
            evtlon = info['evtlon'][ii]

        if old and not review:  # skip already rated files
            continue
        if type(review) == int:
            if review > 4:
                raise Exception("review has to be between 0 and 4")
            if review != int(old):
                continue

        # check automatic rating
        if phase == "S":
            st_f, crit, hf, noisemat = qcs(st, dt, sampling_f, onset=onset)
        elif phase == "P":
            st_f, crit, hf, noisemat = qcp(st, dt, sampling_f, onset=onset)

        # skip files that have not been retained
        if retained and not crit:
            continue

        # create RF
        if not test_tt_calculation:
            _, _, PSV = rotate_PSV(statlat, statlon, rayp, st_f)
        else:
            PSV = st_f  # to see correlation at theoretical arrival

        try:
            RF = createRF(PSV, phase, method=decon_meth, shift=onset)
        except ValueError as e:  # There were some problematic events
            print(e)
            continue

        # TauPy lookup
        ph_name = []
        ph_time = []

        if not test_tt_calculation:
            arrivals = model.get_travel_times(evt_depth,
                                              distance_in_degree=rdelta)

            primary_time = model.get_travel_times(evt_depth,
                                                  distance_in_degree=rdelta,
                                                  phase_list=phase)[0].time

            for arr in arrivals:
                if arr.time < primary_time - onset or arr.time > \
                        primary_time + config.ta or arr.name == phase:
                    continue
                ph_name.append(arr.name)
                ph_time.append(arr.time - primary_time)

        else:
            # Caluclate travel times with different methods
            ph_time.append(model.get_travel_times_geo(
                evt_depth, evtlat, evtlon, statlat, statlon,
                phase_list=phase)[0].time)

            d = []
            d.append(client.distaz(
                statlat, statlon, evtlat, evtlon)['distance'])

            d.append(kilometer2degrees(
                gps2dist_azimuth(statlat, statlon, evtlat, evtlon)[0]/1000))

            for dis in d:
                ph_time.append(model.get_travel_times(
                    evt_depth, dis, phase_list=phase)[0].time)
            ph_name = ['taup', 'iris', 'geodetics']
            ph_time = np.array(ph_time) - (st[0].stats.starttime + onset - \
                                           UTCDateTime(ot))

        # waveform data
        st.sort()
        for tr in st:
            y.append(tr.data)
            ch.append(tr.stats.channel[2])

        # create time vector
        t = np.linspace(0 - onset, tr.stats.npts * tr.stats.delta - onset,
                        len(y[0]))

        # plot
        fig, ax = __draw_plot(starttime, t, y, ph_time, ph_name, ch, noisemat,
                              RF, old, rdelta, mag, crit, ot, evt_depth, phase,
                              test_tt_calculation)
        while not plt.waitforbuttonpress(30):
            # Taper when input is there
            if len(rating["taper"]) == 2:
                if rating["taper"][0] < rating["taper"][1]:
                    trim = [rating["taper"][0], rating["taper"][1]]
                else:
                    trim = [rating["taper"][1], rating["taper"][0]]
                trim[0] = trim[0] - t[0]
                trim[1] = t[-1] - trim[1]

                RF = createRF(PSV, phase, method=decon_meth, shift=onset,
                              trim=trim)

                __draw_plot(starttime, t, y, ph_time, ph_name, ch, noisemat,
                            RF, old, rdelta, mag, crit, ot, evt_depth, phase,
                            test_tt_calculation)
                rating["taper"].clear()
        # fig.canvas.mpl_connect('key_press_event', ontype)
        if rating["k"] == "q":
            break
        elif "k" in rating:
            __w_file(network, st[0].stats.station, starttime)


def __draw_plot(starttime, t, y, ph_time, ph_name, ch, noisemat, RF, old,
                rdelta, mag, crit, ot, evt_depth, phase, test_tt_calculation):
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
        if test_tt_calculation:
            x.set_xlim(-10,10)
    # if old:
    #     ax[2].text(0, 1, old,
    #                bbox=dict(facecolor='green', alpha=0.5))
    #     ax[2].set_xlabel(['time in s, old rating', old])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[2].text(0.05, 0.95, str(noisemat), transform=ax[2].transAxes,
               fontsize=10, verticalalignment='top', bbox=props)
    # show RF
    if phase == 'S':
        ax[3].plot(t, -np.flip(RF.data), color='k')
    elif phase == 'P':
        ax[3].plot(t, RF.data, color='k')
    ax[3].set_title(str(ot))
    ax[3].set_xlabel('time in s')
    ax[3].set_xlim(-10, 40)

    # textbox
    if old:
        textstr = '\n'.join((
            r'$\Delta=%.2f$' % (rdelta,),
            r'$\mathrm{mag}=%.2f$' % (mag,),
            r'$\mathrm{z}=%.2f$' % (evt_depth,),
            r'$\mathrm{rat_{man}}=%.2f$' % (int(old),),
            r'$\mathrm{rat_{auto}}=%.2f$' % (crit,)))
    else:
        textstr = '\n'.join((
            r'$\Delta=%.2f$' % (rdelta,),
            r'$\mathrm{mag}=%.2f$' % (mag,),
            r'$\mathrm{z}=%.2f$' % (evt_depth,),
            r'$\mathrm{rat_{auto}}=%.2f$' % (crit,)))
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
    fig.canvas.mpl_connect('key_press_event', __ontype)
    fig.canvas.mpl_connect('button_press_event', __onclick)
    print("1: discard, 2: ok, 3: good, 4: very good, 5: keep old rating",
          "quit with q. Click on figure to taper (first left then right)")
    return fig, ax


def __ontype(event):
    """Deals with keyboard input"""
    rating["k"] = event.key


def __onclick(event):
    """Deals with mouse input"""
    rating["taper"].append(event.xdata)


def __w_file(network, station, starttime):
    try:
        int(rating["k"])
    except ValueError:
        return False
    with shelve.open(config.ratings + network + "." + station + "rating") as f:
        if int(rating["k"]) != 5:
            f[starttime] = rating["k"]
            print("You've rated the stream", rating["k"])


def __r_file(network, station, starttime):
    with shelve.open(config.ratings + network + "." + station + "rating") as f:
        if starttime in f:
            old = f[starttime]
        else:
            old = False
    return old


def sort_rated(network, station, phase=config.phase):
    """
    Functions that sorts waveforms in 4 folders,
    depending on their manual rating.

    Parameters
    ----------
    network : STRING
        Network code (2 letters).
    station : STRING
        Station code (3 letters).
    phase : STRING, optional
        "P" or "S". The default is config.phase.

    Returns
    -------
    None.

    """

    inloc = config.outputloc[:-1] + phase + "/by_station/" + network + \
            '/' + station + '/'
    for n in range(1, 5):
        subprocess.call(["mkdir", "-p", inloc + str(n)])
    dic = shelve.open(config.ratings + network + "." + station + "rating")
    for file in os.listdir(inloc):
        if file[:4] == "info":  # Skip the info files
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


def automatic_rate(network, station, phase=config.phase):
    """
    Checks the automatic QC criteria for SRF waveforms.

    Parameters
    ----------
    network : STRING
        Network code (2 letters).
    station : STRING
        Station code (3 letters).
    phase : STRING, optional
        "P" or "S". The default is config.phase.

    Returns
    -------
    diff : INTEGER
        Number of waveforms that were not rated 3 or 4.
    ret : INTEGER
        Number of automatically retained waveforms.
    sts : LIST
        List containing all retained + filtered streams.
    crits : LIST
        List containing bools (retained or not) corresponding to streams in
        sts.

    """

    inloc = config.outputloc[:-1] + phase + "/by_station/" + network + \
            '/' + station + '/'
    diff = 0
    ret = 0
    sts = []
    crits = []
    for file in os.listdir(inloc):
        if file[:4] == "info":  # Skip the info files
            continue
        try:
            st = read(inloc + file)
        except IsADirectoryError as e:
            print(e)
            continue
        starttime = str(st[0].stats.starttime)
        if phase == "S":
            st, crit, hf, noisemat = qcs(st, st[0].stats.delta,
                                         st[0].stats.sampling_rate)
        elif phase == "P":
            st, crit, lf, noisemat = qcp(st, st[0].stats.delta,
                                         st[0].stats.sampling_rate)

        with shelve.open(config.ratings + network + "." + station
                         + "rating") as f:
            f[starttime + "_auto"] = crit
            if starttime in f and int(f[starttime]) < 3 and crit:
                diff = diff + 1
            if crit:
                ret = ret + 1
        sts.append(st)
        crits.append(crit)
    return diff, ret, sts, crits


def sort_auto_rated(network, station, phase=config.phase):
    """
    Puts all retained RF and waveforms in one folder.

    Parameters
    ----------
    network : STRING
        Network code (2 letters).
    station : STRING
        Station code (3 letters).
    phase : STRING, optional
        "P" or "S". The default is config.phase.
    """

    inloc = config.outputloc[:-1] + phase + "/by_station/" + network + \
            '/' + station + '/'
    inloc_RF = config.RF[:-1] + phase + '/' + network + '/' + station + '/'
    subprocess.call(["mkdir", "-p", inloc + "ret"])
    subprocess.call(["mkdir", "-p", inloc_RF + "ret"])
    dic = shelve.open(config.ratings + network + "." + station + "rating")
    for file in os.listdir(inloc):
        if file[:4] == "info":  # Skip the info files
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
                    decon_meth=config.decon_meth, rot="PSS"):
    """
    Does a quality control on the downloaded files and shows a plot of the
    stacked receiver function. Just meant to facilitate
    the decision for QC parameters (config.SNR_criteria).


    Parameters
    ----------
    network : STRING
        Network code (2 letters).
    station : STRING
        Station code (3 letters).
    phase : STRING, optional
        "P" or "S". The default is config.phase.
    decon_meth : STRING, optional
        Deconvolution method, "waterlevel", 'dampedf' for constant
        damping level, 'it' for iterative time domain deconvoltuion, 'multit'
        for multitaper or 'fqd' for frequency dependently damped spectral
        division. The default is config.decon_meth.
        The default is config.decon_meth.
    rot : STRING, optional
        Kind of rotation that should be performed.
        Either "PSS" for P-Sv-Sh, "LQT" for LQT via singular value
        decomposition or "LQT_min" for LQT by minimising first arrival energy
        on Q. The default is "LQT_min".

    Raises
    ------
    Exception
        For unknown inputs.

    Returns
    -------
    z : np.array
        Vector containing depths.
    stack : RFTrace
        RFTrace object containing the stacked RF.
    RF_mo : RFStream
        RFStream object containg depth migrated receiver functions.

    """

    diff, ret, sts, crits = automatic_rate(network, station, phase=phase)
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
    subprocess.call(['cp', info_file + '.bak', outloc])
    subprocess.call(['cp', info_file + '.dir', outloc])
    subprocess.call(['cp', info_file + '.dat', outloc])
    RFs = []
    for jj, st in enumerate(sts):
        if not crits[jj]:
            continue
        dt = st[0].stats.delta
        with shelve.open(info_file) as info:
            ii = info["starttime"].index(st[0].stats.starttime)
            rayp = info["rayp_s_deg"][ii] / 111319.9
            ot = info["ot_ret"][ii]
            evt_depth = info["evt_depth"][ii]
            rdelta = info["rdelta"][ii]
        # to reproduce Rychert et al (2007)
        # if int(ot[:4]) > 2002 or evt_depth > 1.5e5:
        #     ret = ret-1
        #     continue
        if rot == "PSS":
            _, _, st = rotate_PSV(statlat, statlon, rayp, st)
        elif rot == "LQT":
            st, rat = rotate_LQT(st)
            if 0.75 < rat < 1.5 and phase == "S":
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
        if phase == "S":
            trim = [40, 0]
            if rdelta >= 70:
                trim[1] = config.ta - (-2 * rdelta + 180)
            else:
                trim[1] = config.ta - 40
        elif phase == "P":
            trim = False
        info = shelve.open(info_file)
        try:
            RF = createRF(st, phase=phase, method=decon_meth,
                          shift=config.tz, trim=trim, info=info)
        except ValueError:
            print("corrupted event")
            continue
        RFs.append(RF)
        # with shelve.open(outloc+"info", writeback=True) as info:
        #     info["starttime"][ii] = st[0].stats.starttime
        #     info.sync()
    RF = RFStream(traces=RFs)
    # RF.write(outloc + station + '.sac', format="SAC")
    print("N files that were not 3/4: ", diff, "N retained: ", ret)
    # sta = station + "/test"
    # plot_stack(network, sta, phase=phase)
    z, stack, RF_mo = RF.station_stack()
    stack.plot()
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
    return z, stack, RF_mo, RF
