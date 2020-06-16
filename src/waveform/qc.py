#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains quality control for waveforms used for receiver function creation.

Created on Fri Apr 10 11:38:21 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""

import numpy as np
from obspy.signal import filter

# QC parameters

# filter frequencies for SNR check, all in Hz
lowco = [.03, .1, .5]  # only for PRF

lowcoS = .01  # Changed 18.05 to compare with Hopper et. al. 2018
highco = np.linspace(.33, .25, 4)  # For SRFs, 16.06.2020 from .175 to .25 Hz

# SNR criteria for QC
SNR_criteriaP = [7.5, 1, 10]  # [snrr, snrr2/snrr, snrz]

SNR_criteriaS = [24, .4, 1]  # QC4
# [primary/noise, sidelobe/primary, r/z conversions]


def qcp(st, dt, sampling_f, onset):
    """
    Quality control for the downloaded waveforms that are used to create
    PRFS. Works with various filters and SNR criteria

    Parameters
    ----------
    st : '~obspy.Stream'
        Input stream.
    dt : FLOAT
        Sampling interval [s].
    sampling_f : FLOAT
        Sampling frequency (Hz).
    onset : float
        Onset in seconds after trace's start.

    Returns
    -------
    st : '~obspy.Stream'
        Output stream. If stream was accepted, then this will contain the
        filtered stream, filtered with the broadest accepted filter.
    crit : BOOL
        True if stream was accepted, false if it wasn't.
    f : FLOAT
        Last used low-cut frequency. If crit=True, this will be the frequency
        for which the stream was accepted.
    noisemat : np.array
        SNR values in form of a matrix. Rows represent the different filters
        and columns the different criteria.

    """
    # Create stream dict to identify channels
    stream = {}
    for tr in st:
        stream[tr.stats.channel[2]] = tr.data
    ptn1 = round(5/dt)
    ptn2 = round((onset-5)/dt)
    nptn = ptn2-ptn1+1
    # First part of the signal
    pts1 = round(onset/dt)
    pts2 = round((onset+7.5)/dt)
    npts = pts2-pts1+1
    # Second part of the signal
    ptp1 = round((onset+15)/dt)
    ptp2 = round((onset+22.5)/dt)
    nptp = ptp2-ptp1+1

    # Then, I'll have to filter in the for-loop
    # and calculate the SNR

    # matrix to save SNR
    noisemat = np.zeros((len(lowco),
                         3), dtype=float)

    for ii, f in enumerate(lowco):
        ftcomp = filter.bandpass(stream["T"], f,
                                 4.99, sampling_f, corners=2, zerophase=True)
        frcomp = filter.bandpass(stream["R"], f,
                                 4.99, sampling_f, corners=2, zerophase=True)
        if "Z" in stream:
            fzcomp = filter.bandpass(stream["Z"], f, 4.99,
                                     sampling_f, corners=2, zerophase=True)
        elif "3" in stream:
            fzcomp = filter.bandpass(stream["3"], f, 4.99,
                                     sampling_f, corners=2, zerophase=True)

        # Compute the SNR for given frequency bands
        snrr = (sum(np.square(frcomp[pts1:pts2]))/npts)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        snrr2 = (sum(np.square(frcomp[ptp1:ptp2]))/nptp)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        snrz = (sum(np.square(fzcomp[pts1:pts2]))/npts)\
            / (sum(np.square(fzcomp[ptn1:ptn2]))/nptn)

        # Reject or accept traces depending on their SNR
        # #1: snr1 > 10 (30-37.5s, near P)
        # snr2/snr1 < 1 (where snr2 is 45-52.5 s, in the coda of P)
        # note: - next possibility might be to remove those events that
        # generate high snr between 200-250 s

        noisemat[ii, 0] = snrr
        noisemat[ii, 1] = snrr2/snrr
        noisemat[ii, 2] = snrz

        if snrr > SNR_criteriaP[0] and\
            snrr2/snrr < SNR_criteriaP[1] and\
                snrz > SNR_criteriaP[2]:  # accept
            crit = True

            # overwrite the old traces with the sucessfully filtered ones
            for tr in st:
                if tr.stats.channel[2] == "R":
                    tr.data = frcomp
                elif tr.stats.channel[2] == "Z"\
                    or tr.stats.channel[2] == "3" or\
                        tr.stats.channel[2] == "Z":
                    tr.data = fzcomp
                elif tr.stats.channel[2] == "T":
                    tr.data = ftcomp
            break  # waveform is accepted
        else:
            crit = False
    return st, crit, f, noisemat


def qcs(st, dt, sampling_f, onset):
    """
    Quality control for waveforms that are used to produce SRF. In contrast
    to the ones used for PRF this is a very rigid criterion and will reject
    >95% of the waveforms.

    Parameters
    ----------
    st : '~obspy.Stream'
        Input stream.
    dt : FLOAT
        Sampling interval [s].
    sampling_f : FLOAT
        Sampling frequency (Hz).
    onset : float
        Onset in seconds after trace's start.

    Returns
    -------
    st : '~obspy.Stream'
        Output stream. If stream was accepted, then this will contain the
        filtered stream, filtered with the broadest accepted filter.
    crit : BOOL
        True if stream was accepted, false if it wasn't.
    f : FLOAT
        Last used high-cut frequency. If crit=True, this will be the frequency
        for which the stream was accepted.
    noisemat : np.array
        SNR values in form of a matrix. Rows represent the different filters
        and columns the different criteria.

    """
    # Create stream dict to identify channels
    stream = {}
    for tr in st:
        stream[tr.stats.channel[2]] = tr.data

    ptn1 = round(5/dt)  # Hopefully relatively silent time
    ptn2 = round((onset-5)/dt)
    nptn = ptn2-ptn1
    # First part of the signal
    pts1 = round(onset/dt)  # theoretical arrival time
    pts2 = round((onset+10)/dt)
    npts = pts2-pts1
    # Second part of the signal
    ptp1 = round((onset+10)/dt)
    ptp2 = round((onset+35)/dt)
    nptp = ptp2-ptp1
    # part where the Sp converted arrival should be strong
    ptc1 = round((onset-50)/dt)
    ptc2 = round((onset-10)/dt)
    nptc = ptc2-ptc1

    # filter
    # matrix to save SNR
    noisemat = np.zeros((len(highco), 3), dtype=float)

    # Filter
    # At some point, I might also want to consider to change the lowcof
    for ii, hf in enumerate(highco):
        ftcomp = filter.bandpass(stream["T"], lowcoS,
                                 hf, sampling_f, corners=2, zerophase=True)
        frcomp = filter.bandpass(stream["R"], lowcoS,
                                 hf, sampling_f, corners=2, zerophase=True)
        if "Z" in stream:
            fzcomp = filter.bandpass(stream["Z"], lowcoS, hf,
                                     sampling_f, corners=2, zerophase=True)
        elif "3" in stream:
            fzcomp = filter.bandpass(stream["3"], lowcoS, hf,
                                     sampling_f, corners=2, zerophase=True)

        # Compute the SNR for given frequency bands
        # strength of primary arrival
        snrr = (sum(np.square(frcomp[pts1:pts2]))/npts)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        # how spiky is the arrival?
        snrr2 = (sum(np.square(frcomp[ptp1:ptp2]))/nptp)\
            / (sum(np.square(frcomp[ptn1:ptn2]))/nptn)
        # horizontal vs vertical
        # snrz = (sum(np.square(frcomp[pts1:pts2]))/npts)\
        #     / (sum(np.square(fzcomp[pts1:pts2]))/npts)
        snrc = (sum(np.square(frcomp[ptc1:ptc2]))/nptc)\
            / sum(np.square(fzcomp[ptc1:ptc2])/nptc)

        noisemat[ii, 0] = snrr
        noisemat[ii, 1] = snrr2/snrr
        noisemat[ii, 2] = snrc

        # accept if
        if snrr > SNR_criteriaS[0] and\
            snrr2/snrr < SNR_criteriaS[1] and\
                snrc < SNR_criteriaS[2]:
            # This bit didn't give good results
            # max(frcomp) == max(frcomp[round((onset-2)/dt):
            #                           round((onset+10)/dt)]):
            crit = True

            # overwrite the old traces with the sucessfully filtered ones
            for tr in st:
                if tr.stats.channel[2] == "R":
                    tr.data = frcomp
                elif tr.stats.channel[2] == "Z"\
                    or tr.stats.channel[2] == "3" or\
                        tr.stats.channel[2] == "Z":
                    tr.data = fzcomp
                elif tr.stats.channel[2] == "T":
                    tr.data = ftcomp
            break  # waveform is accepted
        else:
            crit = False
    return st, crit, hf, noisemat
