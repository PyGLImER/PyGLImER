'''
Shared building blocks of the two preprocessing pipelines
(:mod:`~pyglimer.waveform.preprocess` for mseed databases and
:mod:`~pyglimer.waveform.preprocessh5` for hdf5 databases):
program-specific exceptions and the rotation + SNR quality control core.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import obspy
from obspy import Stream
from obspy.core.utcdatetime import UTCDateTime

from .qc import qcp, qcs


@dataclass
class PreprocessConfig:
    """
    Static settings of a preprocessing run. Bundles the parameters that
    are passed down unchanged through the event and waveform loops of
    both preprocessing pipelines.
    """
    phase: str
    rot: str
    pol: str
    taper_perc: float
    taper_type: str
    model: obspy.taup.TauPyModel
    logger: logging.Logger
    rflogger: logging.Logger
    eh: bool
    tz: float
    ta: float
    statloc: str
    rawloc: str
    preproloc: str
    rfloc: str
    deconmeth: str
    hc_filt: float
    netrestr: str
    statrestr: str
    remove_response: bool


@dataclass
class EventContext:
    """
    Per-event data and paths used by the mseed preprocessing loop.
    """
    event: obspy.core.event.Event
    origin_time: UTCDateTime
    ot_fiss: str
    evtlat: float
    evtlon: float
    depth: float
    prepro_folder: str
    by_event: str


class SNRError(Exception):
    """raised when the SNR is too high"""
    # Constructor method

    def __init__(self, value):
        self.value = value
    # __str__ display function

    def __str__(self):
        return repr(self.value)


class StreamLengthError(Exception):
    """raised when stream has fewer than 3 components"""
    # Constructor method

    def __init__(self, value):
        self.value = value
    # __str__ display function

    def __str__(self):
        return repr(self.value)


def rotate_and_snr_qc(
    phase: str, st: Stream, station_inv, baz: float, tz: float
) -> Tuple[Stream, bool, float, np.ndarray, float, float]:
    """
    Rotate the stream to ZNE and subsequently RT, normalise it, reduce it
    to its 3 primary components, and evaluate the SNR criteria.

    :param phase: Main phase ("P" or "S").
    :param st: Stream to process (modified in place by the rotations).
    :param station_inv: Inventory attached to the stream's station.
    :param baz: Back azimuth towards the event.
    :param tz: Time window before the first arrival (QC window).
    :return: The processed stream, whether it passed the SNR criteria,
        the corner frequency used by the QC, the noise matrix, the
        sampling interval, and the sampling rate.
    """
    st.rotate(method='->ZNE', inventory=station_inv)

    st.rotate(method='NE->RT', inventory=station_inv,
              back_azimuth=baz)
    st.normalize()

    # Sometimes streams contain more than 3 traces:
    if st.count() > 3:
        stream = {}
        for tr in st:
            stream[tr.stats.component] = tr
        if "Z" in stream:
            st = Stream([stream["Z"], stream["R"], stream["T"]])
        elif "3" in stream:
            st = Stream([stream["3"], stream["R"], stream["T"]])
        del stream

    # SNR CRITERIA
    dt = st[0].stats.delta  # sampling interval
    sampling_f = st[0].stats.sampling_rate

    if phase[-1] == "P":
        st, crit, f, noisemat = qcp(st, dt, sampling_f, tz)
    elif phase[-1] == "S":
        st, crit, f, noisemat = qcs(st, dt, sampling_f, tz)

    return st, crit, f, noisemat, dt, sampling_f


def set_station_info(
        infodict: dict, network: str, station: str, station_inv, dt: float,
        sampling_f: float):
    """
    Record the station and sampling metadata in the info dictionary that
    is later written into the shelve info files.
    """
    infodict['dt'] = dt
    infodict['sampling_rate'] = sampling_f
    infodict['network'] = network
    infodict['station'] = station
    infodict['statlat'] = station_inv[0][0][0].latitude
    infodict['statlon'] = station_inv[0][0][0].longitude
    infodict['statel'] = station_inv[0][0][0].elevation
