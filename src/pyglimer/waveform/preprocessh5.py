'''
This is a newer version of preprocess.py meant to be used with pyasdf.
Now, we will have to work in a very different manner than for .mseed files
and process files station wise rather than event wise.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 18th February 2021 02:26:03 pm
Last Modified: Wednesday, 11th August 2021 08:17:15 pm
'''

from glob import glob
import logging
import os
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed, cpu_count
import obspy
from obspy import Stream, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from pyasdf import ASDFDataSet

from pyglimer.utils.signalproc import resample_or_decimate
from pyglimer.database.rfh5 import RFDataBase
from .qc import qcp, qcs
from .rotate import rotate_LQT_min, rotate_PSV
from ..rf.create import RFStream, createRF


# program-specific Exceptions
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


def preprocessh5(
    phase, rot, pol, taper_perc, model, taper_type, tz, ta, rawloc, rfloc,
        deconmeth, hc_filt, netrestr, statrestr, logger, rflogger):
    os.makedirs(rfloc, exist_ok=True)

    # Open ds
    for f in glob(os.path.join(rawloc, '*.h5')):
        # Here we do multicore processing for each station rather than for
        # each event
        net, stat, _ = os.path.basename(f).split('.')
        code = '%s.%s' % (net, stat)
        with ASDFDataSet(f, mode='r') as ds:
            # get station inventory
            inv = ds.waveforms[code].StationXML
            rf = RFStream()
            for evt in ds.events:
                toa, rayp, rayp_s_deg, baz, distance = compute_toa(
                    evt, inv, phase, model)
                st = ds.get_waveforms(
                    net, stat, '*', '*', toa-tz, toa+ta, 'raw_recording')
                rf_temp = __station_process__(
                    st, inv, evt, phase, rot, pol, taper_perc, taper_type, tz,
                    ta, deconmeth, hc_filt, logger, rflogger, net, stat, baz,
                    distance, rayp, rayp_s_deg, toa)
                if rf_temp is not None:
                    rf.append(rf_temp)
                # Write regularly to not clutter too much into the RAM
                if rf.count() >= 100:
                    with RFDataBase(os.path.join(rfloc, code)) as rfdb:
                        rfdb.add_rf(rf)
                    rf.clear()
        with RFDataBase(os.path.join(rfloc, code)) as rfdb:
            rfdb.add_rf(rf)


def compute_toa(
    evt: obspy.core.event.Event, inv: obspy.core.inventory.inventory.Inventory,
        phase: str, model) -> Tuple[UTCDateTime, float, float, float]:
    origin = (evt.preferred_origin() or evt.origins[0])
    distance, baz, _ = gps2dist_azimuth(
        inv[0][0].latitude, inv[0][0].longitude, origin.latitude,
        origin.longitude)
    distance = kilometer2degrees(distance/1000)

    # compute time of first arrival & ray parameter
    arrival = model.get_travel_times(
        source_depth_in_km=origin.depth / 1000, distance_in_degree=distance,
        phase_list=[phase])[0]
    rayp_s_deg = arrival.ray_param_sec_degree
    rayp = rayp_s_deg / 111319.9  # apparent slowness
    toa = origin.time + arrival.time

    return toa, rayp, rayp_s_deg, baz, distance


def __station_process__(
    st, inv, evt, phase, rot, pol, taper_perc, taper_type, tz, ta,  deconmeth,
    hc_filt, logger, rflogger, net, stat, baz, distance, rayp,
        rayp_s_deg, toa):
    """
    Processing that is equal for each waveform recorded on one station
    """
    # Change dtype
    # for tr in st:
    #     np.require(tr.data, dtype=np.float64)
    #     tr.stats.mseed.encoding = 'FLOAT64'

    # Resample and Anti-Alias
    # is done before saving already
    # st = resample_or_decimate(st, 10)

    # Remove repsonse
    st.attach_response(inv)
    st.remove_response()

    # DEMEAN AND DETREND #
    st.detrend(type='demean')

    # TAPER #
    st.taper(
        max_percentage=taper_perc, type=taper_type, max_length=None,
        side='both')

    infodict = {}

    origin = (evt.preferred_origin() or evt.origins[0])
    ot_fiss = UTCDateTime(origin.time).format_fissures()
    ot_loc = UTCDateTime(origin.time, precision=-1).format_fissures()[:-6]

    # create RF
    try:
        st, crit, infodict = __rotate_qc(
            phase, st, inv, net, stat, baz, distance, ot_fiss, evt,
            origin.latitude, origin.longitude, origin.depth, rayp_s_deg, toa,
            logger, infodict, tz, pol)
        if hc_filt:
            st.filter('lowpass', freq=hc_filt, zerophase=True, corners=2)
        # Rotate to LQT or PSS
        if rot == "LQT":
            st, ia = rotate_LQT_min(st, phase)
            # additional QC
            if ia < 5 or ia > 75:
                crit = False
                raise SNRError("""The estimated incidence angle is
                                unrealistic with """ + str(ia) + 'degree.')

        elif rot == "PSS":
            _, _, st = rotate_PSV(
                inv[0][0][0].latitude,
                inv[0][0][0].longitude,
                rayp, st, phase)

        # Create RF object
        if phase[-1] == "S":
            trim = [40, 0]
            if distance >= 70:
                trim[1] = ta - (-2*distance + 180)
            else:
                trim[1] = ta - 40
        elif phase[-1] == "P":
            trim = False

        RF = createRF(
            st, phase, pol=pol, info=infodict, trim=trim,
            method=deconmeth)

    except SNRError as e:
        rflogger.info(e)
        return None

    except Exception as e:
        print("RF creation failed")
        rflogger.exception([net, stat, ot_loc, e])
        return None

    return RF


def __rotate_qc(
    phase, st, station_inv, network, station, baz, distance, ot_fiss,
    event, evtlat, evtlon, depth, rayp_s_deg, first_arrival, logger, infodict,
        tz, pol):
    """REMOVE INSTRUMENT RESPONSE + convert to vel + SIMULATE
    Bugs occur here due to station inventories without response information
    Looks like the bulk downloader sometimes donwnloads
    station inventories without response files. I could fix that here by
    redownloading the response file (alike to the 3 traces problem)"""

    st.rotate(method='->ZNE', inventory=station_inv)

    st.rotate(method='NE->RT', inventory=station_inv,
              back_azimuth=baz)
    st.normalize()

    # Sometimes streams contain more than 3 traces:
    if st.count() > 3:
        stream = {}
        for tr in st:
            stream[tr.stats.channel[2]] = tr
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
        if not crit:
            infodict['dt'] = dt
            infodict['sampling_rate'] = sampling_f
            infodict['network'] = network
            infodict['station'] = station
            infodict['statlat'] = station_inv[0][0][0].latitude
            infodict['statlon'] = station_inv[0][0][0].longitude
            infodict['statel'] = station_inv[0][0][0].elevation
            raise SNRError(np.array2string(noisemat))

    elif phase[-1] == "S":
        st, crit, f, noisemat = qcs(st, dt, sampling_f, tz)

        if not crit:
            infodict['dt'] = dt
            infodict['sampling_rate'] = sampling_f
            infodict['network'] = network
            infodict['station'] = station
            infodict['statlat'] = station_inv[0][0][0].latitude
            infodict['statlon'] = station_inv[0][0][0].longitude
            infodict['statel'] = station_inv[0][0][0].elevation
            raise SNRError(np.array2string(noisemat))

    # WRITE AN INFO FILE
    # append_info: [key,value]
    append_inf = [
        ['magnitude', (
            event.preferred_magnitude() or event.magnitudes[0])['mag']],
        ['magnitude_type', (
            event.preferred_magnitude() or event.magnitudes[0])[
                'magnitude_type']],
        ['evtlat', evtlat], ['evtlon', evtlon],
        ['ot_ret', ot_fiss], ['ot_all', ot_fiss],
        ['evt_depth', depth],
        ['evt_id', event.get('resource_id')],
        ['noisemat', noisemat],
        ['co_f', f], ['npts', st[1].stats.npts],
        ['rbaz', baz],
        ['rdelta', distance],
        ['rayp_s_deg', rayp_s_deg],
        ['onset', first_arrival],
        ['starttime', st[0].stats.starttime],
        ['pol', pol]
        ]

    # Check if values are already in dict
    for key, value in append_inf:
        infodict.setdefault(key, []).append(value)

    infodict['dt'] = dt
    infodict['sampling_rate'] = sampling_f
    infodict['network'] = network
    infodict['station'] = station
    infodict['statlat'] = station_inv[0][0][0].latitude
    infodict['statlon'] = station_inv[0][0][0].longitude
    infodict['statel'] = station_inv[0][0][0].elevation

    logger.info("Stream accepted. Preprocessing successful")

    return st, crit, infodict
