'''
Files contains all Errorhandler for the Glimer to obspy project


:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Saturday, 21th March 2020 19:16:41
Last Modified: Friday, 20th January 2023 03:50:29 pm
'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from obspy.clients.fdsn import Client, header  # web sevice

from .. import tmp


def redownload(network, station, starttime, endtime, st):
    """Errorhandler that Redownloads the stream for the given input.
    Used when the stream has less than three channels."""
    for c in tmp.re_client:
        try:
            client = Client(c)
            if len(st) < 3:
                raise ValueError
            else:
                break
        except ValueError:
            try:
                st = client.get_waveforms(
                    network, station, '*', st[0].stats.channel[0:2] + '*',
                    starttime, endtime)
            except (header.FDSNNoDataException, header.FDSNException,
                    ValueError):
                continue  # wrong client
        if len(st) == 3:
            break
    return st


def redownload_statxml(st, network, station, statfile):
    """Errorhandler: Redownload station xml in case that it is not found."""
    for c in tmp.re_client:
        try:
            client = Client(c)
            station_inv = client.get_stations(level="response",
                                              network=network, station=station)
            # write the new, working stationxml file
            station_inv.write(statfile, format="STATIONXML")
            break
        except (header.FDSNNoDataException, header.FDSNException):
            pass  # wrong client
    return station_inv


def NoMatchingResponseHandler(st, network, station, statloc):
    """Error handler for when the No matching response found error occurs."""
    for c in tmp.re_client:
        try:
            client = Client(c)
            station_inv = client.get_stations(
                level="response", channel=st[0].stats.channel[0:2] + '*',
                network=network, station=station)
            st.remove_response(inventory=station_inv, output='VEL',
                               water_level=60)
            # write the new, working stationxml file
            station_inv.write(
                os.path.join(statloc, network + "." + station + ".xml"),
                format="STATIONXML")
            return station_inv, st
        except (header.FDSNNoDataException, header.FDSNException):
            pass  # wrong client
        except Exception:
            break  # the response file doesn't seem to be available at all
    return None, None


def NotLinearlyIndependentHandler(st, network, station, starttime, endtime,
                                  station_inv, paz_sim):
    for c in tmp.re_client:
        client = Client(c)
        while len(st) < 3:
            try:
                st = client.get_waveforms(network, station, '*',
                                          st[0].stats.channel[0:2] + '*',
                                          starttime, endtime)
                st.remove_response(inventory=station_inv, output='VEL',
                                   water_level=60)
                st.remove_sensitivity(inventory=station_inv)
                st.simulate(paz_remove=None, paz_simulate=paz_sim,
                            simulate_sensitivity=True)
                st.rotate(method='->ZNE', inventory=station_inv)
            except (header.FDSNNoDataException, header.FDSNException):
                continue  # wrong client chosen
        return st
