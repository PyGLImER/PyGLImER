#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Database management and overview for the PyGLImER database.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 12th February 2020 03:24:30 pm
Last Modified: Monday, 30th May 2022 04:29:39 pm
'''

import fnmatch
import logging
import os
import shelve
from copy import deepcopy
from typing import Tuple
import glob

from joblib import Parallel, delayed
from obspy.clients.fdsn import Client, header
import pandas as pd

from pyglimer.data import finddir
from pyglimer.database.rfh5 import RFDataBase
from pyglimer.plot.plot_map import plot_station_db


def redownload_missing_statxmls(clients, phase, statloc, rawdir, verbose=True):
    """
    Fairly primitive program that walks through existing raw waveforms in
    rawdir and looks for missing station xmls in statloc and, afterwards,
    attemps a redownload

    :param clients: List of clients (see obspy documentation for
        `~obspy.Client`).
    :type clients: list
    :param phase: Either "P" or "S", defines in which folder to look for
        mseeds.
    :type phase: str
    :param statloc: Folder, in which the station xmls are saved
    :type statloc: str
    :param rawdir: Uppermost directory, in which the raw mseeds are saved
        (i.e. the directory above the phase division).
    :type rawdir: str
    :param verbose: Show some extra information, by default True
    :type verbose: bool, optional
    """

    ex = os.listdir(statloc)

    # remove file identifier
    for i, xml in enumerate(ex):
        x = xml.split('.')
        ex[i] = x[0] + '.' + x[1]
    wavdir = os.path.join(rawdir, phase)

    Parallel(n_jobs=-1)(
        delayed(__client__loop__)(client, ex, wavdir, statloc)
        for client in clients)


def __client__loop__(client, existing, wavdir, statloc):
    client = Client(client)

    for _, _, files in os.walk(wavdir):
        for fi in files:
            f = fi.split('.')
            if f[-1] != 'mseed':
                continue
            code = f[0] + '.' + f[1]
            if code and code not in existing:
                out = os.path.join(statloc, code + '.xml')
                try:
                    stat_inv = client.get_stations(
                        network=f[0], station=f[1], level='response',
                        filename=out)
                    stat_inv.write(out, format="STATIONXML")
                    existing.append(code)
                except (header.FDSNNoDataException, header.FDSNException):
                    pass  # wrong client


class StationDB(object):
    def __init__(
        self, dir: str, phase: str = None, use_old: bool = False,
            hdf5: bool = True, logdir: str = None):
        """
        Creates a pandas database of all available receiver functions.
        This database is entirely based on the info files in the "preprocessed"
        folder. Make sure that the output folder is not corrupted before
        running this. Creating this database does not take much time, so there
        will be no option to save it, as it should be up to date.
        However, there is an export function

        :param dir: Parental folder, in which the preprocessed mseeds are
            saved (i.e. the folder above the phase division).
        :type dir: str
        :param phase: If just one of the primary phases should be checked -
            useful for computational efficiency, when creating ccp.
            Default is None.
        :type phase: str, optional
        :param use_old: When turned on it will read in the saved csv file.
            That is a lot faster, but it will obviously not update,
            defaults to False
        :type use_old: bool, optional
        :param hdf5: Use HDF5 database instead of station xmls? Defaults to
            True.
        :type hdf5: bool, optional
        :param logdir: Directory for log file
        :type logdr: str, optional
        """

        self.dir = dir

        if phase:
            self.phase = phase.upper()

        # 1. Initiate logger
        self.logger = logging.Logger(
            "pyglimer.database.stations.StationDBaseLogger")
        self.logger.setLevel(logging.WARNING)

        # FileHandler
        if not logdir:
            try:
                fh = logging.FileHandler(
                    os.path.join(
                        dir, os.pardir, os.pardir, 'logs', 'StationDBase.log'))
            except FileNotFoundError:
                os.makedirs(os.path.join(
                    dir, os.pardir, os.pardir, 'logs'), exist_ok=True)
                fh = logging.FileHandler(
                    os.path.join(
                        dir, os.pardir, os.pardir, 'logs', 'StationDBase.log'))
        else:
            fh = logging.FileHandler(os.path.join(logdir, 'StationDBase.log'))
        fh.setLevel(logging.WARNING)
        self.logger.addHandler(fh)

        # Formatter
        fmt = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)

        # Check if there is already a saved database
        oloc = os.path.join(finddir(), 'database.csv')

        if use_old and os.path.isfile(oloc):
            self.db = pd.read_csv(oloc)
        elif hdf5:
            self.db = self._create_from_hdf5()
        else:
            self.db = self._create_from_info()

        # Save Database, don't save if only one phase is requested newly
        if not phase:
            self.db.to_csv(oloc)

    def _create_from_hdf5(self) -> pd.DataFrame:
        """
        Creates a panda database from information read from hdf5 files.

        :return: Simple Pandas Dataframe with station info
        :rtype: pd.DataFrame
        """
        data = {
            'code': [], 'network': [], 'station': [], 'lat': [], 'lon': [],
            'elevation': []}
        for f in glob.glob(os.path.join(
                self.dir, '**', '*.*.h5'), recursive=True):
            net, stat, _ = os.path.basename(f).split('.')
            # For now, we only do coordinates. Never used anything else anyways
            with RFDataBase(f, mode='r') as rfdb:
                lat, lon, el = rfdb.get_coords(net, stat, self.phase)
            if not lat:
                continue
            data['network'].append(net)
            data['station'].append(stat)
            data['code'].append('%s.%s' % (net, stat))
            data['lat'].append(lat)
            data['lon'].append(lon)
            data['elevation'].append(el)

        return pd.DataFrame.from_dict(data)

    def _create_from_info(self) -> pd.DataFrame:
        """
        Create panda database.
        """
        if self.phase:
            # Create data dictionary
            data = {
                'code': [], 'network': [], 'station': [], 'lat': [], 'lon': [],
                'elevation': [], 'NP': [], 'NPret': [], 'NS': [], 'NSret': []
            }

            # oposing phase
            if self.phase == 'P':
                o = 'S'
            elif self.phase == 'S':
                o = 'P'
            elif self.phase == 'SKS':
                o = 'SKS'
            else:
                raise ValueError('Phase %s not supported.' % self.phase)

            folder = os.path.join(
                self.dir, self.phase, 'by_station')

            # Check data availability
            for root, _, files in os.walk(folder):
                if len(fnmatch.filter(files, 'info.*')) == 0:
                    continue  # Skip parent folders
                infof = (os.path.join(root, 'info'))

                with shelve.open(infof, flag='r') as info:
                    data['code'].append(info['network']+'.'+info['station'])
                    data['network'].append(info['network'])
                    data['station'].append(info['station'])
                    data['lat'].append(info['statlat'])
                    data['lon'].append(info['statlon'])
                    data['elevation'].append(info['statel'])
                    data['N'+self.phase].append(info['num'])
                    data['N'+o].append(0)
                    data['N'+o+'ret'].append(0)
                    try:
                        data['N'+self.phase+'ret'].append(info['numret'])
                    except KeyError:
                        data['N'+self.phase+'ret'].append(0)
                        self.logger.debug(
                            ['No P-waveforms retained for station',
                             info['network'] + '.' + info['station']])

            # Return dataframe
            return pd.DataFrame.from_dict(data)

        # Create data dictionary
        dataP = {
            'code': [], 'network': [], 'station': [], 'lat': [], 'lon': [],
            'elevation': [], 'NP': [], 'NPret': [], 'NS': [], 'NSret': []}
        dataS = deepcopy(dataP)

        # Read in info files
        folderP = os.path.join(self.dir, 'P', 'by_station')
        folderS = os.path.join(self.dir, 'S', 'by_station')

        # Check data availability of PRFs
        for root, _, files in os.walk(folderP):
            if 'info.dat' not in files:
                continue  # Skip parent folders
            infof = (os.path.join(root, 'info'))

            with shelve.open(infof, flag='r') as info:
                dataP['code'].append(info['network']+'.'+info['station'])
                dataP['network'].append(info['network'])
                dataP['station'].append(info['station'])
                dataP['lat'].append(info['statlat'])
                dataP['lon'].append(info['statlon'])
                dataP['elevation'].append(info['statel'])
                dataP['NP'].append(info['num'])
                dataP['NS'].append(0)
                dataP['NSret'].append(0)
                try:
                    dataP['NPret'].append(info['numret'])
                except KeyError:
                    dataP['NPret'].append(0)
                    self.logger.debug(
                        'No P-waveforms obtained for station %s.%s' % (
                            info['network'], info['station']))

        # Check data availability of SRFs
        for root, _, files in os.walk(folderS):
            if 'info.dat' not in files:
                continue  # Skip parent folders or empty folders
            infof = (os.path.join(root, 'info'))

            with shelve.open(infof, flag='r') as info:
                dataS['code'].append(info['network']+'.'+info['station'])
                dataS['network'].append(info['network'])
                dataS['station'].append(info['station'])
                dataS['lat'].append(info['statlat'])
                dataS['lon'].append(info['statlon'])
                dataS['elevation'].append(info['statel'])
                dataS['NS'].append(info['num'])
                try:
                    dataS['NSret'].append(info['numret'])
                except KeyError:
                    dataS['NSret'].append(0)
                    self.logger.debug(
                        'No S-waveforms obtaimed for station %s.%s' % (
                            info['network'], info['station']))

        # Merge the two dictionaries
        for i, c in enumerate(dataS['code']):
            if c in dataP['code']:
                j = dataP['code'].index(c)
                dataP['NS'][j] = dataS['NS'][i]
                dataP['NSret'][j] = dataS['NSret'][i]
            else:
                dataP['code'].append(c)
                dataP['network'].append(dataS['network'][i])
                dataP['station'].append(dataS['station'])
                dataP['lat'].append(dataS['lat'][i])
                dataP['lon'].append(dataS['lon'][i])
                dataP['elevation'].append(dataS['elevation'][i])
                dataP['NS'].append(dataS['NS'][i])
                dataP['NSret'].append(dataS['NSret'][i])
                dataP['NP'].append(0)
                dataP['NPret'].append(0)
        del dataS
        return pd.DataFrame.from_dict(dataP)

    def geo_boundary(
        self, lat: Tuple[float, float], lon: Tuple[float, float],
            phase: str = None) -> pd.DataFrame:
        """
        Return a subset of the database filtered by location.

        :param lat: Latitude boundaries in the form
            (minimum latitude, maximum lat).
        :type lat: Tuple
        :param lon: Longitude boundaries in the form
            (minimum longitude, maximum lon).
        :type lon: Tuple
        :param phase: 'P' or 'S'. If option is given, it will only return
            stations with accepted receiver functions for this phase.
            The default is None.
        :type phase: str, optional
        :return: Subset of the original DataFrame filtered by location.
        :rtype: pandas.DataFrame
        """
        a = self.db['lat'] >= lat[0]
        b = self.db['lat'] <= lat[1]
        c = self.db['lon'] >= lon[0]
        d = self.db['lon'] <= lon[1]
        if phase:
            phase = phase.upper()
            e = self.db['N'+phase+'ret'] > 0
            subset = self.db[a & b & c & d & e]
        else:
            subset = self.db[a & b & c & d]

        return subset

    def find_stations(self, lat, lon, phase=None):
        """
        Returns list of networks and stations inside a geoboundary
        """
        subset = self.geo_boundary(lat, lon, phase)
        return list(subset['network']), list(subset['station'])

    def plot(
        self, lat: Tuple[float, float] = None, lon: Tuple[float, float] = None,
        profile: Tuple[float, float] = None, p_direct: bool = True,
            outputfile: str = None, format: str = 'pdf', dpi: int = 300):
        """
        Plot the station coverage for a given area. Also allows plotting
        ccp-profiles.

        Parameters
        ----------
        lat : tuple or None, optional
            Latitude boundaries for the stations that should be included
            in the form (latmin, latmax), by default None.
        lon : tuple or None, optional
            Longitude boundaries for the stations that should be included
            in the form (lonmin, lonmax), by default None
        profile : list or tuple or None, optional
            Profile or profiles that will be drawn as lines on the projection,
            each profile is defined as a tuple: (lon1,lon2,lat1,lat2). Can be
            a list of such tuples. If p_direct is set to false, define the left
            and right corner the same way (then the plot is a rectangle),
            by default None.
        p_direct : bool, optional
            PLot the profile as a rectangle, by default True
        outputfile : str or None, optional
            Write the plot to the given file, by default None
        format : str, optional
            Output format if outputfile, by default 'pdf'
        dpi : int, optional
            Pixel density, by default 300

        Raises
        ------
        ValueError
            For wrong inputs.
        """
        if lat and lon:
            subset = self.geo_boundary(lat=lat, lon=lon)
            lat = (lat[0]-10, lat[1]+10)
            lon = (lon[0]-10, lon[1]+10)
            plot_station_db(
                list(subset['lat']), list(subset['lon']), lat=lat, lon=lon,
                profile=profile, p_direct=p_direct,
                outputfile=outputfile, format=format, dpi=dpi)
        elif not lat and not lon:
            plot_station_db(
                list(self.db['lat']), list(self.db['lon']), profile=profile,
                p_direct=p_direct,
                outputfile=outputfile, format=format, dpi=dpi)
        else:
            raise ValueError(
                'You have to provide both lat and lon as tuple or None.')
