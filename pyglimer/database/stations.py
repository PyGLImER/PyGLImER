#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database management and overview for the PyGLImER database.

Created on Mon May 11 11:09:05 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""

import fnmatch
import logging
import os
import pickle
import shelve
from copy import deepcopy
import time

from joblib import Parallel, delayed
import numpy as np
from obspy.clients.fdsn import Client, header
from obspy import read_inventory
import pandas as pd
from pathlib import Path

from pyglimer.data import finddir
from pyglimer.utils.utils import dt_string


def redownload_missing_statxmls(clients, phase, statloc, rawdir, verbose=True):
    """
    Fairly primitive program that walks through existing raw waveforms in
    rawdir and looks for missing station xmls in statloc and, afterwards,
    attemps a redownload

    :param clients: List of clients (see obspy documentation for
        `~obspy.Client`).
    :type clients: list
    :param phase: Either "P" or "S", defines in which folder to look for mseeds.
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

    out = Parallel(n_jobs=-1)(
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
                        network = f[0], station=f[1], level='response',
                        filename=out)
                    stat_inv.write(out, format="STATIONXML")
                    existing.append(code)
                except (header.FDSNNoDataException, header.FDSNException):
                    pass  # wrong client        

class StationDB(object):
    def __init__(self, preproloc, phase=None, use_old=False,
                 logdir:str or None = None):
        """
        Creates a pandas database of all available receiver functions.
        This database is entirely based on the info files in the "preprocessed"
        folder. Make sure that the output folder is not corrupted before
        running this. Creating this database does not take much time, so there
        will be no option to save it, as it should be up to date.
        However, there is an export function

        :param preproloc: Parental folder, in which the preprocessed mseeds are
            saved (i.e. the folder above the phase division).
        :type preproloc: str
        :param phase: If just one of the primary phases should be checked -
            useful for computational efficiency, when creating ccp.
            Default is None.
        :type phase: str, optional
        :param use_old: When turned on it will read in the saved csv file.
            That is a lot faster, but it will obviously not update,
            defaults to False
        :type use_old: bool, optional
        :param logdir: Directory for log file
        :type logdr: str, optional
        """        

        self.preproloc = preproloc
        
        if phase:
            self.phase = phase.upper()           
        #else:
        #    self.phase = phase
        
        # 1. Initiate logger
        self.logger = logging.Logger(
            "pyglimer.database.stations.StationDBaseLogger")
        self.logger.setLevel(logging.WARNING)

        # FileHandler
        if not logdir:
            fh = logging.FileHandler(os.path.join('logs', 'StationDBase.log'))
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
        
        if use_old and Path(oloc).is_file():
            self.db = pd.read_csv(oloc)
        else:
            self.db = self.__create__()     

        # Save Database, don't save if only one phase is requested newly
        if not phase:
            self.db.to_csv(oloc)

    def __create__(self):
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
            else:
                raise ValueError('Phase '+self.phase+' not supported.')
            
            folder = os.path.join(
                self.preproloc, self.phase, 'by_station')
            
            # Check data availability
            for root, _, files in os.walk(folder):
                if 'info.dat' not in files:
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
            'elevation': [], 'NP': [], 'NPret': [], 'NS': [], 'NSret': []
            }
        dataS = deepcopy(dataP)  

        # Read in info files
        folderP = os.path.join(self.preproloc, 'P', 'by_station')
        folderS = os.path.join(self.preproloc, 'S', 'by_station')

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
                    self.logger.debug(['No P-waveforms retained for station',
                                       info['network'] + '.' + info['station']])

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
                    self.logger.debug('No S-waveforms retained for station' +
                                        info['network'] + '.' + info['station'])


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

    def geo_boundary(self, lat, lon, phase=None):
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
    