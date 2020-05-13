#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database management and overview for the GLImER database.

Created on Mon May 11 11:09:05 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""

import numpy as np
import logging
import os
import shelve
from copy import deepcopy

import pandas as pd

import config


class StationDB(object):
    def __init__(self):
        # 1. Initiate logger
        self.logger = logging.Logger(
            "src.database.stations.StationDBaseLogger")
        self.logger.setLevel(logging.WARNING)

        # FileHandler
        fh = logging.FileHandler('logs/StationDBase.log')
        fh.setLevel(logging.WARNING)
        self.logger.addHandler(fh)

        # Formatter
        fmt = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)

        # Create database
        self.create()

    def create(self):
        # Create data dictionary
        dataP = {
            'network': [], 'station': [], 'lat': [], 'lon': [],
            'elevation': [], 'NP': [], 'NPret': [], 'NS': [], 'NSret': []
            }
        dataS = deepcopy(dataP)

        # Read in info files
        folderP = os.path.join(config.outputloc[:-1], 'P', 'by_station')
        folderS = os.path.join(config.outputloc[:-1], 'S', 'by_station')

        # Check data availability of PRFs
        for root, _, files in os.walk(folderP):
            if 'info.dat' not in files:
                continue  # Skip parent folders
            infof = (os.path.join(root, 'info'))

            with shelve.open(infof, flag='r') as info:
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
        for i, station in enumerate(dataS['station']):
            if station in dataP['station']:
                j = dataP['station'].index(station)
                dataP['NS'][j] = dataS['NS'][i]
                dataP['NSret'][j] = dataS['NSret'][i]
            else:
                dataP['network'].append(dataS['network'][i])
                dataP['station'].append(station)
                dataP['lat'].append(dataS['lat'][i])
                dataP['lon'].append(dataS['lon'][i])
                dataP['elevation'].append(dataS['elevation'][i])
                dataP['NS'].append(dataS['NS'][i])
                dataP['NSret'].append(dataS['NSret'][i])
                dataP['NP'].append(0)
                dataP['NPret'].append(0)
        del dataS

        self.db = pd.DataFrame.from_dict(dataP)

    def geo_boundary(self, lat, lon):
        a = self.db['lat'] >= lat[0]
        b = self.db['lat'] <= lat[1]
        c = self.db['lon'] >= lon[0]
        d = self.db['lon'] <= lon[1]
        subset = self.db[a & b & c & d]

        return subset

    def find_stations(self, lat, lon):
        subset = self.geo_boundary(lat, lon)
        return list(subset['network']), list(subset['station'])