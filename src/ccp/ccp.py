#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:10:18 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""

import pickle
import fnmatch
import os
import numpy as np

from obspy import read_inventory

import config
from ..rf.create import read_rf
from .compute.bin import BinGrid
from ..rf.moveout import res, maxz


def init_ccp(spacing, phase=config.phase, network=None, station=None,
             compute_stack=False, save=False, verbose=True):
    """
    Computes a ccp stack in self.ccp using the standard folder
    structure set in config. The stack can be limited to some networks and
    stations.

    Parameters
    ----------
    spacing : float
        Angular distance between each bin point.
    phase : str, optional
        Either 'S' or 'P'. Use 'S' if dataset contains both SRFs and PRFs.
        The default is config.phase.
    network : str or list, optional
        Network or networks that are to be included in the ccp stack.
        Standard unix wildcards are allowed. If None, the whole database
        will be used. The default is None.
    station : str or list, optional
        Station or stations that are to be included in the ccp stack.
        Standard unix wildcards are allowed. Can only be list if
        type(network)=str. If None, the whole database will be used.
        The default is None.
    compute_stack : Bool, optional
        If true it will compute the stack by calling ccp.compute_stack().
        That can take a long time! The default is False.
    save : Bool or str, optional
        Either False if the ccp should not be saved or string with filename
        will be saved in config.ccp. Will be saved as pickle file.
        The default is False.
    verbose : Bool, optional
        Display info in terminal. The default is True.

    Returns
    -------
    ccp : .ccp.CCPStack
        Returns a CCPStack object containing all information.

    """

    # create empty lists for station latitude and longitude
    lats = []
    lons = []

    # Were network and stations provided?
    if network and type(network) == list:
        files = []
        for net in network:
            pattern = net + '.' + (station or '') + '*'
            files.extend(fnmatch.filter(os.listdir(config.statloc), pattern))
    elif network and type(station) == list:
        files = []
        for stat in station:
            pattern = network + '.' + stat + '*'
            files.extend(fnmatch.filter(os.listdir(config.statloc), pattern))
    elif network:
        pattern = network + '.' + (station or '') + '*'
        files = fnmatch.filter(os.listdir(config.statloc), pattern)

    # In this case, it will process all available data (for global ccps)
    else:
        files = os.listdir(config.statloc)

    # read out station latitudes and longitudes
    for file in files:
        stat = read_inventory(config.statloc + '/' + file)
        lats.append(stat[0][0].latitude)
        lons.append(stat[0][0].longitude)

    ccp = CCPStack(lats, lons, spacing, phase=phase, verbose=verbose)

    if compute_stack:
        ccp.compute_stack(network=network, station=station, save=save)

    return ccp


def read_ccp(filename='ccp.pkl', folder=config.ccp, fmt=None):
    """
    Read CCP-Stack class file from input folder.

    Parameters
    ----------
    filename : str, optional
        Filename of the input file with file ending. The default is 'ccp.pkl'.
    folder : str, optional
        Input folder without concluding forward slash.
        The default is config.ccp.
    fmt : str, optional
        File format, can be none if the filename has an ending, possible
        options are "pickle. The default is None.

    Raises
    ------
    ValueError
        For wrong input.

    Returns
    -------
    ccp : ccp.CCPStack
        CCPStack object.

    """
    # Trying to identify filetype from ending:
    if not fmt:
        x = filename.split('.')
        if len(x) == 1:
            raise ValueError("""Could not determine format, please provide
                             a valid format""")
        if x[-1] == 'pkl':
            fmt = 'pickle'
        else:
            raise ValueError("""Could not determine format, please provide
                             a valid format""")

    # Open provided file
    if fmt == "pickle":
        with open(folder + '/' + filename, 'rb') as infile:
            ccp = pickle.load(infile)
    else:
        raise ValueError("Unknown format ", fmt)

    return ccp


class CCPStack(object):
    """Is a CCP Stack matrix. It's size depends on stations that are used as
    input."""

    def __init__(self, latitude, longitude, edist, phase=config.phase,
                 verbose=True):
        """
        Creates an empy Matrix template for a CCP stack

        Parameters
        ----------
        latitude : 1-D np.array
            Latitudes of all seismic stations.
        longitude : 1-D np.array
            Longitudes of all seismic stations.
        edist : float
            Grid density in angular distance.
        phase : str
            Seismic phase either "S" or "P". Phase "S" will lead to more
            grid points being created due to flatter incidence angle. Hence,
            Phase can be "S" for PRFs but not the other way around. However,
            "P" is computationally more efficient.
        verbose : Bool, optional
            If true -> console output. The default is True.

        Returns
        -------
        None.

        """
        # Create bingrid
        self.bingrid = BinGrid(latitude, longitude, edist, phase=phase,
                               verbose=verbose)

        # Compute bins
        self.bingrid.compute_bins()

        # Initialize kdTree for Bins
        self.bingrid.bin_tree()

        # How many RFs are inside of the CCP-Stack?
        self.N = 0

        # Softlink useful variables and functions
        self.coords = self.bingrid.bins
        self.z = np.arange(0, maxz+res, res)
        self.bins = np.zeros([self.coords[0].size, len(self.z)])
        self.illum = np.zeros(np.shape(self.bins))
        # self.latv, self.longv, self.zv = np.meshgrid(self.latitude,
        #                                              self.longitude, self.z)

    def query_bin_tree(self, latitude, longitude, z, data):
        """
        Find closest bins for given latitude and longitude.

        Parameters
        ----------
        latitude : 1D np.array
            Latitudes of piercing points.
        longitude : 1D np.array
            Longitudes of piercing points.
        z : 1D np.array
            depths of piercing points
        data : 1D np.array
            Depth migrated receiver function data

        Returns
        -------
        None.

        """
        eucd, i = self.bingrid.query_bin_tree(latitude, longitude,
                                              self.bingrid.edist/2)

        # Kd tree returns locations that are too far with maxindex+1
        pos = np.where(i < self.bingrid.Nb)

        # Depth index
        j = z[pos]/res
        j = j.astype(int)

        # Lateral position index
        k = i[pos]

        # populate ccp stack and illumination matrix
        self.bins[k, j] = data[j]

        # hit counter + 1
        self.illum[k, j] = self.illum[k, j] + 1

        # Data counter
        self.N = self.N + 1

    def compute_stack(self, network=None, station=None, save=False):
        """
        Computes a ccp stack in self.ccp, using the standard folder
        structure set in config. The stack can be limited to some networks and
        stations. This will take a long while for big data sets!
        Note that the grid should be big enough for the provided networks and
        stations. Best to create the CCPStack object by using ccp.init_ccp().

        Parameters
        ----------
        network : str or list, optional
            Network or networks that are to be included in the ccp stack.
            Standard unix wildcards are allowed. If None, the whole database
            will be used. The default is None.
        station : str or list, optional
            Station or stations that are to be included in the ccp stack.
            Standard unix wildcards are allowed.
            Can only be list if type(network)=str.
            If None, the whole database
            will be used. The default is None.
        save : str or Bool
            Either False if the ccp should not be saved or string with filename
            will be saved in config.ccp. Will be saved as pickle file.

        Returns
        -------
        None (Data are appended to ccp object).

        """
        folder = config.RF[:-1] + self.bingrid.phase

        if network and type(network) == list:
            networks = []
            for net in network:
                networks.extend(fnmatch.filter(os.listdir(folder), net))
        elif network:
            networks = fnmatch.filter(os.listdir(folder), network)
        else:
            networks = os.listdir(folder)
        for net in networks:
            if station and type(station) == list:
                stations = []
                for stat in station:
                    stations.extend(fnmatch.filter(os.listdir(folder+'/'+net),
                                                   stat))
            elif station:
                stations = fnmatch.filter(os.listdir(folder+'/'+net), station)
            else:
                stations = os.listdir(folder+'/'+net)
            for stat in stations:
                # skip info files
                files = fnmatch.filter(os.listdir(folder+'/'+net+'/'+stat),
                                       '*.sac')
                for file in files:
                    rft = read_rf(folder+'/'+net+'/'+stat+'/'+file,
                                  format='SAC')
                    _, rf = rft[0].moveout()
                    lat = np.array(rf.stats.pp_latitude)
                    lon = np.array(rf.stats.pp_longitude)
                    z = np.array(rf.stats.pp_depth)
                    self.query_bin_tree(lat, lon, z, rf.data)
        self.conclude_ccp()

        # save file
        if save:
            self.write(filename=save)

    def conclude_ccp(self):
        """Averages the CCP-bin and populates empty cells with average of
        neighbouring cells."""
        self.ccp = np.divide(self.bins, self.illum+1)

    def write(self, filename='ccp', folder=config.ccp, fmt="pickle"):
        # Remove filetype identifier if provided
        x = filename.split('.')
        if len(x) > 1:
            filename = filename + '.' - x[-1]

        # output location
        oloc = folder + '/' + filename

        if fmt == "pickle":
            with open(oloc + ".pkl", 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("The format type ", fmt, "is unkown.")
