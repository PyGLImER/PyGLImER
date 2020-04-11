#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:10:18 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""

import numpy as np

from .compute.bin import BinGrid
from ..rf.moveout import maxz, res

class CCPStack(object):
    """Is a CCP Stack matrix. It's size depends on stations that are used as
    input."""
    def __init__(self, latitude, longitude, edist, verbose=True):
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
        verbose : Bool, optional
            If true -> console output. The default is True.

        Returns
        -------
        None.

        """
        # Create bingrid
        self.bingrid = BinGrid(latitude, longitude, edist, verbose=verbose)

        # Compute bins
        self.bingrid.compute_bins()

        # Initialize kdTree for Bins
        self.bingrid.bin_tree()

        # Softlink useful variables and functions
        self.coords = self.bingrid.bins
        self.z = np.arange(0, maxz+res, res)
        self.bins = np.zeros([self.coords[0].size, len(self.z)])
        self.illum = np.ones(np.shape(self.bins))
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

    def conclude_ccp(self):
        """Averages the CCP-bin and populates empty cells with average of
        neighbouring cells."""
        self.ccp = np.divide(self.bins, self.illum)

        # Create Hitcounter matrix
        self.hits = self.illum - 1
