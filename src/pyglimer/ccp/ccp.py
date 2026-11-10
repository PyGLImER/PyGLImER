'''
Module for the Common Conversion Stack Computation and handling of the
objects resulting from such.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 10th April 2020 05:30:18 pm
Last Modified: Wednesday, 2nd July 2025 02:20:46 pm
'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import fnmatch
from glob import glob
import os
import pickle
import logging
import time

from copy import deepcopy
from typing import List, Tuple
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import scipy.io as sio
from global_land_mask import globe

from pyglimer.ccp.compute.bin import BinGrid
from pyglimer.ccp.ingest import CCPStackIngestMixin
from pyglimer.ccp.io import _load_ccp_npz, _save_ccp_npz, read_ccp
from pyglimer.ccp.plotting import CCPStackPlotMixin, PhasePick
from pyglimer.ccp.volume import CCPStackVolumeMixin
from pyglimer.database.stations import StationDB
from pyglimer.rf.moveout import res, maxz, maxzm
from pyglimer.utils.geo_utils import epi2euc
from pyglimer.utils.geo_utils import gctrack
from pyglimer.utils.geo_utils import fix_map_extent
from pyglimer.utils.SphericalNN import SphericalNN
from pyglimer.utils.utils import dt_string

__all__ = ['CCPStack', 'PhasePick', 'read_ccp', 'init_ccp']


class CCPStack(CCPStackPlotMixin, CCPStackVolumeMixin, CCPStackIngestMixin):
    """Is a CCP Stack matrix. Its size depends upon stations that are used as
    input."""

    def __init__(
        self, latitude: List[float], longitude: List[float], edist: float,
        phase: str, verbose: bool = True, logdir: str = None,
            _load: str = None):
        """
        Creates an empy object template for a CCP stack.

        :param latitude: Latitudes of all seismic stations.
        :type latitude: 1-D numpy.ndarray
        :param longitude: Longitudes of all seismic stations.
        :type longitude: 1-D numpy.ndarray
        :param edist: Inter bin distance in angular distance.
        :type edist: float
        :param phase: Seismic phase either "S" or "P". Phase "S" will lead to
            more grid points being created due to flatter incidence angle.
            Hence, phase can be "S" for PRFs but not the other way around.
            However, "P" is computationally more efficient.
        :type phase: str
        :param verbose: If true -> console output. The default is True.
        :type verbose: bool, optional
        :param logdir: Directory for log file
        :type logdr: str, optional
        :param _load: Parameter is only used internally when loading from a
            binary numpy file. Then, this passes a filename.
        :type _load: str
        """
        # Loading file from npz
        if _load:
            self = _load_ccp_npz(self, _load)
            return

        # Else just initialise an empty object
        # Loggers for the CCP script
        self.logger = logging.getLogger('pyglimer.ccp.ccp')
        self.logger.setLevel(logging.DEBUG)

        # Create handler to the log
        if not logdir:
            os.makedirs('logs', exist_ok=True)
            fh = logging.FileHandler(os.path.join('logs', 'ccp.log'))
        else:
            fh = logging.FileHandler(os.path.join(logdir, 'ccp.log'))
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        # Create Formatter
        fmt = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)

        # Create bingrid
        self.bingrid = BinGrid(
            latitude, longitude, edist, phase=phase, verbose=verbose)

        # Compute bins
        self.bingrid.compute_bins()

        # Initialize kdTree for Bins
        self.bingrid.bin_tree()

        # How many RFs are inside of the CCP-Stack?
        self.N = 0

        # Softlink useful variables and functions
        self.coords = self.bingrid.bins
        self.z = np.hstack(
            (np.arange(-10, 0, .1), np.arange(0, maxz+res, res)))
        self.bins = np.zeros([self.coords[0].size, len(self.z)])
        self.illum = np.zeros(np.shape(self.bins), dtype=int)
        self.pplat = []
        self.pplon = []
        self.ppz = self.z
        self.binrad = None

    def __str__(self) -> str:
        out = f"Teleseismic Phase: \t\t{self.bingrid.phase}\n" +\
            f"Bin distance: \t\t\t{round(self.bingrid.edist, 3)}\n" +\
            f"Bin radius: \t\t\t{round(self.binrad, 3)}\n" +\
            f"Bounding Box: \tLatitude: \t{round(self.coords[0].min(), 1)}\
                    {round(self.coords[0].max(), 1)}\n" +\
            f"\t\tLongitude: \t{round(self.coords[1].min(), 1)}\
                     {round(self.coords[1].max(),1)}\n" +\
            f"Number of Receiver Functions: \t{self.N}"
        return out

    def query_bin_tree(
        self, latitude: np.ndarray, longitude: np.ndarray,
            n_closest_points: int):
        """
        Find closest bins for provided latitude and longitude.

        :param latitude: Latitudes of piercing points.
        :type latitude: 1-D numpy.ndarray
        :param longitude: Longitudes of piercing points.
        :type longitude: 1-D numpy.ndarray
        :param n_closest_points: Number of closest points to be found.
        :type n_closest_points: int
        :return: bin index k and depth index j
        :rtype: int
        """
        i = self.bingrid.query_bin_tree(
            latitude, longitude, self.binrad_eucl, n_closest_points)

        # Kd tree returns locations that are too far with maxindex+1
        pos = np.where(i < self.bingrid.Nb)

        # Depth index
        j = pos[0]

        # Bin index
        k = i[pos]

        return k, j

    def get_depth_slice(self, z0: float = 410):

        # Area considered around profile points
        area = 2 * self.binrad

        # Resolution
        res = self.binrad/4

        # Depth
        z = self.z.flatten()
        zpos = np.argmin(np.abs(z-z0))
        z0 = z[zpos]

        # Coordinates
        lat = self.coords_new[0].flatten()
        lon = self.coords_new[1].flatten()

        # Map extent with buffer
        extent = fix_map_extent(
            [np.min(lon), np.max(lon), np.min(lat), np.max(lat)],
            fraction=0.05)

        # Create query points
        qlon, qlat = np.meshgrid(
            np.arange(extent[0], extent[1], res),
            np.arange(extent[2], extent[3], res)
        )

        # Create interpolators
        snn = SphericalNN(lat, lon)
        ccp_interpolator = snn.interpolator(
            qlat, qlon, maximum_distance=area, k=10, p=2.0, no_weighting=False)
        ill_interpolator = snn.interpolator(
            qlat, qlon, maximum_distance=area, no_weighting=True)

        # Interpolate
        qccp = ccp_interpolator(self.ccp[:, zpos])
        qill = ill_interpolator(self.hits[:, zpos])

        return qlat, qlon, qill, qccp, extent, z0

    def get_profile(self, slat, slon, ddeg=0.1):

        # Area considered around profile points
        area = 2 * self.binrad

        # Get evenly distributed points
        qlat, qlon, qdists, sdists = gctrack(slat, slon, dist=ddeg)

        # Get interpolation weights and rows.
        # Create SphericalNN kdtree
        snn = SphericalNN(self.coords_new[0], self.coords_new[1])
        ccp_interpolator = snn.interpolator(
            qlat, qlon, maximum_distance=area, k=10, p=2.0, no_weighting=False)
        ill_interpolator = snn.interpolator(
            qlat, qlon, maximum_distance=area, no_weighting=True)

        # Get coordinates array from CCPStack
        qz = deepcopy(self.z)

        # Get data arrays
        Np, Nz = self.ccp.shape
        Nq = len(qlat)

        # Interpolate
        qccp = np.zeros((Nq, Nz))
        qill = np.zeros((Nq, Nz))

        # Interpolate each depth
        for _i, (_ccpcol, _illumcol) in enumerate(
                zip(self.ccp.T, self.hits.T)):
            qccp[:, _i] = ccp_interpolator(_ccpcol)
            qill[:, _i] = ill_interpolator(_illumcol)

        # Interpolate onto regular depth grid for easy representation with
        # imshow
        ccp2D_interpolator = RegularGridInterpolator(
            (qdists, qz), np.where(np.isnan(qccp), 0, qccp))
        ill2D_interpolator = RegularGridInterpolator(
            (qdists, qz), qill)

        # Where to sample
        qqz = np.arange(np.min(qz), np.max(qz), 1)
        xqdists, xqz = np.meshgrid(qdists, qqz)

        # Interpolate
        qccp = ccp2D_interpolator((xqdists, xqz))
        qill = ill2D_interpolator((xqdists, xqz))

        return slat, slon, sdists, qlat, qlon, qdists, qz, qill, qccp, area

    def compute_stack(
        self, vel_model: str, rfloc: str,
        in_format: str = 'hdf5', preproloc: str = None,
        network: str or list or None = None,
        station: str or list or None = None, geocoords: tuple or None = None,
        pattern: list or None = None, save: str or bool = False,
        filt: tuple or None = None,
        binrad: float = 1/(2*np.cos(np.radians(30))), append_pp: bool = False,
            multiple: bool = False, mc_backend: str = 'joblib',
            n_jobs: int = -1):
        """
        Computes a ccp stack in self.ccp, using the data from rfloc.
        The stack can be limited to some networks and stations.

        :param vel_model: Velocity model located in data. Either `iasp91.dat`
            for 1D raytracing or `3D` for 3D raytracing with GYPSuM model.
            Using 3D Raytracing will cause the code to take about 30% longer.
        :type vel_model: str
        :param rfloc: Parental folder in which the receiver functions in
            time domain are. The default is 'output/waveforms/RF.
        :type rfloc: str, optional
        :param in_format: **INPUT FORMAT** of receiver functions. Either
            `'h5'` or `'hdf5'`. Defaults to 'hdf5'.
        :type in_format: str, optional
        :param preproloc: Parental folder containing the preprocessed mseeds.
            Only needed if option geocoords is used. The default is
            None. **Should be None if hdf5 is used!**
        :type preproloc: str, optional
        :param network: This parameter is ignored if the pattern is given.
            Network or networks that are to be included in the ccp stack.
            Standard unix wildcards are allowed. If None, the whole database
            will be used. The default is None.
        :type network: str or list, optional
        :param station: This parameter is ignored if the pattern is given.
            Station or stations that are to be included in the ccp stack.
            Standard unix wildcards are allowed.
            Can only be list if type(network)=str.
            If None, the whole database
            will be used. The default is None.
        :type station: str or list, optional
        :param geocoords: Include all stations in the rectangle given by
            (minlat, maxlat, minlon, maxlon). Will be ignored if pattern or
            network is given, by default None.
        :type geocoords: Tuple, optional
        :param pattern: Search pattern for files. Overwrites network,
            station, and geocooords options. Usually only used by
            :func:`~pyglimer.ccp.ccp.init_ccp()`, defaults to None.
        :type pattern: list, optional
        :param save: Either False if the ccp should not be saved or string with
            filename will be saved in config.ccp. Will be saved as pickle file.
        :type save: str or bool, optional
        :param filt: Decides whether to filter the receiver function prior to
            depth migration. Either a Tuple of form `(lowpassf, highpassf)` or
            `None` / `False`.
        :type filt: tuple, optional
        :param binrad: Defines the bin radius with
            bin radius = binrad*distance_bins.
            Full Overlap = 1/(2*cosd(30)), Full coverage: 1.
            The default is full overlap.
        :type binrad: float, optional
        :param append_pp: appends piercing point coordinates if True, so they
            can later be plotted. Not recommended for big data sets.
            The default is false. **Deprecated for multi-core**
        :type append_pp: bool, optional
        :param multiple: Append receiver functions for first order multiples.
            It can be decided later, whether they should be used in the final
            ccp stack. Will result in a longer computation time.
        :type multiple: bool, optional
        :param mc_backend: Multi-core backend to use for the computations.
            Can be either `"joblib"` (multiprocessing) or `"MPI"`. Not that
            MPI compatibility is only implemented with hdf5 files.
        :param n_jobs: Number of parallel jobs for the joblib backend.
            -1 uses all cores, defaults to -1. Ignored by the MPI backend.
        :type n_jobs: int, optional
        :raises ValueError: For wrong inputs


        .. warning::

            This will take a long while for big data sets!

        .. note::

            When using, `mc_backend='MPI'`. This has to be started from a
            terminal via mpirun.

        .. note::

            MPI is only compatible with receiver functions saved in hdf5
            format.

        .. note::

            Note that the grid should be big enough for the provided networks
            and stations. Best to create the CCPStack object by using
            :func:`~pyglimer.cpp.ccp.init_ccp()`.
        """
        if in_format.lower() == 'hdf5':
            in_format = 'h5'
        elif in_format.lower() == 'sac':
            raise NotImplementedError(
                'Stacking from SAC files is no longer supported. Convert '
                'your receiver functions to the hdf5 database format '
                '(in_format="h5") instead.')

        if binrad < 1/(2*np.cos(np.radians(30))):
            raise ValueError(
                """Minimum allowed binradius is bin distance/(2*cos(30deg)).
                Else the grid will not cover all the surface area."""
            )

        self.binrad = binrad*self.bingrid.edist

        # How many closest points are queried by the bintree?
        # See Gauss circle problem
        # sum of squares for 2 squares for max binrad=4
        # Using the ceiling function to account for inaccuracies
        sosq = [1, 4, 4, 0, 4, 8, 0, 0, 4, 4, 8, 0, 0, 8, 0, 0, 4]
        if int(np.ceil(binrad**2+1)) > len(sosq):
            raise ValueError(
                """Maximum allowed binradius is 4 times the bin distance""")
        n_closest_points = sum(sosq[0:int(np.ceil(binrad**2+1))])

        # Compute maxdist in euclidean space
        self.binrad_eucl = epi2euc(self.binrad)

        folder = os.path.join(rfloc, self.bingrid.phase)

        start = time.time()

        try:
            self.logger.info('Stacking started')
        except AttributeError:
            # Loggers for the CCP script
            self.logger = logging.getLogger('pyglimer.ccp.ccp')
            self.logger.setLevel(logging.INFO)

            # Create handler to the log
            fh = logging.FileHandler('logs/ccp.log')
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
            self.logger.info('Stacking started')

        if multiple:
            # Use multiples?
            endi = np.where(self.z == maxzm)[0][0] + 1
            self.bins_m1 = np.zeros(self.bins[:, :endi].shape)
            self.bins_m2 = np.zeros(self.bins[:, :endi].shape)
            self.illumm = np.zeros(self.bins[:, :endi].shape, dtype=int)
        else:
            endi = None

        if geocoords and not pattern:
            # Stations by coordinates
            # create empty lists for station latitude and longitude
            lat = (geocoords[0], geocoords[1])
            lon = (geocoords[2], geocoords[3])
            db = StationDB(
                preproloc or folder, phase=self.bingrid.phase, use_old=False)
            net, stat = db.find_stations(lat, lon, phase=self.bingrid.phase)
            pattern = ["{}.{}".format(a_, b_) for a_, b_ in zip(net, stat)]
            # Clear memory
            del db, net, stat

        if not pattern:
            pattern = []  # List of input constraints
            if not (network and station):
                pattern.append('*.%s' % in_format)
        else:
            pattern = ["*{}*%s".format(_a) % in_format for _a in pattern]

        files = []  # List of files filtered for input criteria

        # List of all input files in the desired format
        infiles = glob(os.path.join(
            folder, '**', '*.%s' % in_format), recursive=True)

        # Set filter patterns
        if network:
            if isinstance(network, list):
                if station:
                    if isinstance(station, list):
                        for net in network:
                            for stat in station:
                                pattern.append('*%s.%s*.%s' % (
                                    net, stat, in_format))
                    else:
                        raise ValueError(
                            "Combination of network and station is invalid")
                else:
                    for net in network:
                        pattern.append('*%s.*.%s' % (net, in_format))
            elif isinstance(network, str):
                if station:
                    if isinstance(station, str):
                        pattern.append('*%s.%s*.%s' % (
                            network, station, in_format))
                    elif isinstance(station, list):
                        for stat in station:
                            pattern.append('*%s.%s*.%s' % (
                                network, stat, in_format))
                else:
                    pattern.append('*%s.*.%s' % (network, in_format))
        elif station:
            raise ValueError(
                "You have to provide both network and station \
code if you want to filter by station")

        # Do filtering
        for pat in pattern:
            files.extend(fnmatch.filter(infiles, pat))

        # clear memory
        del pattern, infiles

        # Actual CCP stack #
        # The test data needs to be filtered
        if network == 'matlab':
            filt = [.03, 1.5]  # bandpass frequencies

        self.logger.debug('Using data from the following files: %s' % (
            str(files)))

        # Define grid boundaries for 3D RT
        latb = (self.coords[0].min(), self.coords[0].max())
        lonb = (self.coords[1].min(), self.coords[1].max())

        if in_format.lower() == 'h5':
            self._create_ccp_from_hdf5_mc(
                files, multiple, append_pp, n_closest_points, vel_model,
                latb, lonb, filt, endi, mc_backend, n_jobs)

        else:
            raise NotImplementedError(
                'Unknown input format %s.' % in_format
            )

        end = time.time()
        self.logger.info("Stacking finished.")
        self.logger.info(dt_string(end-start))
        self.conclude_ccp()

        # save file
        if save:
            self.write(filename=save)


    def conclude_ccp(
        self, keep_empty: bool = False, keep_water: bool = False, r: int = 0,
            multiple: bool = False, z_multiple: int = 200):
        """
        Averages the CCP-bin and populates empty cells with average of
        neighbouring cells. No matter which option is
        chosen for the parameters, data is never lost entirely. One can always
        execute a new conclude_ccp() command. However, decisions that are
        taken here will affect the .mat output and plotting outputs.

        :param keep_empty: Keep entirely empty bins. The default is False.
        :type keep_empty: bool, optional
        :param keep_water: For False all bins that are on water-covered areas
            will be discarded, defaults to False
        :type keep_water: bool, optional
        :param r: Fields with less than r hits will be set equal 0.
            r has to be >= 1, defaults to 3.
        :type r: int, optional
        :param multiple: Use multiples in stack. Either False or weigthing;
            i.e. 'linear' for linearly weighted stack between the three phases,
            'zk' for a Zhu & Kanamori approach, or 'pws' for a phase weighted
            stack. Use 'm1' to use only first multiple mode (no stack),
            'm2' for
            RFs created only with 2nd multiple phase (PSS), and m for an
            equal-weight stack of m1 and m2. By default False.
        :type multiple: bool or str
        :param z_multiple: Until which depth [km] should multiples be
            considered,
            maximal value is 200 [km]. Will only be used if multiple=True.
            By default 200 km.
        :type z_multiple: int, optional
        """
        if z_multiple > 200:
            raise ValueError('Maximal depth for multiples is 200 km.')
        endi = np.where(self.z == z_multiple)[0][0] + 1
        if multiple == 'linear':

            self.ccp = np.hstack(((
                np.divide(self.bins[:, :endi], self.illum[:, :endi]+1)
                + np.divide(self.bins_m1[:, :endi], self.illumm[:, :endi]+1)
                + np.divide(
                    self.bins_m2[:, :endi], self.illumm[:, :endi]+1))/3,
                np.zeros(self.bins[:, endi:].shape)))
        elif multiple == 'zk':
            self.ccp = np.hstack((
                .7*np.divide(self.bins[:, :endi], self.illum[:, :endi]+1)
                + .2*np.divide(self.bins_m1[:, :endi], self.illumm[:, :endi]+1)
                + .1*np.divide(
                    self.bins_m2[:, :endi], self.illumm[:, :endi]+1),
                np.zeros(self.bins[:, endi:].shape)))
        elif multiple == 'm1':
            self.ccp = np.hstack((
                np.divide(self.bins_m1[:, :endi], self.illumm[:, :endi]+1),
                np.zeros(self.bins[:, endi:].shape)))
        elif multiple == 'm2':
            self.ccp = np.hstack((
                np.divide(self.bins_m2[:, :endi], self.illumm[:, :endi]+1),
                np.zeros(self.bins[:, endi:].shape)))
        elif multiple == 'm':
            self.ccp = np.hstack((
                np.divide(self.bins_m2[:, :endi]+self.bins_m1[:, :endi],
                          2*(self.illumm[:, :endi])+1),
                np.zeros(self.bins[:, endi:].shape)))
        elif not multiple:
            self.ccp = np.divide(self.bins, self.illum+1)
        else:
            raise ValueError(
                'The requested multiple stacking mode is \
misspelled or not yet implemented')

        self.hits = self.illum.copy()

        if r > 1:
            index = np.where(np.logical_and(self.illum > 1, self.illum < r))
            self.ccp[index] = 0
            self.hits[index] = 0

        if not keep_empty:
            i = np.where(self.hits.sum(axis=1) == 0)[0]
            self.ccp = np.delete(self.ccp, i, 0)
            self.hits = np.delete(self.hits, i, 0)
            self.coords_new = (np.delete(self.coords[0], i, 1),
                               np.delete(self.coords[1], i, 1))

        # Check if bins are on land
        if not keep_water:
            if keep_empty:
                self.coords_new = self.coords.copy()

            lats, lons = self.coords_new
            # list of indices that contain water
            index = globe.is_ocean(lats, lons)[0, :]

            self.ccp = np.delete(self.ccp, index, 0)
            self.hits = np.delete(self.hits, index, 0)
            self.coords_new = (np.delete(self.coords_new[0], index, 1),
                               np.delete(self.coords_new[1], index, 1))

        # Else everything will always pick coords_new instead of coords
        if keep_water and keep_empty:
            try:
                del self.coords_new
            except AttributeError:
                # coords_new does not exist (e.g. keep_empty was True but
                # the not-keep_water branch never created it)
                pass

    def write(self, filename=None, folder='.', fmt="pickle"):
        """
        Saves the CCPStream file as pickle or matlab file. Only save
        as Matlab file for exporting, as not all information can be
        preserved!

        :param filename: Name as which to save, file extensions aren't
            necessary.
        :type filename: str, optional
        :param folder: Output folder, defaults to 'output/ccps'
        :type folder: str, optional
        :param fmt: Either `"pickle"`, `'npz'` or `"matlab"` for .mat,
            defaults to "pickle".
        :type fmt: str, optional
        :raises ValueError: For unknown formats.

        .. note::

            Saving in numpy (npz) format is the most robust option. However,
            loading those files will take the longest time as some attributes
            have to be recomputed. **NUMPY FORMAT IS RECOMMENDED FOR
            ARCHIVES!**

        .. warning::

            Pickling files is the fastest option, but might lead to
            incompatibilities between different PyGLImER versions.
            **USE PICKLE IF YOU WOULD LIKE TO SAVE FILES FOR SHORTER AMOUNTS
            OF TIME.**

        .. warning::

            Saving as Matlab files is only meant if one wishes to plot with
            the old Matlab Receiver function exploration toolset (not
            recommended anymore).
        """
        # delete logger (cannot be pickled)
        try:
            del self.logger
        except AttributeError:
            # For backwards compatibility
            pass

        # Standard filename
        if not filename:
            filename = '%s_%s_%s' % (
                self.bingrid.phase, str(self.bingrid.edist),  str(self.binrad))

        # Remove filetype identifier if provided
        x = filename.split('.')
        if len(x) > 1:
            if x[-1].lower() in ('pkl', 'mat', 'npz'):
                fmt = x[-1].lower()
                filename = '.'.join(x[:-1])

        # output location
        oloc = os.path.join(folder, filename)

        if fmt in ("pickle", "pkl"):
            with open(oloc + ".pkl", 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        # Save as Matlab file (exporting to plot)
        elif fmt in ("matlab", 'mat'):

            # Change vectors so it can be corrected for elevation
            # Only for old data standard
            if min(self.z) == 0:
                illum = np.pad(self.hits, ((0, 0), (round(10/.1), 0)))
                depth = np.hstack((np.arange(-10, 0, .1), self.z))
                ccp = np.pad(self.ccp, ((0, 0), (round(10/.1), 0)))
            else:
                illum = self.hits
                depth = self.z
                ccp = self.ccp

            if hasattr(self, 'coords_new') and self.coords_new[0].size:
                lat_ccp, lon_ccp = self.coords_new
            else:
                lat_ccp, lon_ccp = self.coords

            d = {}
            d.update({'RF_ccp': ccp,
                      'illum': illum,
                      'depth_ccp': depth,
                      'lat_ccp': lat_ccp,
                      'lon_ccp': lon_ccp,
                      'CSLAT': self.bingrid.latitude,
                      'CSLON': self.bingrid.longitude,
                      'clat': np.array(self.pplat),
                      'clon': np.array(self.pplon),
                      'cdepth': self.ppz.astype(float)})

            sio.savemat(oloc + '.mat', d)

        elif fmt.lower() in ('numpy', 'np', 'npz'):
            _save_ccp_npz(self, oloc)

        # Unknown formats
        else:
            raise ValueError("The format type ", fmt, "is unkown.")

def init_ccp(
    rfloc: str, spacing: float, vel_model: str, phase: str,
    preproloc: str = None, network: str or List[str] = None,
    station: str or List[str] = None,
    geocoords: Tuple[float, float, float, float] = None,
    compute_stack: bool = False, filt: Tuple[float, float] = None,
    binrad: float = np.cos(np.radians(30)), append_pp: bool = False,
    multiple: bool = False, save: str or bool = False, verbose: bool = True,
    format: str = 'hdf5', mc_backend: str = 'joblib'
) -> CCPStack:
    """
    Computes a ccp stack in self.ccp using data from statloc and rfloc.
    The stack can be limited to some networks and
    stations.

    :param rfloc: Parental directory in that the RFs are saved.
    :type rfloc: str
    :param spacing: Angular distance between each bingrid point.
    :type spacing: float
    :param vel_model: Velocity model located in data. Either iasp91 (1D
        raytracing) or 3D for 3D raytracing using a model compiled from GyPSuM.
    :type vel_model: str
    :param phase: The teleseismic phase to be used.
    :type phase: str
    :param preproloc: Parental directory in that the preprocessed miniseeds are
        saved, defaults to None. **Should be None when working with h5 files!**
    :type preproloc: str, optional
    :param network: Network or networks that are to be included in the ccp
        stack.
        Standard unix wildcards are allowed. If None, the whole database
        will be used. The default is None., defaults to None
    :type network: str or list, optional
    :param station: Station or stations that are to be included in the ccp
        stack. Standard unix wildcards are allowed. Can only be list if
        type(network)==str. If None, the whole database will be used.
        The default is None.
    :type station: str or list, optional
    :param geocoords: An alternative way of selecting networks and stations.
        Given in the form (minlat, maxlat, minlon, maxlon), defaults to None
    :type geocoords: Tuple, optional
    :param compute_stack: If true it will compute the stack by calling
        :meth:`~pyglimer.ccp.ccp.CCPStack.compute_stack()`.
        That can take a long time! The default is False.
    :type compute_stack: bool, optional
    :param filt: Decides whether to filter the receiver function prior to
            depth migration. Either a Tuple of form `(lowpassf, highpassf)` or
            `None` / `False`.
    :type filt: Tuple[float, float], optional
    :param binrad: *Only used if compute_stack=True*
            Defines the bin radius with bin radius = binrad*distance_bins.
            Full Overlap = cosd(30), Full coverage: 1.
            The default is full overlap.
    :type binrad: float, optional
    :param append_pp: *Only used if compute_stack=True.*
        Appends piercing point locations to object. Can be used to plot
        pps on map. Not recommended for large datasets as it takes A LOT
        longer and makes the file a lot larger.
        The default is False., defaults to False
    :type append_pp: bool, optional
    :param multiple: Should the CCP Stack be prepared to work with multiples?
        It can be chosen later, whether the RFs from multiples are to be
        incorporated into the CCP Stack. By default False.
    :type multiple: bool, optional
    :param save: Either False if the ccp should not be saved or string with
        filename will be saved. Will be saved as pickle file.
        The default is False.
    :type save: bool or str, optional
    :param verbose: Display info in terminal., defaults to True
    :type verbose: bool, optional
    :param format: The **INPUT FORMAT** from which receiver functions are read.
        If you saved your receiver function as hdf5 files, set ==hdf5.
        Defaults to hdf5.
    :type format: str
    :param mc_backend: Multi-core backend to use for the computations.
        Can be either `"joblib"` (multiprocessing) or `"MPI"`. Not that MPI
        compatibility is only implemented with hdf5 files.
    :raises TypeError: For wrong inputs.
    :return: CCPStack object.
    :rtype: :class:`~pyglimer.ccp.ccp.CCPstack`

    .. note::

        When using, `mc_backend='MPI'`. This has to be started from a terminal
        via mpirun.

    .. note::

        MPI is only compatible with receiver functions saved in hdf5 format.
    """
    if phase[-1].upper() == 'S' and multiple:
        raise NotImplementedError(
            'Multiple mode is not supported for phase S.')

    if format.lower() in ('hdf5', 'h5', 'hdf'):
        preproloc = None
        format = 'hdf5'
    db = StationDB(
        preproloc or rfloc, phase=phase, use_old=False, hdf5=format == 'hdf5')

    # Were network and stations provided?
    # Possibility 1 as geo boundaries
    if geocoords:
        lat = (geocoords[0], geocoords[1])
        lon = (geocoords[2], geocoords[3])
        subset = db.geo_boundary(lat, lon, phase=phase)
    else:
        subset = db.db

    net = list(subset['network'])
    stat = list(subset['station'])
    codes = list(subset['code'])
    lats = list(subset['lat'])
    lons = list(subset['lon'])

    # Filter
    if isinstance(network, str) and not isinstance(station, list):
        codef = fnmatch.filter(codes, '%s.%s' % (network, station or '*'))
    elif isinstance(network, str) and isinstance(station, list):
        codef = list(set(codes).intersection([
            '%s.%s' % (network, stat) for stat in station]))
    elif isinstance(network, list):
        if station is not None and not isinstance(station, list):
            raise TypeError(
                """Provide a list of stations, when using a list of
                networks as parameter.""")
        elif not station:
            codef = list(set(codes).intersection([
                '%s.*' % net for net in network]))
        elif isinstance(station, list):
            if len(station) != len(network):
                raise ValueError(
                    'Length of network and station list have to be equal.')
            codef = list(set(codes).intersection([
                '%s.%s' % (net, stat) for net, stat in zip(network, station)]))

    if (network and station) is None:
        codef = codes

    else:
        # Find indices (first occurrence, like list.index, but O(1) lookups
        # instead of a linear scan per requested station)
        index_of = {}
        for i, c in enumerate(codes):
            if c not in index_of:
                index_of[c] = i
        ind = [index_of[cf] for cf in codef]
        net = np.array(net)[ind]
        stat = np.array(stat)[ind]
        lats = np.array(lats)[ind]
        lons = np.array(lons)[ind]

    logdir = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.abspath(rfloc))), 'logs')
    ccp = CCPStack(
        lats, lons, spacing, phase=phase, verbose=verbose, logdir=logdir)

    # Clear Memory
    del stat, lats, lons

    if not geocoords:
        pattern = None

    if compute_stack:
        ccp.compute_stack(
            vel_model=vel_model, network=network, station=station, save=save,
            filt=filt, multiple=multiple,
            pattern=pattern, append_pp=append_pp, binrad=binrad, rfloc=rfloc,
            in_format=format, mc_backend=mc_backend)
    return ccp
