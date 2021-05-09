'''
Module for the Common Conversion Stack Computation and handling of the
objects resulting from such.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 10th April 2020 05:30:18 pm
Last Modified: Wednesday, 28th April 2021 05:04:08 pm
'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import fnmatch
import os
import pickle
import logging
# import shutil
import time

from copy import deepcopy
from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator


from obspy import read_inventory
import scipy.io as sio
# from mpl_toolkits.basemap import Basemap
from global_land_mask import globe
# from pathlib import Path
from psutil import virtual_memory
# import subprocess
from tqdm import tqdm
import pyvista as pv
import vtk

import pyglimer
from pyglimer.ccp.compute.bin import BinGrid
from pyglimer.ccp.plot_utils.plot_bins import plot_bins
from pyglimer.ccp.plot_utils.plot_cross_section import plot_cross_section
from pyglimer.constants import R_EARTH, DEG2KM, KM2DEG
from pyglimer.database.stations import StationDB
from pyglimer.plot.plot_map import plot_map_ccp, plot_vel_grad
from pyglimer.plot.plot_volume import VolumePlot, VolumeExploration
from pyglimer.rf.create import read_rf
from pyglimer.rf.moveout import res, maxz, maxzm
from pyglimer.utils.createvmodel import ComplexModel
from pyglimer.utils.geo_utils import epi2euc, geo2cart
from pyglimer.utils.geo_utils import gctrack
from pyglimer.utils.geo_utils import fix_map_extent
from pyglimer.utils.SphericalNN import SphericalNN
from pyglimer.utils.utils import dt_string, chunks


class PhasePick(object):
    """
    A phasepick object, just created for more convenient plotting.
    """

    def __init__(
        self, coords: np.ndarray, amplitudes: np.ndarray, polarity: str,
            z: np.ndarray, depthrange: list = [None, None]):
        """
        Initialise object

        Parameters
        ----------
        coords : np.ndarray
            2-d array containing latitudes and longitudes.
        amplitudes : np.ndarray
            1-D array containg amplitude values of each bin
        polarity : str
            Either '+' for positive velocity gradients or '-' for negative
        z : np.ndarray
            1D array containing depth of maxima/minima per bin
        depthrange : list, optional
            list containing depth restrictions in form of [zmin, zmax],
            by default [None, None]
        """
        # assign vars
        self.coords = coords
        self.a = amplitudes
        self.z = z
        self.pol = polarity
        self.depthr = depthrange

    def plot(
        self, plot_amplitude: bool = False, outputfile: str or None = None,
        format: str = 'pdf', dpi: int = 300, cmap: str = 'gist_rainbow',
            geology: bool = False, title: str or None = None):
        """
        Plot heatmap containing depth or amplitude of picked phase.

        Parameters
        ----------
        plot_amplitude : bool, optional
            If True amplitude instead of depths is plotted, by default False
        outputfile : str or None, optional
            Write Figure to file, by default None
        format : str, optional
            File format, by default 'pdf'
        dpi : int, optional
            Resolution for non-vector graphics, by default 300
        cmap : str, optional
            Colormap
        geology : bool, optional
            Plot geological map.
        title
        """
        lat = (
            np.floor(min(self.coords[0][0]))-1,
            np.ceil(max(self.coords[0][0])+1))
        lon = (np.floor(min(self.coords[1][0]))-1,
               np.ceil(max(self.coords[1][0])+1))

        plot_vel_grad(
            self.coords, self.a, self.z, plot_amplitude, lat, lon, outputfile,
            dpi=dpi, format=format, cmap=cmap, geology=geology, title=title)


class CCPStack(object):
    """Is a CCP Stack matrix. Its size depends upon stations that are used as
    input."""

    def __init__(
        self, latitude: float, longitude: float, edist: float, phase: str,
            verbose: bool = True, logdir: str or None = None):
        """
        Creates an empy object template for a CCP stack

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
        """

        # Loggers for the CCP script
        self.logger = logging.Logger('pyglimer.ccp.ccp')
        self.logger.setLevel(logging.INFO)

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
        # self.z = np.arange(0, maxz+res, res)
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
        self, latitude: np.ndarray, longitude: np.ndarray, data: np.ndarray,
            n_closest_points: int):
        """
        Find closest bins for provided latitude and longitude.

        :param latitude: Latitudes of piercing points.
        :type latitude: 1-D numpy.ndarray
        :param longitude: Longitudes of piercing points.
        :type longitude: 1-D numpy.ndarray
        :param data: Depth migrated receiver function data
        :type data: 1-D numpy.ndarray
        :param n_closest_points: Number of closest points to be found.
        :type n_closest_points: int
        :return: bin index k and depth index j
        :rtype: int
        """
        i = self.bingrid.query_bin_tree(latitude, longitude, self.binrad_eucl,
                                        n_closest_points)

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

    def get_profile(self, slat, slon):

        # Area considered around profile points
        area = 2 * self.binrad

        # Get evenly distributed points
        qlat, qlon, qdists, sdists = gctrack(slat, slon, self.bingrid.edist/4)

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
        for _i, (_ccpcol, _illumcol) in enumerate(zip(self.ccp.T, self.hits.T)):
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

    def plot_cross_section(self, *args, **kwargs):
        return plot_cross_section(self, *args, **kwargs)

    def compute_stack(
        self, vel_model: str, rfloc: str = 'output/waveforms/RF',
        statloc: str = 'output/stations',
        preproloc: str = 'output/waveforms/preprocessed',
        network: str or list or None = None,
        station: str or list or None = None, geocoords: tuple or None = None,
        pattern: list or None = None, save: str or bool = False,
        filt: tuple or None = None,
        binrad: float = 1/(2*np.cos(np.radians(30))), append_pp: bool = False,
            multiple: bool = False):
        """
        Computes a ccp stack in self.ccp, using the data from rfloc.
        The stack can be limited to some networks and
        stations. This will take a long while for big data sets!
        Note that the grid should be big enough for the provided networks and
        stations. Best to create the CCPStack object by using ccp.init_ccp().

        :param vel_model: Velocity model located in data. Either `iasp91.dat`
            for 1D raytracing or `3D` for 3D raytracing with GYPSuM model.
            Using 3D Raytracing will cause the code to take about 30% longer.
        :type vel_model: str
        :param rfloc: Parental folder in which the receiver functions in
            time domain are. The default is 'output/waveforms/RF.
        :type rfloc: str, optional
        :param statloc: Folder containing the response information. Only needed
             if option geocoords is used. The default is 'output/stations'.
        :type statloc: str, optional
        :param preproloc: Parental folder containing the preprocessed mseeds.
            Only needed if option geocoords is used. The default is
            'output/waveforms/preprocessed'.
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
        :param pattern: Search pattern for .sac files. Overwrites network,
            station, and geocooords options. Usually only used by
            :func:`~pyglimer.ccp.ccp.init_ccp()`, defaults to None.
        :type pattern: list, optional
        :param save: Either False if the ccp should not be saved or string with
            filename will be saved in config.ccp. Will be saved as pickle file.
        :param filt: Decides whether to filter the receiver function prior to
            depth migration. Either a Tuple of form `(lowpassf, highpassf)` or
            `None` / `False`.
        :type filt: tuple, optional
        :type save: str or bool, optional
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
        :raises ValueError: For wrong inputs
        """

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
        try:
            n_closest_points = sum(sosq[0:int(np.ceil(binrad**2+1))])
        except IndexError:
            raise ValueError(
                """Maximum allowed binradius is 4 times the bin distance""")

        # Compute maxdist in euclidean space
        self.binrad_eucl = epi2euc(self.binrad)

        folder = os.path.join(rfloc, self.bingrid.phase)

        start = time.time()

        try:
            self.logger.info('Stacking started')
        except AttributeError:
            # Loggers for the CCP script
            self.logger = logging.Logger('pyglimer.ccp.ccp')
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

        if network and type(network) == str and not pattern:
            # Loop over fewer files
            folder = os.path.join(folder, network)
            if station and type(station) == str:
                folder = os.path.join(folder, station)

        elif geocoords and not pattern:
            # Stations by coordinates
            # create empty lists for station latitude and longitude
            lat = (geocoords[0], geocoords[1])
            lon = (geocoords[2], geocoords[3])
            db = StationDB(preproloc, phase=self.bingrid.phase, use_old=False)
            net, stat = db.find_stations(lat, lon, phase=self.bingrid.phase)
            pattern = ["{}.{}".format(a_, b_) for a_, b_ in zip(net, stat)]
            # Clear memory
            del db, net, stat

        if not pattern:
            pattern = []  # List of input constraints
            if not network and not station:  # global
                pattern.append('*.sac')
        else:
            pattern = ["*{}.*.sac".format(_a) for _a in pattern]

        streams = []  # List of files filtered for input criteria
        infiles = []  # List of all files in folder

        for root, _, files in os.walk(folder):
            for name in files:
                infiles.append(os.path.join(root, name))

        # Special rule for files imported from Matlab
        if network == 'matlab' or network == 'raysum':
            pattern.append('*.sac')

        # Set filter patterns
        elif network:
            if type(network) == list:
                if station:
                    if type(station) == list:
                        for net in network:
                            for stat in station:
                                pattern.append('*%s.%s.*.sac' % (net, stat))
                    else:
                        raise ValueError("""The combination of network
                                         and station are invalid""")
                else:
                    for net in network:
                        pattern.append('*%s.*.sac' % (net))
            elif type(network) == str:
                if station:
                    if type(station) == str:
                        pattern.append('*%s.%s.*.sac' % (network, station))
                    elif type(station) == list:
                        for stat in station:
                            pattern.append('*%s.%s.*.sac' % (net, stat))
                else:
                    pattern.append('*%s.*.sac' % (network))
        elif station:
            raise ValueError("""You have to provide both network and station
                             code if you want to filter by station""")

        # Do filtering
        for pat in pattern:
            streams.extend(fnmatch.filter(infiles, pat))

        # clear memory
        del pattern, infiles

        # Data counter
        self.N = self.N + len(streams)

        self.logger.info('Number of receiver functions used: '+str(self.N))
        print('Number of receiver functions used: '+str(self.N))

        # Split job into n chunks
        num_cores = cpu_count()

        self.logger.info('Number of cores used: '+str(num_cores))
        print('Number of cores used: '+str(num_cores))

        mem = virtual_memory()

        self.logger.info('Available system memory: '+str(mem.total*1e-6)+'MB')
        print('Available system memory: '+str(mem.total*1e-6)+'MB')

        # Check maximum information that can be saved
        # using half of the RAM.
        # approximately 8 byte per element in RF + 8 byte for idx
        mem_needed = 3*8*850*len(streams)*100
        # For stacking with multiple modes, we'll need more memory
        if multiple:
            mem_needed = mem_needed*1.75

        # Split into several jobs if too much data
        if mem_needed > mem.total:
            N_splits = int(np.ceil(mem_needed/mem.total))
            split_size = int(np.ceil(len(streams)/N_splits))
            print('Splitting RFs into '+str(N_splits)+' chunks \
due to insufficient memory. Each progressbar will \
only show the progress per chunk.')
        else:
            split_size = len(streams)

        # Actual CCP stack
        # Note loki does mess up the output and threads is slower than
        # using a single core

        # The test data needs to be filtered
        if network == 'matlab':
            filt = [.03, 1.5]  # bandpass frequencies

        # Define grid boundaries for 3D RT
        latb = (self.coords[0].min(), self.coords[0].max())
        lonb = (self.coords[1].min(), self.coords[1].max())

        # How many tasks should the main process be split in?
        # If the number is too high, it will need unnecessarily
        # much disk space or caching (probably also slower).
        # If it's too low, the progressbar won't work anymore.
        # !Memmap arrays are getting extremely big, size in byte
        # is given by: nrows*ncolumns*nbands
        # with 64 cores and 10 bands/core that results in about
        # 90 GB for each of the arrays for a finely gridded
        # CCP stack of North-America

        for stream_chunk in chunks(streams, split_size):
            num_split_max = num_cores*100  # maximal no of jobs
            len_split = int(np.ceil(len(stream_chunk)/num_cores))
            if len_split > 10:
                if int(np.ceil(len(stream_chunk)/len_split)) > num_split_max:
                    len_split = int(np.ceil(len_split/num_split_max))
                else:
                    len_split = int(np.ceil(len_split/(len_split/10)))
            num_split = int(np.ceil(len(stream_chunk)/len_split))

            out = Parallel(n_jobs=num_cores)(  # prefer='processes'
                delayed(self.multicore_stack)(
                    st, append_pp, n_closest_points, vel_model,
                    latb, lonb, filt, i, multiple)
                for i, st in zip(
                    tqdm(range(num_split)),
                    chunks(stream_chunk, len_split)))

            # Awful way to solve it, but the best I could find
            if multiple:
                for kk, jj, datal, datalm1, datalm2 in out:
                    for k, j, data, datam1, datam2 in zip(
                            kk, jj, datal, datalm1, datalm2):
                        self.bins[k, j] = self.bins[k, j] + data[j]

                        # hit counter + 1
                        self.illum[k, j] = self.illum[k, j] + 1

                        # multiples
                        iii = np.where(j <= endi-1)[0]
                        jm = j[iii]
                        km = k[iii]
                        try:
                            self.bins_m1[
                                km, jm] = self.bins_m1[km, jm] + datam1[jm]
                            self.bins_m2[
                                km, jm] = self.bins_m1[km, jm] + datam1[jm]
                            self.illumm[km, jm] = self.illum[km, jm] + 1
                        except IndexError as e:
                            if not len(datam1) or not len(datam2):
                                continue
                            else:
                                raise IndexError(e)

            else:
                for kk, jj, datal, _, _ in out:
                    for k, j, data in zip(
                            kk, jj, datal):
                        self.bins[k, j] = self.bins[k, j] + data[j]

                        # hit counter + 1
                        self.illum[k, j] = self.illum[k, j] + 1

        end = time.time()
        self.logger.info("Stacking finished.")
        print('Stacking finished.')
        self.logger.info(dt_string(end-start))
        self.conclude_ccp()

        # save file
        if save:
            self.write(filename=save)

    def multicore_stack(self, stream, append_pp, n_closest_points, vmodel,
                        latb, lonb, filt, idx, multiple):
        """
        Takes in chunks of data to be processed on one core.

        :param stream: List of file locations.
        :type stream: list
        :param append_pp: Should piercing points be appended?.
        :type append_pp: Bool
        :param n_closest_points: Number of Closest points that the KDTree
            should query.
        :type n_closest_points: int
        :param vmodel: Name of the velocity model that should be used for the
            raytraycing.
        :type vmodel: str
        :param latb: Tuple in Form (minlat, maxlat). To save RAM on 3D
            raytraycing. Will remain unused for 1D RT.
        :type latb: Tuple
        :param lonb: Tuple in Form (minlon, maxlon). To save RAM on 3D
            raytraycing. Will remain unused for 1D RT
        :type lonb: Tuple
        :param filt: Should the RFs be filtered before the ccp stack?
            If so, provide (lowcof, highcof).
        :type filt: bool or tuple]
        :param idx: Index for progress bar
        :type idx: int
        :return: Three lists containing indices and rf-data.
        :rtype: list, list, list
        """

        kk = []
        jj = []
        datal = []
        datalm1 = []
        datalm2 = []

        for st in stream:
            # read RFs in time domain
            try:
                rft = read_rf(st, format='SAC')
            except (IndexError, Exception) as e:
                # That happens when there is a corrupted sac file
                self.logger.exception(e)
                continue

            if filt:
                rft.filter(
                    'bandpass', freqmin=filt[0], freqmax=filt[1],
                    zerophase=True, corners=2)
            try:
                z, rf, rfm1, rfm2 = rft[0].moveout(
                    vmodel, latb=latb, lonb=lonb, taper=False,
                    multiple=multiple)
            except ComplexModel.CoverageError as e:
                # Wrong stations codes can raise this
                self.logger.warning(e)
                continue
            except Exception as e:
                # Just so the script does not interrupt. Did not occur up
                # to now
                self.logger.exception(e)
                continue

            lat = np.array(rf.stats.pp_latitude)
            lon = np.array(rf.stats.pp_longitude)
            if append_pp:
                plat = np.pad(
                    lat, (0, len(self.z)-len(lat)), constant_values=np.nan)
                plon = np.pad(
                    lon, (0, len(self.z)-len(lon)), constant_values=np.nan)
                self.pplat.append(plat)
                self.pplon.append(plon)
            k, j = self.query_bin_tree(lat, lon, rf.data, n_closest_points)

            kk.append(k)
            jj.append(j)
            datal.append(rf.data)

            if multiple:
                depthi = np.where(z == maxzm)[0][0]
                try:
                    datalm1.append(rfm1.data[:depthi+1])
                    datalm2.append(rfm2.data[:depthi+1])
                except AttributeError:
                    # for Interpolationerrors
                    datalm1.append(None)
                    datalm2.append(None)

        return kk, jj, datal, datalm1, datalm2

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

            # self.ccp = np.divide(self.bins, self.illum+1)
            self.ccp = np.hstack(((
                np.divide(self.bins[:, :endi], self.illum[:, :endi]+1) +
                np.divide(
                    self.bins_m1[:, :endi], self.illumm[:, :endi]+1) +
                np.divide(
                    self.bins_m2[:, :endi], self.illumm[:, :endi]+1))/3,
                np.zeros(self.bins[:, endi:].shape)))
        elif multiple == 'zk':
            # self.ccp = np.divide(self.bins, self.illum+1)
            self.ccp = np.hstack((
                .7*np.divide(self.bins[:, :endi], self.illum[:, :endi]+1) +
                .2*np.divide(self.bins_m1[:, :endi], self.illumm[:, :endi]+1) +
                .1*np.divide(self.bins_m2[:, :endi], self.illumm[:, :endi]+1),
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
            raise ValueError('The requested multiple stacking mode is \
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
            # bm = Basemap(
            #     resolution='c', projection='cyl',
            #     llcrnrlat=self.coords[0][0].min(),
            #     llcrnrlon=self.coords[1][0].min(),
            #     urcrnrlat=self.coords[0][0].max(),
            #     urcrnrlon=self.coords[1][0].max())

            if keep_empty:
                self.coords_new = self.coords.copy()

            lats, lons = self.coords_new
            # index = []  # list of indices that contain water
            # list of indices that contain water
            index = globe.is_ocean(lats, lons)[0, :]
            print(index.shape)

            # for i, (lat, lon) in enumerate(zip(lats[0], lons[0])):
            #     # if not bm.is_land(lon, lat):
            #     if not globe.is_land(lat, lon):
            #         index.append(i)
            self.ccp = np.delete(self.ccp, index, 0)
            self.hits = np.delete(self.hits, index, 0)
            self.coords_new = (np.delete(self.coords_new[0], index, 1),
                               np.delete(self.coords_new[1], index, 1))

        # Else everything will always pick coords_new instead of coords
        if keep_water and keep_empty:
            try:
                del self.coords_new
            except NameError:
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
        :param fmt: Either "pickle" or "matlab" for .mat, defaults to "pickle".
        :type fmt: str, optional
        :raises ValueError: For unknown formats.
        """
        # delete logger (cannot be pickled)
        try:
            del self.logger
        except AttributeError:
            # For backwards compatibility
            pass

        # Standard filename
        if not filename:
            filename = self.bingrid.phase + '_' + str(self.bingrid.edist) + \
                '_' + str(self.binrad)

        # Remove filetype identifier if provided
        x = filename.split('.')
        if len(x) > 1:
            if x[-1] == 'pkl' or x[-1] == 'mat':
                filename = ''.join(x[:-1])

        # output location
        oloc = os.path.join(folder, filename)

        if fmt == "pickle":
            with open(oloc + ".pkl", 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        # Save as Matlab file (exporting to plot)
        elif fmt == "matlab":

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

        # Unknown formats
        else:
            raise ValueError("The format type ", fmt, "is unkown.")

    def plot_bins(self):
        """
        A simple map view of the bins.
        """
        if hasattr(self, 'coords_new'):
            coords = self.coords_new
        else:
            coords = self.coords
        plot_bins(self.bingrid.stations, coords)

    def compute_kdtree_volume(self,
                              qlon: np.ndarray or list = None,
                              qlat: np.ndarray or list = None,
                              zmax: float = None):
        """Using the CCP kdtree, we get the closest few points and compute
        the weighting using a distance metric. if points are too far away,
        they aren't weighted


        Parameters
        ----------
        qlon : np.ndarray or list
            grid defining array for longitude
        qlat : np.ndarray or list
            grid defining array for latitude
        gz : np.array or list
            grid defining array for z
        r : float or None
            outside r everything is nan
        minillum: int or None, optional
            Minimum number of illumation points use in the interpolation,
            everything below is downweighted by the square reciprocal

        Returns
        -------
        [type]
            [description]
        """

        # Area considered around profile points
        area = 2 * self.binrad

        # Get global bounds
        if qlat is None or qlon is None:

            minlat = np.min(self.coords_new[0])
            maxlat = np.max(self.coords_new[0])
            minlon = np.min(self.coords_new[1])
            maxlon = np.max(self.coords_new[1])
            minlon, maxlon, minlat, maxlat = fix_map_extent(
                [minlon, maxlon, minlat, maxlat, ])

            # Create mesh vectors
            qlat = np.arange(minlat, maxlat + self.binrad/2, self.binrad/2)
            qlon = np.arange(minlon, maxlon + self.binrad/2, self.binrad/2)

        else:

            minlat, maxlat = np.min(qlat), np.max(qlat)
            minlon, maxlon = np.min(qlon), np.max(qlon)
            minlon, maxlon, minlat, maxlat = fix_map_extent(
                [minlon, maxlon, minlat, maxlat])

        # Create mesh
        mlat, mlon = np.meshgrid(qlat, qlon)

        # smaller/larger is fine as the extent was fixed.
        cpos = np.where(
            (self.coords_new[0] > minlat) &
            (self.coords_new[0] < maxlat) &
            (self.coords_new[1] > minlon) &
            (self.coords_new[1] < maxlon)
        )[1]

        # Get interpolation weights and rows.
        snn = SphericalNN(
            self.coords_new[0][:, cpos], self.coords_new[1][:, cpos])
        ccp_interpolator = snn.interpolator(
            mlat, mlon, maximum_distance=area, k=10, p=2.0, no_weighting=False)
        ill_interpolator = snn.interpolator(
            mlat, mlon, maximum_distance=area, no_weighting=True)

        # Get coordinates array from CCPStack
        qz = deepcopy(self.z)

        if zmax is not None:
            pos = np.argmin(np.abs(qz-zmax))
            qz = qz[:pos]
        else:
            pos = len(qz)

        # Get data arrays
        Nz = pos
        Nlat, Nlon = len(qlat), len(qlon)

        # Interpolate
        qccp = np.zeros((Nlat, Nlon, Nz))
        qill = np.zeros((Nlat, Nlon, Nz))

        # Interpolate each depth
        for _i, (_ccpcol, _illumcol) in enumerate(
                zip(self.ccp[cpos, :pos].T, self.hits[cpos, :pos].T)):
            qccp[:, :, _i] = ccp_interpolator(_ccpcol).T
            qill[:, :, _i] = ill_interpolator(_illumcol).T

        # Interpolate onto regular depth grid for easy representation with
        # imshow
        ccp3D_interpolator = RegularGridInterpolator(
            (qlat, qlon, qz), np.where(np.isnan(qccp), 0, qccp))
        ill3D_interpolator = RegularGridInterpolator(
            (qlat, qlon, qz), qill)

        # Where to sample
        qqz = np.arange(np.min(qz), np.max(qz), 1)
        xqlat, xqlon, xqz = np.meshgrid(qlat, qlon, qqz)

        # Interpolate
        qccp = ccp3D_interpolator((xqlat, xqlon, xqz))
        qill = ill3D_interpolator((xqlat, xqlon, xqz))

        return qlat, qlon, qz, qill, qccp, area

    def create_vtk_mesh(self, geo=True,
                        bbox: list or None = None,
                        filename: str or None = None):
        """Creates a mesh with given bounding box s

        Parameters
        ----------
        geo : bool, optional
            flag whether the output mesh is in geographical coordinates or
            meters, by default True
        bbox : list or None, optional
            bounding box [minlon, maxlon, minlat, maxlat]. If None
            No boundaries are taken, by default None
        filename : str or None, optional
            If set, the computed grid will be output as VTK file under the given
            filename. This file can then later be opened using either the
            plotting tool or, e.g., Paraview. If None, no file is written.
            By default, None.

        Returns
        -------
        VTK.UnstructuredGrid
            outputs a vtk mesh that can be opened in, e.g., Paraview.
        """

        # Get coordinates
        lat = np.squeeze(self.coords_new[0])
        lon = np.squeeze(self.coords_new[1])

        # Filter ccpstacks if not none
        if bbox is None:
            bbox = [-180, 180, -90, 90]
        pos = np.where(((bbox[0] <= lon) & (lon <= bbox[1]) &
                        (bbox[2] <= lat) & (lat <= bbox[3])))[0]
        lat = lat[pos]
        lon = lon[pos]

        # Create VTK point cloud at the surface to triangulate.
        r = R_EARTH * np.ones_like(lat)
        points = np.vstack((deepcopy(lon*DEG2KM).flatten(),
                            deepcopy(lat*DEG2KM).flatten(),
                            deepcopy(r).flatten())).T
        pc = pv.PolyData(points)

        # Triangulate 2D surface
        mesh = pc.delaunay_2d(alpha=self.binrad*1.5*DEG2KM)

        # Use triangles and their connectivity to create 3D Mesh of wedges
        points = deepcopy(mesh.points)
        n_points = mesh.n_points

        # Get cells and create first layer of wedges at the surface
        cells = mesh.faces.reshape(mesh.n_cells, 4)
        cells[:, 0] = 6
        cells = np.hstack((cells, n_points + cells[:, 1:]))
        newcells = deepcopy(cells)

        # Give second layer of points in the wedge the right depth!
        zpoints = deepcopy(np.array(points))
        zpoints[:, 2] = R_EARTH - self.z[1]
        newpoints = np.vstack((points, zpoints))

        # Loop over remaining depths to populated the mesh.
        for _z in self.z[2:]:

            # Add cells
            extra_cells = cells
            extra_cells[:, 1:] += n_points
            newcells = np.vstack((newcells, extra_cells))

            # Add points
            zpoints = deepcopy(np.array(points))
            zpoints[:, 2] = R_EARTH - _z
            newpoints = np.vstack((newpoints, zpoints))

        # Define Cell types
        newcelltypes = np.array(
            [vtk.VTK_WEDGE] * newcells.shape[0], dtype=np.uint8)

        # Redefine location of the points if! Geo location is wanted instead of
        # Cartesian(-ish)
        if geo:
            x, y, z = geo2cart(
                newpoints[:, 2],
                newpoints[:, 1] * KM2DEG,
                newpoints[:, 0] * KM2DEG)

            newpoints = np.vstack((x, y, z)).T

        # Create Unstructured Grid
        grid = pv.UnstructuredGrid(newcells, newcelltypes, newpoints)

        # Populate with RF and illumination values
        grid['RF'] = deepcopy(self.ccp[pos, :].T.ravel())
        grid['illumination'] = deepcopy(self.hits[pos, :].T.ravel())

        # If file name is set write unstructured grid to file!
        if filename is not None:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetInputData(grid)
            writer.SetFileName(filename)
            writer.Write()

        return grid

    def explore(self, qlon: np.ndarray or list = None,
                qlat: np.ndarray or list = None,
                zmax: float = None):
        """Creates a volume exploration window set. One window is for all
        plots, and the other window is generated with sliders such that one
        can explore how future plots should be generated. It technically does
        not require any inputs, as simply the binning size will be used to
        create a meshgrid fitting to the bin distance distribution.
        One can however set a factor by which to divide the bin distance for a
        finer grid.


        Parameters:
        -----------

        factor: float, optional
            Bingrid epicentral distance multiplier to refine grid.
            Defaults to 0.5.
        maxz: float or None, optional
            Maximum Depth if None, the max ccp bin depth is used.
            Defaults to None.
        minillum: int or None, optional
            Minimum number of illumation points use in the interpolation,
            everything below is downweighted by the square reciprocal
        extent : list or tuple or Non, optional
        Returns
        -------

        VolumeExploration

        """
        # Compute the volume max radius at depth is z*0.33
        qlat, qlon, qz, qill, qccp, area = \
            self.compute_kdtree_volume(qlon, qlat, zmax=zmax)

        # Launch plotting tool
        return VolumeExploration(qlon, qlat, qz, qccp)

    def plot_volume_sections(self, qlon: np.ndarray, qlat: np.ndarray,
                             zmax: float or None = None,
                             lonsl: float or None = None,
                             latsl: float or None = None,
                             zsl: float or None = None,
                             r: float or None = None,
                             minillum: int or None = None):
        """Creates the same plot as the `explore` tool, but(!) statically and
        with more options left to the user.

        !!! IT IS IMPORTANT for the plotting function that the input arrays
        are monotonically increasing!!!

        Parameters
        ----------
        qlon : np.ndarray
            Monotonically increasing ndarray. No failsafes implemented.
        qlat : np.ndarray
            Monotonically increasing ndarray. No failsafes implemented.
        z : np.ndarray
            Monotonically increasing ndarray. No failsafes implemented.
        lonsl : float or None, optional
            Slice location, if None, the center of lon is used,
            by default None
        latsl : float or None, optional
            Slice location, if None, the center of lat is used,
            by default None
        zsl : float or None, optional
            Slice location, if None, the center of depth is used,
            by default None
        r : float or None, optional
            Max radius for interpolation values taken into account [km],
            if None bingrid.edist * DEG2KM * 2.0, by default None
        minillum: int or None, optional
            Minimum number of illumation points use in the interpolation,
            everything below is downweighted by the square reciprocal

        Returns
        -------
        VolumePlot
            [description]
        """

        # Change distance to delete things
        if r is None:
            r = self.bingrid.edist * DEG2KM * 2.0

        # Compute the volume max radius at depth is z*0.33
        qlat, qlon, qz, qill, qccp, area = \
            self.compute_kdtree_volume(qlon, qlat, zmax=zmax)

        return VolumePlot(qlon, qlat, qz, qccp, xl=lonsl, yl=latsl, zl=zsl)

    def map_plot(
        self, plot_stations: bool = False, plot_bins: bool = False,
        plot_illum: bool = False, profile: list or tuple or None = None,
        p_direct: bool = True, outputfile: str or None = None,
        format: str = 'pdf', dpi: int = 300, geology: bool = False,
            title: str or None = None):
        """
        Create a map plot of the CCP Stack containing user-defined information.

        Parameters
        ----------
        plot_stations : bool, optional
            Plot the station locations, by default False
        plot_bins : bool, optional
            Plot bin location, by default False
        plot_illum : bool, optional
            Plot bin location with colour depending on the depth-cumulative
            illumination at bin b, by default False
        profile : list or tupleor None, optional
            Plot locations of cross sections into the plot. Information about
            each cross section is given as a tuple (lon1, lon2, lat1, lat2),
            several cross sections are given as a list of tuples in said
            format, by default None.
        p_direct : bool, optional
            If true the list in profile decribes areas with coordinates of the
            lower left and upper right corner, by default True
        outputfile : str or None, optional
            Save plot to file, by default None
        format : str, optional
            Format for the file to be saved as, by default 'pdf'
        dpi : int, optional
            DPI for none vector format plots, by default 300
        geology : bool, optional
            Plot a geological map. Requires internet connection
        title : str, optional
            Set title for plot.
        """
        if hasattr(self, 'coords_new') and self.coords_new[0].size:
            bincoords = self.coords_new
        else:
            bincoords = self.coords
        # Set boundaries for map:
        lat = (
            np.floor(min(bincoords[0][0]))-1, np.ceil(max(bincoords[0][0])+1))
        lon = (np.floor(min(bincoords[1][0]))-1,
               np.ceil(max(bincoords[1][0])+1))
        plot_map_ccp(
            lat, lon, plot_stations, self.bingrid.latitude,
            self.bingrid.longitude, plot_bins, bincoords, self.bingrid.edist,
            plot_illum, self.hits, profile, p_direct, outputfile=outputfile,
            format=format, dpi=dpi, geology=geology, title=title)

    def pick_phase(
            self, pol: str = '+', depth: list = [None, None]) -> PhasePick:
        """
        Pick the strongest negative or strongest positive gradient from a
        predefined depth-range.

        Parameters
        ----------
        pol : str, optional
            Either '+' for positive velocity gradient or '-' for negative
            gradient, by default '+'
        depth : list, optional
            List with two elements [minz, maxz], defines in which depth window
            the phase picker is gonna look for maxima and minima,
            by default [None, None]

        Returns
        -------
        PhasePick
            A phasepick object, which subsequently can be plotted.

        Raises
        ------
        ValueError
            For errandeous inputs.
        """
        self.conclude_ccp(r=3)
        # Find indices
        for ii, d in enumerate(depth):
            if d or d == 0:
                depth[ii] = np.abs(self.z-d).argmin()
        # Find minimum in vector
        ccp_short = self.ccp[:, depth[0]:depth[1]]
        # There should be a line here excluding insufficiently illuminated bins
        illum_flat = np.sum(self.hits[:, depth[0]:depth[1]], axis=1)
        id = np.where(illum_flat < ccp_short.shape[1]*10)  # delete those bins

        ccp_short = np.delete(ccp_short, id, 0)
        coords = (
            np.delete(self.coords_new[0], id, 1),
            np.delete(self.coords_new[1], id, 1))

        if pol == '+':
            # Amplitudes
            a = ccp_short.max(axis=1)
            # Depths
            z = self.z[depth[0]:depth[1]][ccp_short.argmax(axis=1)]
        elif pol == '-':
            a = ccp_short.min(axis=1)
            z = self.z[depth[0]:depth[1]][ccp_short.argmin(axis=1)]
        else:
            raise ValueError(
                'Choose either \'+\' to return the highest ' +
                'positive velocity gradient or \'-\' to return the ' +
                'highest negative velocity gradient.')

        p_phase = PhasePick(coords, a, pol, z, depth)
        return p_phase


def read_ccp(filename: str, fmt: str or None = None) -> CCPStack:
    """
    Read CCP-Stack class file from input folder.

    :param filename: Filename of the input file with file ending.
        The default is 'ccp.pkl'.
    :type filename: str, optional
    :param fmt: File format, can be none if the filename has an ending,
        possible options are "pickle. The default is None.
    :type fmt: str, optional
    :raises ValueError: For unknown formats
    :return: CCPStack object
    :rtype: :class:`~pyglimer.ccp.ccp.CCPStack`
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
        with open(filename, 'rb') as infile:
            ccp = pickle.load(infile)
    else:
        raise ValueError("Unknown format ", fmt)

    return ccp


def init_ccp(
    spacing: float, vel_model: str, phase: str,
    statloc: str = 'output/stations',
    preproloc: str = 'output/waveforms/preprocessed',
    rfloc: str = 'output/waveforms/RF', network: str or list or None = None,
    station: str or list or None = None, geocoords: tuple or None = None,
    compute_stack: bool = False, filt: tuple or None = None,
    binrad: float = np.cos(np.radians(30)), append_pp: bool = False,
    multiple: bool = False, save: str or bool = False, verbose: bool = True
) -> CCPStack:
    """
    Computes a ccp stack in self.ccp using data from statloc and rfloc.
    The stack can be limited to some networks and
    stations.

    :param spacing: Angular distance between each bingrid point.
    :type spacing: float
    :param vel_model: Velocity model located in data. Either iasp91 (1D
        raytracing) or 3D for 3D raytracing using a model compiled from GyPSuM.
    :type vel_model: str
    :param phase: Either 'S' or 'P'. Use 'S' if dataset contains both SRFs and
        PRFs.
    :type phase: str
    :param statloc: Directory in that the station xmls are saved,
        defaults to 'output/stations'
    :type statloc: str, optional
    :param preproloc: Parental directory in that the preprocessed miniseeds are
        saved, defaults to 'output/waveforms/preprocessed'
    :type preproloc: str, optional
    :param rfloc: Parental directory in that the RFs are saved,
        defaults to 'output/waveforms/RF'
    :type rfloc: str, optional
    :param network: Network or networks that are to be included in the ccp
        stack.
        Standard unix wildcards are allowed. If None, the whole database
        will be used. The default is None., defaults to None
    :type network: str or list, optional
    :param station: Station or stations that are to be included in the ccp
        stack. Standard unix wildcards are allowed. Can only be list if
        type(network)=str. If None, the whole database will be used.
        The default is None.
    :type station: str or list, optional
    :param geocoords: An alternative way of selecting networks and stations.
        Given in the form (minlat, maxlat, minlon, maxlon), defaults to None
    :type geocoords: Tuple, optional
    :param compute_stack: If true it will compute the stack by calling
        :func:`~pyglimer.ccp.ccp.CCPStack.compute_stack()`.
        That can take a long time! The default is False.
    :type compute_stack: bool, optional
    :param filt: Decides whether to filter the receiver function prior to
            depth migration. Either a Tuple of form `(lowpassf, highpassf)` or
            `None` / `False`.
    :type filt: tuple, optional
    :param binrad: Only used if compute_stack=True
            Defines the bin radius with bin radius = binrad*distance_bins.
            Full Overlap = cosd(30), Full coverage: 1.
            The default is full overlap.
    :type binrad: float, optional
    :param append_pp: Only used if compute_stack=True.
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
    :type save: ool or str, optional
    :param verbose: Display info in terminal., defaults to True
    :type verbose: bool, optional
    :raises TypeError: For wrong inputs.
    :return: CCPStack object.
    :rtype: :class:`~pyglimer.ccp.ccp.CCPstack`
    """
    if phase[-1].upper() == 'S' and multiple:
        raise NotImplementedError(
            'Multiple mode is not supported for phase S.')
    # create empty lists for station latitude and longitude
    lats = []
    lons = []

    # Were network and stations provided?
    # Possibility 1 as geo boundaries
    if geocoords:
        lat = (geocoords[0], geocoords[1])
        lon = (geocoords[2], geocoords[3])
        db = StationDB(preproloc, phase=phase, use_old=False)
        net, stat = db.find_stations(lat, lon, phase=phase)
        pattern = ["{}.{}".format(a_, b_) for a_, b_ in zip(net, stat)]
        files = []
        for pat in pattern:
            files.extend(
                fnmatch.filter(os.listdir(statloc), pat+'.xml'))

    # As strings
    elif network and type(network) == list:
        files = []
        if station:
            if type(station) != list:
                raise TypeError(
                    """Provide a list of stations, when using a list of
                    networks as parameter.""")
            for net in network:
                for stat in station:
                    pattern2 = net + '.' + stat + '.xml'
                    files.extend(
                        fnmatch.filter(os.listdir(statloc), pattern2))
        else:
            for net in network:
                pattern2 = net + '.*.xml'
                files.extend(
                    fnmatch.filter(os.listdir(statloc), pattern2))
    elif network and type(station) == list:
        files = []
        for stat in station:
            pattern2 = network + '.' + stat + '.xml'
            files.extend(fnmatch.filter(os.listdir(statloc), pattern2))
    elif network:
        pattern2 = network + '.' + (station or '*') + '.xml'
        files = fnmatch.filter(os.listdir(statloc), pattern2)

    # In this case, it will process all available data (for global ccps)
    else:
        files = os.listdir(statloc)

    # read out station latitudes and longitudes
    for file in files:
        try:
            stat = read_inventory(os.path.join(statloc, file))
        except TypeError as e:
            print(
                "Corrupt station xml, original error message: %s" % e)
        lats.append(stat[0][0].latitude)
        lons.append(stat[0][0].longitude)

    logdir = os.path.join(os.path.dirname(os.path.abspath(statloc)), 'logs')

    ccp = CCPStack(
        lats, lons, spacing, phase=phase, verbose=verbose, logdir=logdir)

    # Clear Memory
    del stat, lats, lons, files

    if not geocoords:
        pattern = None

    if compute_stack:
        ccp.compute_stack(
            vel_model=vel_model, network=network, station=station, save=save,
            filt=filt, multiple=multiple,
            pattern=pattern, append_pp=append_pp, binrad=binrad, rfloc=rfloc)

    # _MODEL_CACHE.clear()  # So the RAM doesn't stay super full

    return ccp
