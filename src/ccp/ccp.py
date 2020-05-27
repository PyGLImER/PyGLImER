'''
Author: Peter Makus (peter.makus@student.uib.no)

Created: Friday, 10th April 2020 05:30:18 pm
Last Modified: Wednesday, 27th May 2020 10:30:54 am
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fnmatch
import os
import pickle
import logging
import time
from joblib import Parallel, delayed, cpu_count

import numpy as np
from obspy import read_inventory
import scipy.io as sio
from mpl_toolkits.basemap import Basemap
from pathlib import Path
import subprocess

import config
from .compute.bin import BinGrid
from ..database.stations import StationDB
from ..rf.create import read_rf
from ..rf.moveout import res, maxz
from ..utils.utils import dt_string, chunks
from ..utils.createvmodel import _MODEL_CACHE, ComplexModel
from ..utils.geo_utils import epi2euc
from .plot_utils.plot_bins import plot_bins

# Loggers for the CCP script
logger = logging.Logger('PyGlimer.src.ccp.ccplogger')
logger.setLevel(logging.INFO)

# Create handler to the log
fh = logging.FileHandler('logs/ccp.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# Create Formatter
fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)


def init_ccp(spacing, vel_model, phase=config.phase, network=None,
             station=None, geocoords=None,
             compute_stack=False, binrad=np.cos(np.radians(30)),
             append_pp=False, save=False, verbose=True):
    """
    Computes a ccp stack in self.ccp using the standard folder
    structure set in config. The stack can be limited to some networks and
    stations.

    Parameters
    ----------
    spacing : float
        Angular distance between each bin point.
    vel_model : str
        Velocity model located in data. Either iasp91 (1D raytracing) or
        3D for 3D raytracing using a model compiled from GyPSuM.
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
    geocoords : Tuple, optional
        An alternative way of selecting networks and stations. Given
        in the form (minlat, maxlat, minlon, maxlon)
    compute_stack : Bool, optional
        If true it will compute the stack by calling ccp.compute_stack().
        That can take a long time! The default is False.
    binrad : float, optional
            Only used if compute_stack=True
            Defines the bin radius with bin radius = binrad*distance_bins.
            Full Overlap = cosd(30), Full coverage: 1.
            The default is full overlap
    append_pp : Bool, optional
        Only used if compute_stack=True.
        Appends piercing point locations to object. Can be used to plot
        pps on map. Not recommended for large datasets as it takes A LOT
        longer and makes the file a lot larger.
        The default is False
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
    # Possibility 1 as geo boundaries
    if geocoords:
        lat = (geocoords[0], geocoords[1])
        lon = (geocoords[2], geocoords[3])
        db = StationDB()
        net, stat = db.find_stations(lat, lon, phase=phase)
        pattern = ["{}.{}".format(a_, b_) for a_, b_ in zip(net, stat)]
        files = []
        for pat in pattern:
            files.extend(
                fnmatch.filter(os.listdir(config.statloc), pat+'.xml'))

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
                    pattern = net + '.' + stat
                    files.extend(
                        fnmatch.filter(os.listdir(config.statloc), pattern))
        else:
            for net in network:
                pattern = net + '.' + (station or '') + '*'
                files.extend(
                    fnmatch.filter(os.listdir(config.statloc), pattern))
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
        stat = read_inventory(os.path.join(config.statloc, file))
        lats.append(stat[0][0].latitude)
        lons.append(stat[0][0].longitude)

    ccp = CCPStack(
        lats, lons, spacing, phase=phase, verbose=verbose)

    # Clear Memory
    del stat, lats, lons, files

    if compute_stack:
        ccp.compute_stack(
            vel_model=vel_model, network=network, station=station, save=save,
            pattern=pattern, append_pp=append_pp, binrad=binrad)

    # _MODEL_CACHE.clear()  # So the RAM doesn't stay super full

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
        # self.z = np.arange(0, maxz+res, res)
        self.z = np.hstack(
            (np.arange(-10, 0, .1), np.arange(0, maxz+res, res)))
        self.bins = np.zeros([self.coords[0].size, len(self.z)])
        self.illum = np.zeros(np.shape(self.bins))
        self.pplat = []
        self.pplon = []
        self.ppz = self.z
        self.binrad = None

    def query_bin_tree(self, latitude, longitude, data, n_closest_points):
        """
        Find closest bins for given latitude and longitude.

        Parameters
        ----------
        latitude : 1D np.array
            Latitudes of piercing points.
        longitude : 1D np.array
            Longitudes of piercing points.
        data : 1D np.array
            Depth migrated receiver function data

        Returns
        -------
        None.

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

    def compute_stack(self, vel_model, network=None, station=None, pattern=None,
                      save=False,
                      binrad=np.cos(np.radians(30)), append_pp=False):
        """
        Computes a ccp stack in self.ccp, using the standard folder
        structure set in config. The stack can be limited to some networks and
        stations. This will take a long while for big data sets!
        Note that the grid should be big enough for the provided networks and
        stations. Best to create the CCPStack object by using ccp.init_ccp().

        Parameters
        ----------
        vel_model : str
            Velocity model located in data. Either 'iasp91' for 1D raytracing
            or 3D for 3D raytracing with GYPSuM model. Using 3D Raytracing
            will cause the code to take about 30% longer.
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
        binrad : float, optional
            Defines the bin radius with bin radius = binrad*distance_bins.
            Full Overlap = cosd(30), Full coverage: 1.
            The default is full overlap
        append_pp : Bool, optional
            appends piercing point coordinates if True, so tey can later be
            plotted. Not recommended for big data sets.
            The default is false.


        Returns
        -------
        None (Data are appended to ccp object).

        """

        self.binrad = binrad*self.bingrid.edist

        # How many closest points are queried by the bintree?
        # See Gauss circle problem
        # sum of squares for 2 squares for max binrad=4
        # Using the ceiling funciton to account for inaccuracies
        sosq = [1, 4, 4, 0, 4, 8, 0, 0, 4, 4, 8, 0, 0, 8, 0, 0, 4]
        try:
            n_closest_points = sum(sosq[0:int(np.ceil(binrad**2+1))])
        except IndexError:
            raise ValueError(
                """Maximum allowed binradius is 4 times the bin distance""")

        # Compute maxdist in euclidean space
        self.binrad_eucl = epi2euc(self.binrad)

        folder = config.RF[:-1] + self.bingrid.phase

        start = time.time()
        logger.info('Stacking started')

        if network and type(network) == str and not pattern:
            # Loop over fewer files
            folder = os.path.join(folder, network)
            if station and type(station) == str:
                folder = os.path.join(folder, station)

        infiles = []  # List of all files in folder
        if not pattern:
            pattern = []  # List of input constraints
        streams = []  # List of files filtered for input criteria

        for root, dirs, files in os.walk(folder):
            for name in files:
                infiles.append(os.path.join(root, name))

        # Special rule for files imported from Matlab
        if network == 'matlab' or network == 'raysum':
            pattern.append('*.sac')

        elif pattern:
            pattern = ["*{}.*.sac".format(_a) for _a in pattern]

        # Set filter patterns
        elif network:
            if type(network) == list:
                if station:
                    if type(station) == list:
                        for net in network:
                            for stat in station:
                                pattern.append('*%s*%s*.sac' % (net, stat))
                    else:
                        raise ValueError("""The combination of network
                                         and station are invalid""")
                else:
                    for net in network:
                        pattern.append('*%s*.sac' % (net))
            elif type(network) == str:
                if station:
                    if type(station) == str:
                        pattern.append('*%s*%s*.sac' % (net, station))
                    elif type(station) == list:
                        for stat in station:
                            pattern.append('*%s*%s*.sac' % (net, stat))
                else:
                    pattern.append('*%s*.sac' % (network))
        elif station:
            raise ValueError("""You have to provide both network and station
                             code if you want to filter by station""")
        else:
            # Global CCP
            pattern.append('*.sac')

        # Do filtering
        for pat in pattern:
            streams.extend(fnmatch.filter(infiles, pat))

        # clear memory
        del pattern, infiles

        # Data counter
        self.N = self.N + len(streams)

        # Split job into n chunks
        num_cores = cpu_count()

        # Actual CCP stack
        # Note loki does mess up the output and threads is slower than
        # using a single core

        # The test data needs to be filtered
        if network == 'matlab':
            filt = [.03, 1.5]  # bandpass frequencies
        else:
            filt = False

        # Define grid boundaries for 3D RT
        latb = (self.coords[0].min(), self.coords[0].max())
        lonb = (self.coords[1].min(), self.coords[1].max())

        out = Parallel(n_jobs=num_cores, prefer='processes')(
                        delayed(self.multicore_stack)(
                            st, append_pp, n_closest_points, vel_model, latb,
                            lonb, filt)
                        for st in chunks(streams, num_cores))

        # The stacking is done here (one should not reassign variables in
        # multi-core processes).
        # Awful way to solve it, but the best I could find
        for kk, jj, datal in out:
            for k, j, data in zip(kk, jj, datal):
                self.bins[k, j] = self.bins[k, j] + data[j]

                # hit counter + 1
                self.illum[k, j] = self.illum[k, j] + 1

        end = time.time()
        logger.info("Stacking finished.")
        logger.info(dt_string(end-start))
        self.conclude_ccp()

        # save file
        if save:
            self.write(filename=save)

    def multicore_stack(self, stream, append_pp, n_closest_points, vmodel,
                        latb, lonb, filt=False):
        """
        Takes in chunks of data to be processed on one core.

        Parameters
        ----------
        stream : list
            List of file locations.
        append_pp : Bool
            Should piercing points be appended?.
        n_closest_points : int
            Number of Closest points that the KDTree should query.
        vmodel : str
            Name of the velocity model that should be used for the raytraycing.
        latb : Tuple
            Tuple in Form (minlat, maxlat). To save RAM on 3D raytraycing.
            Will remain unused for 1D RT.
        lonb : Tuple
            Tuple in Form (minlon, maxlon).
        filter : Bool, Tuple - optional
            Should the RFs be filtered before the ccp stack? If so, provide
            (lowcof, highcof).

        Returns
        -------
        None.

        """
        kk = []
        jj = []
        datal = []
        for st in stream:
            rft = read_rf(st, format='SAC')

            if filt:
                rft.filter('bandpass', freqmin=filt[0], freqmax=filt[1],
                           zerophase=True, corners=2)
            try:
                _, rf = rft[0].moveout(
                    vmodel, latb=latb, lonb=lonb, taper=False)
            except ComplexModel.CoverageError as e:
                # Wrong stations codes can raise this
                logger.warning(e)
                continue

            lat = np.array(rf.stats.pp_latitude)
            lon = np.array(rf.stats.pp_longitude)
            if append_pp:
                plat = np.pad(lat, (0, len(self.z)-len(lat)),
                              constant_values=np.nan)
                plon = np.pad(lon, (0, len(self.z)-len(lon)),
                              constant_values=np.nan)
                self.pplat.append(plat)
                self.pplon.append(plon)
            k, j = self.query_bin_tree(lat, lon, rf.data, n_closest_points)
            kk.append(k)
            jj.append(j)
            datal.append(rf.data)
        return kk, jj, datal

    def conclude_ccp(self, keep_empty=False, keep_water=False, r=3):
        """
        Averages the CCP-bin and populates empty cells with average of
        neighbouring cells. No matter which option is
        chosen for the parameters, data is never lost entirely. One can always
        execute a new conclude_ccp() command. However, decisions that are
        taken here will affect the *.mat output and plotting outputs.


        Parameters
        ----------
        keep_empty : Bool or str, optional
            Keep entirely empty bins. The default is True.
        keep_water : Bool , optional
            For False all bins that are on water-covered areas
            will be discarded. The default is True.
        r : int, optional
            Fields with less than r hits will be set equal 0.
            r has to be >= 1. The default is 1.

        Returns
        -------
        None.

        """
        self.ccp = np.divide(self.bins, self.illum+1)

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
            bm = Basemap(
                resolution='c', projection='cyl',
                llcrnrlat=self.coords[0][0].min(),
                llcrnrlon=self.coords[1][0].min(),
                urcrnrlat=self.coords[0][0].max(),
                urcrnrlon=self.coords[1][0].max())
            if not hasattr(self, 'coords_new'):
                self.coords_new = self.coords.copy()

            lats, lons = self.coords_new
            index = []  # list of indices that contain water

            for i, (lat, lon) in enumerate(zip(lats[0], lons[0])):
                if not bm.is_land(lon, lat):
                    index.append(i)
            self.ccp = np.delete(self.ccp, index, 0)
            self.hits = np.delete(self.hits, index, 0)
            self.coords_new = (np.delete(self.coords_new[0], index, 1),
                               np.delete(self.coords_new[1], index, 1))

    def write(self, filename=None, folder=config.ccp, fmt="pickle"):
        """
        Saves the CCPStream file as pickle or matlab file. Only save
        as Matlab file for exporting, as not all information can be
        preserved!

        Parameters
        ----------
        filename : str, optional
            Name as which to save, file extensions aren't necessary.
        folder : str, optional
            Output folder, standard is defined in config.
        fmt : str, optional
            Either "pickle" or "matlab" for .mat.
        """

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
        oloc = folder + '/' + filename

        if not Path(oloc).is_dir:
            subprocess.call(['mkdir', '-p', oloc])

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
        if hasattr(self, 'coords_new'):
            coords = self.coords_new
        else:
            coords = self.coords
        plot_bins(self.bingrid.stations, coords)
