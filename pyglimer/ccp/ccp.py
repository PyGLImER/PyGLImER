'''
Author: Peter Makus (peter.makus@student.uib.no)

Created: Friday, 10th April 2020 05:30:18 pm
Last Modified: Tuesday, 21st July 2020 12:57:02 pm
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fnmatch
import os
import pickle
import logging
import shutil
import time

from joblib import Parallel, delayed, cpu_count
import numpy as np
from obspy import read_inventory
import scipy.io as sio
from mpl_toolkits.basemap import Basemap
from pathlib import Path
from psutil import virtual_memory
import subprocess
from tqdm import tqdm

from pyglimer.ccp.compute.bin import BinGrid
from pyglimer.database.stations import StationDB
from pyglimer.rf.create import read_rf
from pyglimer.rf.moveout import res, maxz
from pyglimer.utils.utils import dt_string, chunks
from pyglimer.utils.createvmodel import _MODEL_CACHE, ComplexModel
from pyglimer.utils.geo_utils import epi2euc
from pyglimer.ccp.plot_utils.plot_bins import plot_bins


def init_ccp(spacing, vel_model, phase, statloc='output/stations',
             preproloc='output/waveforms/preprocessed',
             rfloc='output/waveforms/RF', network=None,
             station=None, geocoords=None,
             compute_stack=False, filt=None, binrad=np.cos(np.radians(30)),
             append_pp=False, multiple=False, save=False, verbose=True):
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
    :param network: Network or networks that are to be included in the ccp stack.
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
    :param save: Either False if the ccp should not be saved or string with filename
        will be saved. Will be saved as pickle file.
        The default is False.
    :type save: ool or str, optional
    :param verbose: Display info in terminal., defaults to True
    :type verbose: bool, optional
    :raises TypeError: For wrong inputs.
    :return: CCPStack object.
    :rtype: :class:`~pyglimer.ccp.ccp.CCPstack`
    """    
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
        stat = read_inventory(os.path.join(statloc, file))
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


def read_ccp(filename='ccp.pkl', folder='output/ccps', fmt=None):
    """
    Read CCP-Stack class file from input folder.

    :param filename: Filename of the input file with file ending.
        The default is 'ccp.pkl'.
    :type filename: str, optional
    :param folder: Input folder, defaults to 'output/ccps'
    :type folder: str, optional
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
        with open(os.path.join(folder, filename), 'rb') as infile:
            ccp = pickle.load(infile)
    else:
        raise ValueError("Unknown format ", fmt)

    return ccp


class CCPStack(object):
    """Is a CCP Stack matrix. Its size depends upon stations that are used as
    input."""

    def __init__(self, latitude, longitude, edist, phase,
                 verbose=True, logdir: str or None=None):
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
        :param verbose: If true -> console output. The default is True., defaults to True
        :type verbose: bool, optional
        :param logdir: Directory for log file
        :type logdr: str, optional
        """

        # Loggers for the CCP script
        self.logger = logging.Logger('pyglimer.ccp.ccp')
        self.logger.setLevel(logging.INFO)

        # Create handler to the log
        if not logdir:
            fh = logging.FileHandler(os.path.join('logs', 'ccp.log'))
        else:
            fh = logging.FileHandler(os.path.join(logdir, 'ccp.log'))
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        # Create Formatter
        fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
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

    def query_bin_tree(self, latitude, longitude, data, n_closest_points):
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

    def compute_stack(self, vel_model, rfloc='output/waveforms/RF',
                      statloc='output/stations',
                      preproloc='output/waveforms/preprocessed',
                      network=None, station=None, geocoords=None, pattern=None,
                      save=False, filt=None,
                      binrad=np.cos(np.radians(30)), append_pp=False,
                      multiple=False):
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
            Full Overlap = cosd(30), Full coverage: 1.
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

        if binrad < np.cos(np.radians(30)):
            raise ValueError(
                """Minimum allowed binradius is cos(30deg)* bin distance.
                Else the grid will not cover all the surface area."""
            )

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
            endi = np.where(self.z==100)[0][0]+1
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

        for root, dirs, files in os.walk(folder):
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

            out = Parallel(n_jobs=num_cores)( # prefer='processes'
                            delayed(self.multicore_stack)(
                                st, append_pp, n_closest_points, vel_model, latb,
                                lonb, filt, i, multiple)
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
                        iii = np.where(j<=endi-1)[0]
                        jm = j[iii]
                        km = k[iii]
                        try:
                            self.bins_m1[km, jm] = self.bins_m1[km, jm] + datam1[jm]
                            self.bins_m2[km, jm] = self.bins_m1[km, jm] + datam1[jm]
                            self.illumm[km, jm] = self.illum[km, jm] + 1
                        except TypeError:
                            print(type(datam1))
                            continue

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
                rft.filter('bandpass', freqmin=filt[0], freqmax=filt[1],
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
            
            if multiple:
                depthi = np.where(z==100)[0][0]
                try:
                    datalm1.append(rfm1.data[:depthi+1])
                    datalm2.append(rfm2.data[:depthi+1])
                except AttributeError:
                    # for Interpolationerrors
                    datalm1.append(None)
                    datalm2.append(None)

        return kk, jj, datal, datalm1, datalm2

    def conclude_ccp(
        self, keep_empty=False, keep_water=False, r=3,
        multiple=False, z_multiple:int = 100):
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
            stack. By default False.
        :type multiple: bool or str
        :param z_multiple: Until which depth [km] should multiples be considered,
            maximal value is 100 [km]. Will only be used if multiple=True.
            By default 100 km.
        :type z_multiple: int, optional
        """
        if z_multiple > 100:
            raise ValueError('Maximal depth for multiples is 100 km.')
        endi = np.where(self.z == z_multiple)[0][0]+1   
        if multiple == 'linear':
            
            self.ccp = np.divide(self.bins, self.illum+1)
            self.ccp[:, :endi] = (self.ccp[:, :endi] +
                np.divide(self.bins_m1[:, :endi], self.illumm[:, :endi]) +
                    np.divide(self.bins_m2[:, :endi], self.illumm[:, :endi]))/3
        elif multiple == 'zk':
            self.ccp = np.divide(self.bins, self.illum+1)
            self.ccp[:, :endi] = (.7*self.ccp[:, :endi] +
                .2*np.divide(self.bins_m1[:, :endi], self.illumm[:, :endi]) +
                    .1*np.divide(self.bins_m2[:, :endi], self.illumm[:, :endi]))
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

    def write(self, filename=None, folder='output/ccps', fmt="pickle"):
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
