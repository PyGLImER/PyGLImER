"""

Module that covers the binning part of the CCP computation.

Author:
    Lucas Sawade (lsawade@princeton.edu)


Last Update: November 2019

"""

import logging
import time

import numpy as np
from scipy.spatial import KDTree

import config
from src.constants import R_EARTH
# Local imports
# from ... import logger
from src.utils.geo_utils import geo2cart, cart2geo, epi2euc
from src.utils.utils import dt_string

logger = logging.Logger("binlogger")


def fibonacci_sphere(epi=1):
    # Get number of samples from epicentral distance first get euclidean
    # distance d
    d = 2 * np.sin(epi / 180 * np.pi / 2)
    samples = int(np.round(1 / (d / np.sqrt(10)) ** 2))
    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y ** 2)
        phi = ((i + 1) % samples) * increment
        # print(phi)
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])
    return np.array(points)


class BinGrid(object):
    """Creates Bingrid object from parameters"""

    def __init__(self, latitude, longitude, edist, phase=config.phase,
                 verbose=True):
        """

        Parameters
        ----------
        latitude (1-D `numpy.array`): Latitudes

        longitude (1-D `numpy.array`): Longitudes

        edist (float): epicentral distance between bin centres

        phase : str
            Seismic phase either "S" or "P". Phase "S" will lead to more
            grid points being created due to flatter incidence angle. Hence,
            Phase can be "S" for PRFs but not the other way around. However,
            "P" is computationally more efficient.

        verbose: if True --> console output

        Returns
        -------
        BinGrid object that can be used to find closest neighbours to
        stations and CCP bins

        """

        # Populate stuff
        self.stations = np.unique(np.stack((latitude, longitude)).T, axis=0)
        self.latitude = self.stations[:, 0]
        self.longitude = self.stations[:, 1]
        self.edist = edist
        self.phase = phase

        # Verbosity
        self.verbose = verbose

        # number of stations
        self.ns = self.latitude.size

        # Create Cartesian coordinates from lat/lons
        self.xs, self.ys, self.zs = geo2cart(R_EARTH, self.latitude,
                                             self.longitude)

        # Create CCP grid
        self.KDS = self.station_tree()

        # Create CCP Bin kdtree
        # self.KDB =

    def station_tree(self):
        """Using the input"""
        if self.verbose:
            logger.info(" ")
            logger.info("--- Creating KDTree for stations ---")
            logger.info(" ")
            start = time.time()

        K = KDTree(np.vstack((self.xs.reshape((self.ns,)),
                              self.ys.reshape((self.ns,)),
                              self.zs.reshape((self.ns,)))).T)

        if self.verbose:
            end = time.time()
            logger.info("   Finished station KDTree.")
            logger.info(dt_string(end - start))
            logger.info(" ")

        return K

    def bin_tree(self):
        # if 'bins' not in self:
        #     raise Exception("Compute bins before computing the bin_tree.")

        if self.verbose:
            logger.info(" ")
            logger.info("--- Creating KDTree for bins ---")
            logger.info(" ")
            start = time.time()

        self.xb, self.yb, self.zb = geo2cart(R_EARTH, self.bins[0],
                                             self.bins[1])

        K = KDTree(np.vstack((self.xb.reshape((self.Nb,)),
                              self.yb.reshape((self.Nb,)),
                              self.zb.reshape((self.Nb,)))).T)

        if self.verbose:
            end = time.time()
            logger.info("   Finished station KDTree.")
            logger.info(dt_string(end - start))
            logger.info(" ")

        self.KDB = K

    def compute_bins(self):
        """Computes bins close to all stations"""

        if self.verbose:
            logger.info(" ")
            logger.info("--- Computing points on Fibonacci sphere ---")
            logger.info(" ")
            start = time.time()

        # Create points on the sphere using edist
        points = fibonacci_sphere(self.edist)

        # Number of points on fibonacci sphere
        Nf = len(points)

        if self.verbose:
            end = time.time()
            logger.info("   Finished computing points on Fibonacci sphere.")
            logger.info("   Total Number of possible bins: %d" % Nf)
            logger.info(dt_string(end - start))
            logger.info(" ")

        # Query points that are close enough
        # The upper bound of 4 degrees here is chosen because that's an upper
        # bound where there are definitely no conversion anymore. Meaning if
        # a recorded receiver function is converted that far away, there is a
        # bin that captures it.
        if self.verbose:
            logger.info(" ")
            logger.info("--- Checking whether points are close to stations ---")
            logger.info(" ")
            start = time.time()

        # maximal distance of bins to station depends upon phase
        if self.phase == "S":
            maxepid = 10
        elif self.phase == "P":
            maxepid = 4
        d, i = self.KDS.query(R_EARTH * points,
                              distance_upper_bound=epi2euc(maxepid))

        # pick out close enough stations only
        pos = np.where(i < self.ns)

        # Get geolocations
        _, binlat, binlon = cart2geo(points[pos, 0],
                                     points[pos, 1],
                                     points[pos, 2])

        self.bins = (binlat, binlon)
        self.Nb = self.bins[0].size

        if self.verbose:
            end = time.time()
            logger.info("   Finished sorting out bad bins.")
            logger.info("   Total number of accepted bins: %d" % self.Nb)
            logger.info(dt_string(end - start))
            logger.info(" ")

        return self.bins

    def query_station_tree(self, latitude, longitude, maxdist):
        """Uses the query function of the KDTree but takes geolocations as
        input.

        Parameters
        ----------
        latitude: latitudes to be queried
        longitude: longitudes to be queried
        maxdist: maximum  distance to be queried in epicentral distance

        Returns
        -------
        eucd : 1Dnp.array
            Euclidian distances to next point

        i: 1D np.array
            Indices of next neighbour.
        """

        # Compute maxdist in euclidean space
        maxdist_euc = epi2euc(maxdist)

        # Get cartesian coordinate points
        points = geo2cart(R_EARTH, latitude, longitude)

        # Query the points and return the distances and closest neighbours
        eucd, i = self.KDS.query(np.vstack((points[0],
                                            points[1],
                                            points[2])).T,
                                 distance_upper_bound=maxdist_euc)
        return eucd, i

    def query_bin_tree(self, latitude, longitude, binrad):
        """Uses the query function of the KDTree but takes geolocations as
        input.

        Parameters
        ----------
        latitude: latitudes to be queried
        longitude: longitudes to be queried
        binrad: maximum  distance to be queried in epicentral distance
                Equals the radius of the bin, higher = more averaging ->
                --> less noise, lower resolution


        Returns
        -------
        eucd : 1Dnp.array
            Euclidian distances to next point

        i: 1D np.array
            Indices of next neighbour.
        """

        # Compute maxdist in euclidean space
        maxdist_euc = epi2euc(binrad)

        # Get cartesian coordinate points
        points = geo2cart(R_EARTH, latitude, longitude)

        # Query the points and return the distances and closest neighbours
        # Eps: approximate closest neighbour (big boost in computation time)
        # Returned distance is not more than (1+eps)*real distance
        eucd, i = self.KDB.query(np.vstack((points[0],
                                            points[1],
                                            points[2])).T, eps=0.05, k=4,
                                 distance_upper_bound=maxdist_euc)
        return eucd, i
