'''
Module that covers the binning part of the CCP computation.
Authors: Peter Makus (peter.makus@student.uib.no)
        and
        Lucas Sawade (lsawade@princeton.edu)

Created: November 2019
Last Modified: Wednesday, 22nd July 2020 11:45:19 am
'''

import logging
import time

import numpy as np
from scipy.spatial import KDTree

from ...constants import R_EARTH
# Local imports
# from ... import logger
from ...utils.geo_utils import geo2cart, cart2geo, epi2euc
from ...utils.utils import dt_string

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

    def __init__(self, latitude, longitude, edist, phase,
                 verbose=True):
        """
        BinGrid object that can be used to find closest neighbours to
        stations and CCP bins

        :param latitude: Latitudes
        :type latitude: 1-D `numpy.array`
        :param longitude: Longitudes
        :type longitude: 1-D `numpy.array`
        :param edist: Distance (in degree) between bin centres.
        :type edist: float
        :param phase: Seismic phase either "S" or "P". Phase "S" will lead to
            more grid points being created due to flatter incidence angle.
            Hence, phase can be "S" for PRFs but not the other way around.
            However, "P" is computationally more efficient.
        :type phase: str
        :param verbose: consoel output?, defaults to True
        :type verbose: bool, optional
        """        

        # Populate stuff
        self.stations = np.unique(np.stack((latitude, longitude)).T, axis=0)
        self.latitude = self.stations[:, 0]
        self.longitude = self.stations[:, 1]
        self.edist = edist
        self.phase = phase.upper()

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
        if self.phase[-1] == "S":
            maxepid = 12
        elif self.phase[-1] == "P":
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
        """
        Uses the query function of the KDTree but takes geolocations as
        input.

        :param latitude: latitudes to be queried
        :type latitude: 1D `numpy.ndarray`
        :param longitude: longitudes to be queried
        :type longitude: 1D `numpy.ndarray`
        :param maxdist: maximum  distance to be queried in degree
        :type maxdist: float
        :return: Euclidian distances to next point, indices of next neighbour.
        :rtype: Tuple with 2 1D `numpy.ndarray`
        """        

        # Compute maxdist in euclidean space
        maxdist_euc = epi2euc(maxdist)

        # Get cartesian coordinate points
        points = geo2cart(R_EARTH, latitude, longitude)

        # Query the points and return the distances and closest neighbours
        eucd, i = self.KDS.query(np.column_stack((points[0],
                                                  points[1],
                                                  points[2])),
                                 distance_upper_bound=maxdist_euc)
        return eucd, i

    def query_bin_tree(self, latitude, longitude, binrad, k):
        """
        Uses the query function of the KDTree but takes geolocations as
        input.

        :param latitude: latitudes to be queried
        :type latitude: 1-D `numpy.ndarray`
        :param longitude: longitudes to be queried
        :type longitude: 1-D `numpy.ndarray`
        :param binrad: maximum  distance to be queried in euclidean distance
                Equals the radius of the bin, higher = more averaging ->
                --> less noise, lower resolution
        :type binrad: float
        :param k: Number of neighbours to return. Note that this directly
            depends on the ratio of deltabin/binrad.
        :type k: int
        :return: Indices of next neighbour.
        :rtype: 1D np.array
        """

        # Get cartesian coordinate points
        points = geo2cart(R_EARTH, latitude, longitude)

        # Create tree to be KDTree queried against

        # Query the points and return the distances and closest neighbours
        # Eps: approximate closest neighbour (big boost in computation time)
        # Returned distance is not more than (1+eps)*real distance
        _, i = self.KDB.query(np.column_stack((points[0],
                                               points[1],
                                               points[2])), eps=0.05, k=k,
                              distance_upper_bound=binrad)
        return i
