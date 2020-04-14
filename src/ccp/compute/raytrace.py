"""

Module that covers the raytracing part of the CCP computation.

Author:
    Lucas Sawade (lsawade@princeton.edu)


Last Update: November 2019

"""

import multiprocessing
import time
# Filter raytrace warning due tue turning point reach of shallow rays.
import warnings

from joblib import delayed
# enabling multiprocessing.
from joblib import Parallel
import numpy as np

# Import the logger
from ... import logger
from ...constants import R_EARTH
# Local imports
from ...geo_utils import reckon
from ...utils import dt_string

warnings.filterwarnings("ignore",
                        message="invalid value encountered in sqrt")


class Raytracing(object):

    def __init__(self, rayp, baz, lat, lon, elev,
                 vlat, vlon, vdep, vp, vs,
                 mp=True, verbose=True):
        """

        Parameters
        ----------
        rayp (`numpy.array` of size (M,)): vector containing the rayparameters
        baz (`numpy.array` of size (M,)): vector containing the back-azimuths
        lat (`numpy.array` of size (M,)): vector containing the station
                                          latitudes
        lon (`numpy.array` of size (M,)): vector containing the station
                                          longitudes
        elev (`numpy.array` of size (M,)): vector containing the elevations
        vlat (`numpy.array` of size (R)): Vector containing the latitudes of
                                          the velocity model
        vlon (`numpy.array` of size (S)): Vector containing the longitudes of
                                          the velocity model
        vdep (`numpy.array` of size (T)): Vector containing the depth of the
                                          velocity model
        vp (`numpy.array` of size (R,S,T)): 3D-matrix containing the vp
                                            velocities
        vs (`numpy.array` of size (R,S,T)): 3D-matrix containing the vs
                                            velocities
        mp (boolean): Multiprocessing. Decides whether rays are computed in
                      parallel or not. Default is True.

        Returns
        -------
        clat (`numpy.array` of size (M, T)): matrix containing the latitude
                                            values of the conversion points
                                            corresponding to RF and depth
                                            vector.
        clat (`numpy.array` of size (M, T)): matrix containing the longitude
                                            values of the conversion points
                                            corresponding to RF and depth
                                            vector.
        dtimes (`numpy.array` of size (M, T)): RF traveltimes to the conversion
                                               points corresponding to RF and
                                               depth vector.
        """

        # Allocate stuff
        # RF params
        if type(rayp) == float:
            self.rayp = [rayp]
            self.baz = [baz]
            self.lat = [lat]
            self.lon = [lon]
            self.elev = [elev / 1000]
            self.M = 1
        else:
            self.rayp = rayp
            self.baz = baz
            self.lat = lat
            self.lon = lon
            self.elev = elev / 1000
            self.M = np.shape(rayp)[0]

        # Velocity model parameters
        self.vlat = vlat
        self.vlon = vlon
        self.vdep = vdep
        self.vp = vp
        self.vs = vs
        self.N = vdep.shape[0] - 1

        # Multiprocessing
        self.mp = mp

        # Verbosity
        self.verbose = verbose

    def __call__(self):

        if self.verbose:
            logger.info("--- Raytracing ---")
            logger.info(" ")
            start = time.time()

        if not self.mp:

            # Preallocate stuff
            self.clat = np.zeros((self.M, self.N))
            self.clon = np.zeros((self.M, self.N))
            self.d = np.zeros((self.M, self.N))
            self.dtimes = np.zeros((self.M, self.N + 1, 3))

            for _i, (r, b, la, lo, el) in enumerate(zip(self.rayp, self.baz,
                                                        self.lat, self.lon,
                                                        self.elev)):
                # print("Tracing RF: %7d/%d" % (_i, self.M))
                self.clat[_i, :], self.clon[_i, :], \
                self.d[_i, :], self.dtimes[_i, :, :] = \
                    self.trace(_i, r, b, la, lo, el)

        else:
            # Set the number of cores
            num_cores = multiprocessing.cpu_count()
            logger.info("   Number of cores used: %d" % num_cores)
            logger.info("   ")

            # Embarassingly parallel line of code to run the raytracing in
            # parallel
            results = Parallel(n_jobs=num_cores)(
                delayed(self.trace)(_i, r, b, la, lo, el)
                for _i, (r, b, la, lo, el) in enumerate(zip(self.rayp,
                                                            self.baz,
                                                            self.lat,
                                                            self.lon,
                                                            self.elev)))

            # Get results
            self.clat = np.vstack(tuple((row[0] for row in results)))
            self.clon = np.vstack(tuple((row[1] for row in results)))
            self.d = np.vstack(tuple((row[2] for row in results)))
            self.dtimes = np.stack(tuple((row[3] for row in results)))

        if self.verbose:
            end = time.time()
            dt = end - start

            logger.info("   ")
            logger.info("   Finished raytracing.")
            logger.info(dt_string(dt))
            logger.info(" ")
            logger.info(" ")

        return self.clat, self.clon, self.d, self.dtimes

    def save_tracefile(self, filename, rf):
        """ Save RF File with traced rays.

        Parameters
        ----------
        filename (str): String with filename to save the Traced file
        rfs: actual receiver functions.

        Returns
        -------
        None

        """
        if self.verbose:
            logger.info("--- Saving Raytraced file ---")
            start = time.time()

        np.savez(filename, rf=rf, lat=self.lat, lon=self.lon, baz=self.baz,
                 rayp=self.rayp, elev=self.elev, clat=self.clat, clon=self.clon,
                 depth=self.vdep, dtimes=self.dtimes)

        if self.verbose:
            end = time.time()
            logger.info("   Finished saving.")
            logger.info(dt_string(end - start))
            logger.info(" ")
            logger.info(" ")

    def trace(self, _i, rayp, baz, lat, lon, elev):

        # Print after 10% of rays have been computed
        if self.verbose:
            if np.mod(_i + 1, np.round(self.M / 20)) == 0:
                logger.info("   Finished  ~ %3f%%." % (100 * _i / self.M))

        # Earth's radius in km
        re = R_EARTH - np.min(self.vdep)

        # minimum S - velocity
        minvs = 3.
        minvp = 5.5

        # adjust for coordinate ssystem difference on z axis, by the max
        # elevation of the velocity model!
        dfix = np.abs(np.min(self.vdep))

        # depth in each layer:
        d = np.diff(self.vdep)

        # Earth flattening tranformation, equation from Peter M.Shearer's
        # Introduction to Seismology(1999), p.49
        zf = -re * np.log((re - self.vdep) / re)
        df = np.diff(zf)

        # number of layers:
        m = d.shape[0]

        # epicentral distance between previous and new conversion point
        ep1 = np.zeros(m)
        # updated
        ep2 = np.zeros(m)

        # traveltimes in each layer for p and s wave velocity
        dtimes_p = np.zeros(m)
        dtimes_s = np.zeros(m)

        # outputs s - wave ccp locations:
        clat1 = np.zeros(m)
        clon1 = np.zeros(m)
        # updated:
        clat2 = np.zeros(m)
        clon2 = np.zeros(m)

        # velocities:
        vp1 = np.zeros(m)
        vs1 = np.zeros(m)
        # updated
        vp2 = np.zeros(m)
        vs2 = np.zeros(m)

        # earth flattening velocities:
        vp2f = np.zeros(m)
        vs2f = np.zeros(m)

        # velocity location
        velat1 = np.zeros(m, dtype=int)
        velon1 = np.zeros(m, dtype=int)
        # updated
        velat2 = np.zeros(m, dtype=int)
        velon2 = np.zeros(m, dtype=int)

        # Calculation:

        # Finding the depth consistent with elevation of station. Make sure max
        # elevation is smaller than max elevation in velocity model. The code will
        # not work otherwise.
        depth_pos = np.where(self.vdep <= elev)[0][-1]

        # for equation rayp needs to be:
        # s / radian = s / km * 111.11[km / deg] * 180 / pi[deg / rad]:
        rayp_km = rayp
        rayp = rayp / np.pi * 180 * 40000 / 360

        for k in range(m):

            # Checking whether the elevation bin is reached.

            if k >= depth_pos:

                # Find velocity in grid and depth
                if k == depth_pos:
                    velat1[k] = np.argmin(np.abs(self.vlat - lat))
                    velon1[k] = np.argmin(np.abs(self.vlon - lon))
                else:
                    velat1[k] = np.argmin(np.abs(self.vlat - clat2[k - 1]))
                    velon1[k] = np.argmin(np.abs(self.vlon - clon2[k - 1]))

                # p and s layer velocity:
                vp1[k] = self.vp[velon1[k], velat1[k], k]
                vs1[k] = self.vs[velon1[k], velat1[k], k]

                # Fix for water velocities - especially in coast locations
                if vs1[k] < minvs:
                    nonzero_lon, nonzero_lat = np.where(
                        self.vs[:, :, k] >= minvs)

                    closestnz_ind = np.argmin(np.abs(np.sqrt(
                        (nonzero_lon - velon1[k]) ** 2
                        + (nonzero_lat - velat1[k]) ** 2)))

                    vp1[k] = self.vp[nonzero_lon[closestnz_ind],
                                     nonzero_lat[closestnz_ind], k]
                    vs1[k] = self.vs[nonzero_lon[closestnz_ind],
                                     nonzero_lat[closestnz_ind], k]

                # Fix for peaks being higher than the first entry in the
                # velocity model.
                elif np.isnan(vs1[k]) or np.isnan(vp1[k]):
                    # vs must be larger than minvs as well to ensure correct
                    # ray path and escape water issue

                    nonzero_vs = np.where(self.vs[velon1[k], velat1[k],
                                          :] >= minvs)[0][0]
                    vs1[k] = self.vs[velon1[k], velat1[k], nonzero_vs]

                    # vp must be larger than minvp as well to ensure correct ray
                    # path
                    nonzero_vp = np.where(self.vp[velon1[k], velat1[k],
                                          :] >= minvp)[0][0]
                    vp1[k] = self.vs[velon1[k], velat1[k], nonzero_vp]

                # epicentral distance, equation from Peter M.Shearer's Introduction to
                # Seismology(1999), p. 48:
                if k == depth_pos:
                    ep1[k] = rayp * (self.vdep[k + 1] - elev) \
                             / (((re - (elev + dfix)
                                  - (self.vdep[k + 1] - elev) / 2) / vs1[k]) ** 2
                                - rayp ** 2) ** (1 / 2) \
                             / (re - (elev + dfix) - (self.vdep[k + 1] - elev) / 2)
                else:
                    ep1[k] = rayp * d[k] \
                             / (((re - (self.vdep[k] + dfix) -
                                  d[k] / 2) / vs1[k]) ** 2 - rayp ** 2) ** (1 / 2) \
                             / (re - (self.vdep[k] + dfix) - d[k] / 2)

                # calculate location of conversion point with epicentral distance and
                # backazimuth:
                if k == depth_pos:
                    clat1[k], clon1[k] = reckon(lat, lon, ep1[k] / np.pi * 180,
                                                baz)
                else:
                    clat1[k], clon1[k] = reckon(clat2[k - 1], clon2[k - 1],
                                                ep1[k] / np.pi * 180, baz)

                # Updating velocities, since the conversion point could be in different
                # velocity gridpoint than the starting point:

                # Update of velocity if necessary due to possible velocity change.
                velat2[k] = np.argmin(np.abs(self.vlat - clat1[k]))
                velon2[k] = np.argmin(np.abs(self.vlon - clon1[k]))

                # Fix for water velocities - especially in coast locations
                if self.vs[velon2[k], velat2[k], k] < minvs:

                    nonzero_lon, nonzero_lat = np.where(
                        self.vs[:, :, k] >= minvs)

                    closestnz_ind = np.argmin(np.abs(np.sqrt(
                        (nonzero_lon - velon1[k]) ** 2
                        + (nonzero_lat - velat1[k]) ** 2)))

                    velat2[k] = nonzero_lat[closestnz_ind]
                    velon2[k] = nonzero_lon[closestnz_ind]

                    kindvs = k
                    kindvp = k

                # Fix for peaks being higher than the first entry in the
                # velocity model.
                elif np.isnan(self.vs[velon2[k], velat2[k], k]) \
                        or np.isnan(self.vp[velon2[k], velat2[k], k]):
                    # vs must be larger than minvs as well to ensure correct
                    # ray path and escape water issue
                    kindvs = np.where(
                        self.vs[velon2[k], velat2[k], :] >= minvs)[0][0]

                    # vp must be larger than minvp as well to ensure correct
                    # ray path
                    kindvp = np.where(
                        self.vp[velon1[k], velat1[k], :] >= minvp)[0][0]

                else:
                    kindvs = k
                    kindvp = k

                # p and s wave layer velocity, average over traveltime in both
                # gridcells
                vp2[k] = (vp1[k] + self.vp[velon2[k], velat2[k], kindvp]) / 2
                vs2[k] = (vs1[k] + self.vs[velon2[k], velat2[k], kindvs]) / 2

                # epicentral distance, equation from Peter M.Shearer's Introduction
                # to Seismology(1999), p. 48:
                # 2 * rayp * delta_r / sqrt((r / v) ^ 2 - rayp ^ 2) / r
                if k == depth_pos:
                    ep2[k] = rayp * np.abs(self.vdep[k + 1] - elev) \
                             / (((re - (elev + dfix) - (self.vdep[k + 1] - elev) / 2)
                                 / vs2[k]) ** 2 - rayp ** 2) ** (1 / 2) \
                             / (re - (elev + dfix) - (self.vdep[k + 1] - elev) / 2)
                else:
                    ep2[k] = rayp * d[k] / (
                            ((re - (self.vdep[k] + dfix)
                              - d[k] / 2) / vs2[k]) ** 2
                            - rayp ** 2) ** (1 / 2) \
                             / (re - (self.vdep[k] + dfix) - d[k] / 2)

                # calculate location of conversion point with help of epicentral
                # distance and backazimuth:
                if k == depth_pos:
                    clat2[k], clon2[k] = reckon(lat, lon, ep2[k] / np.pi * 180,
                                                baz)
                else:
                    clat2[k], clon2[k] = reckon(clat2[k - 1], clon2[k - 1],
                                                ep2[k] / np.pi * 180, baz)

                # traveltimes with earth flattening model, equation from Peter
                # M.Shearer 's Introduction to Seismology (1999), p. 49 and
                # linear traveltime equation from Stephane Rondenay's review
                # paper on Receiver Functions and upper mantle imaging(2009):
                # Earth flattening transformation for velocities:

                if k == depth_pos:
                    vp2f[k] = (re / (re - (elev + dfix))) * vp2[k]
                    vs2f[k] = (re / (re - (elev + dfix))) * vs2[k]
                else:
                    vp2f[k] = (re / (re - (self.vdep[k] + dfix))) * vp2[k]
                    vs2f[k] = (re / (re - (self.vdep[k] + dfix))) * vs2[k]

                # linear traveltime, Rondenay(2009):
                if k == depth_pos:
                    # earth flattening
                    dfelev = - re * np.log((re - (self.vdep[k + 1]
                                                  + dfix)) / re) \
                             + re * np.log((re - (elev + dfix)) / re)
                    dtimes_p[k] = dfelev * np.sqrt(1 / vp2f[k] ** 2
                                                   - rayp_km ** 2)
                    dtimes_s[k] = dfelev * np.sqrt(1 / vs2f[k] ** 2
                                                   - rayp_km ** 2)
                else:
                    dtimes_p[k] = df[k] * np.sqrt(1 / vp2f[k] ** 2
                                                  - rayp_km ** 2)
                    dtimes_s[k] = df[k] * np.sqrt(1 / vs2f[k] ** 2
                                                  - rayp_km ** 2)

            else:
                dtimes_p[k] = np.nan
                dtimes_s[k] = np.nan
                clat1[k] = np.nan
                clon1[k] = np.nan
                clat2[k] = np.nan
                clon2[k] = np.nan

        # For loop stops

        # linear traveltime of the different modes, Rondenay(2009):
        difftimes_ps = dtimes_s - dtimes_p
        difftimes_ppps = dtimes_s + dtimes_p
        difftimes_ppss = 2 * dtimes_s

        dtimes_ps = np.zeros(difftimes_ps.shape[0] + 1)
        dtimes_ppps = np.zeros(difftimes_ppps.shape[0] + 1)
        dtimes_ppss = np.zeros(difftimes_ppss.shape[0] + 1)

        dtimes_ps[:depth_pos] = np.nan
        dtimes_ppps[:depth_pos] = np.nan
        dtimes_ppss[:depth_pos] = np.nan

        for k in range(depth_pos + 1, dtimes_ps.shape[0]):
            dtimes_ps[k] = np.sum(difftimes_ps[depth_pos:k - 1])
            dtimes_ppps[k] = np.sum(difftimes_ppps[depth_pos:k - 1])
            dtimes_ppss[k] = np.sum(difftimes_ppss[depth_pos:k - 1])

        dtimes = np.zeros((dtimes_ps.shape[0], 3))

        dtimes[:, 0] = dtimes_ps
        dtimes[:, 1] = dtimes_ppps
        dtimes[:, 2] = dtimes_ppss

        clat = clat2
        clon = clon2

        return clat, clon, d, dtimes
