#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a 3D velocity model using Litho1.0

Created on Fri May  1 12:11:03 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""
import subprocess
import pickle
import os
import fnmatch

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from obspy.geodetics import gps2dist_azimuth
from pathlib import Path

from ..constants import R_EARTH, maxz, res, DEG2KM
from ..utils.geo_utils import geo2cart, cart2geo

# location of lith1 file
lith1 = os.path.join('/home', 'pm', 'LITHO1.0', 'bin', 'access_litho')

#  Location of the GyPSuM textfiles
gyps = os.path.join('data', 'velocity_models', 'GyPSuM')

_MODEL_CACHE = {}


def load_gyps(save=True, latb=None, lonb=None):
    """
    Compiles the GyPSuM 3D-velocity object from included GyPSuM text files

    Parameters
    ----------
    save : Bool, optional
        Pickle the 3D velocity model after compiling it for the first time.
        This will allow for faster access to the model. Saving the model takes
        about 800 MB disk space.
        The default is True.
    latb : Tuple, optional
        Creates a submodel from the full model. In form (minlat, maxlat).
    lonb : Tuple, optional
        (minlon, maxlon)

    Returns
    -------
    ComplexModel object
        Object that can be queried for velocities.

    """
    if latb and not lonb or lonb and not latb:
        raise ValueError(
            """"Provide either no geographic boundaries or both latitude
            and longitude boundaries.""")

    if latb:
        # Changes the boundaries to ints (mainly for filenames)
        latb = (int(np.floor(latb[0])), int(np.ceil(latb[1])))
        lonb = (int(np.floor(lonb[0])), int(np.ceil(lonb[1])))

        try:
            return _MODEL_CACHE['gyps' + str(latb) + str(lonb)]
        except KeyError:
            pass
        try:
            with open('tmp/'+str(latb)+str(lonb)+'.pkl', 'rb') as infile:
                model = pickle.load(infile)

            _MODEL_CACHE['gyps' + str(latb) + str(lonb)] = model
            return model
        except FileNotFoundError:
            pass

    try:
        model = _MODEL_CACHE['gyps']
        if latb:
            _MODEL_CACHE['gyps' + str(latb) + str(lonb)] = model = \
                    model.submodel(latb, lonb)
            model.write(filename=str(latb)+str(lonb), folder='tmp')
        return model
    except KeyError:
        pass

    try:
        with open('data/velocity_models/gypsum.pkl', 'rb') as infile:
            model = pickle.load(infile)
        if not latb:
            _MODEL_CACHE['gyps'] = model

        else:
            _MODEL_CACHE['gyps' + str(latb) + str(lonb)] = model = \
                model.submodel(latb, lonb)
            model.write(filename=str(latb)+str(lonb), folder='tmp')
            return model

    except FileNotFoundError:
        pass

    # Create initial, full model
    # Create the velocity deviation grids
    vpd, vsd, _ = np.mgrid[-90:91, -180:181, 0:18]
    vpd = vpd.astype(float)
    vsd = vsd.astype(float)

    # Load background model
    rp, vpb = zip(*np.loadtxt(os.path.join(gyps, 'StartingVpModel.txt')))
    rs, vsb = zip(*np.loadtxt(os.path.join(gyps, 'StartingVsModel.txt')))

    zbp = R_EARTH - np.array(rp, dtype=float)  # background model depth vector
    zbs = R_EARTH - np.array(rs, dtype=float)  # background model depth vecto
    vpb = np.array(vpb)
    vsb = np.array(vsb)

    del rp, rs

    # Load deviations
    dirlist = os.listdir(gyps)

    # vp deviations
    for i, p in enumerate(fnmatch.filter(dirlist, 'P.*')):
        vpd[:, :, 2*i] = np.reshape(
            np.loadtxt(os.path.join(gyps, p)), vpd[:, :, 0].shape) / 100
        vpd[:, :, 2*i + 1] = np.reshape(
            np.loadtxt(os.path.join(gyps, p)), vpd[:, :, 0].shape) / 100

    # vs deviations
    for i, p in enumerate(fnmatch.filter(dirlist, 'S.*')):
        vsd[:, :, 2*i] = np.reshape(
            np.loadtxt(os.path.join(gyps, p)), vpd[:, :, i].shape) / 100
        vsd[:, :, 2*i + 1] = np.reshape(
            np.loadtxt(os.path.join(gyps, p)), vpd[:, :, i].shape) / 100

    # boundaries for the velocity deviations vectors
    zd = np.hstack(
        (0, np.repeat(np.hstack((np.arange(100, 475, 75),
                                 np.array([525, 650, 750]))), 2),
         850))

    # Interpolation depth
    # zq = np.unique(np.sort(np.hstack(zb, zd)))
    # imax = np.where(zq > 850)
    # zq = zq[:imax]
    zq = np.arange(0, maxz+res, res)

    # Interpolate background velocity model
    vp_bg = np.interp(zq, zbp, vpb)
    vs_bg = np.interp(zq, zbs, vsb)

    del vpb, vsb, zbp, zbs

    # Interpolate velocity disturbances
    intf = interp1d(zd, vpd, axis=2)
    dvp = intf(zq)
    intf = interp1d(zd, vsd, axis=2)
    dvs = intf(zq)

    vp = np.multiply(dvp, vp_bg) + vp_bg
    vs = np.multiply(dvs, vs_bg) + vs_bg

    del vpd, vsd, intf, dvp, dvs

    lat = np.arange(-90, 91, 1)
    lon = np.arange(-180, 181, 1)

    # Create a velocity model with 1km spacing
    model = ComplexModel(zq, vp, vs, lat, lon)

    # Pickle model
    if save:
        model.write()

    if not latb:
        _MODEL_CACHE['gyps'] = model
    else:
        _MODEL_CACHE['gyps' + str(latb) + str(lonb)] = model = \
            model.submodel(latb, lonb)
        model.write(filename=str(latb)+str(lonb), folder='tmp')

    return model


def raysum3D(dip):
    """
    Compiles the provided Raysum test model (3D). Fairly clumpsy solved,
    which makes it quite slow and RAM intense. An interpolated model instead
    of a gridded model would probably be faster and more accurate. However,
    this one is closer to the "real 3D model".

    Parameters
    ----------
    dip : int
        Dip of the LAB in degree. Options are 10, 20, 30, 40 deg.
        Rest of the model is set.

    Returns
    -------
    model : ComplexModel
        Complexmodel object that can be queried.

    """

    try:
        return _MODEL_CACHE['raysum'+str(dip)]
    except KeyError:
        pass

    z = np.arange(0, maxz+res, res)  # depth in km
    lat = np.arange(-5, 5.01, 0.05)
    lon = np.arange(-5, 5.01, 0.05)

    # Populate velocity models
    vp = np.empty((len(lat), len(lon), len(z)))
    vs = np.empty((len(lat), len(lon), len(z)))

    # above Moho
    im = np.where(z == 20)[0][0]
    vp[:, :, :im].fill(3.54)
    vs[:, :, :im].fill(1.41)
    vp[:, :, im:].fill(7.7)
    vs[:, :, im:].fill(4.2)

    # calculate position of dipping layer
    d0 = 150  # depth at origin (0,0)
    a = np.tan(np.deg2rad(dip))

    for m, latitude in enumerate(lat):
        x = DEG2KM*latitude
        for n, longitude in enumerate(lon):
            y, _, _ = gps2dist_azimuth(latitude, 0, latitude, longitude)
            y = y/1000
            if longitude < 0:
                y = -y
                # Depth depending on x, y
                d = int(round(a*np.dot([1, -1], (x, y))/np.sqrt(2) + d0))
                vp[m, n, d:] = 6.3
                vs[m, n, d:] = 3
    _MODEL_CACHE['raysum'+str(dip)] = \
        model = ComplexModel(z, vp, vs, lat, lon, flatten=False, zf=z)

    return model


class ComplexModel(object):
    def __init__(self, z, vp, vs, lat, lon, flatten=True, zf=None):
        """
        Velocity model based on GyPSuM model. Compiled and loaded with function
        load_gyps(). The model will be compiled once into a relatively large
        pickle file. Afterwards, the file will be unpickled rather than
        recompiled. Self.vpf, self.vsf, and self.zf are contain values for
        an Earth-flattening approximation.

        Parameters
        ----------
        z : 1D ndarray
            Contains depths in km.
        vp : 3D ndarray
            P-wave velocity grid with velocities in km/s. Spherical
        vs : 3D ndarray
            S-wave velocity grid with velocities in km/s. Spherical
        lat : 1D ndarray
            Latitude vector.
        lon : 1D ndarray
            Longitude vector.
        flatten : Bool - optional
            Apply Earth flattening. Should be False for submodels. In that
            case the values for vp and vs are already flattened.
        zf : 1d ndarray
            flattened depth vector. Only needed if flatten=False.

        Returns
        -------
        None.

        """
        self.lat = lat
        self.lon = lon
        grid = np.meshgrid(self.lat, self.lon)
        self.coords = (grid[0].ravel(), grid[1].ravel())
        del grid

        xs, ys, zs = geo2cart(R_EARTH, self.coords[0],
                              self.coords[1])

        self.tree = KDTree(list(zip(xs, ys, zs)))
        del xs, ys, zs

        self.z = z
        # self.ndec = len(str(lat[1]-lat[0]))-1

        if flatten:
            self.vpf, self.vsf, self.zf = self.flatten(vp, vs)
        else:
            # if not zf:
            #     raise ValueError("""If flatten=False, a zf vector has to be
            #                      provided""")
            self.vpf = vp
            self.vsf = vs
            self.zf = zf

    def query(self, lat, lon, z):
        """
        Query the 3D velocity model.

        Parameters
        ----------
        lat : float/int
            Latitude.
        lon : float/int
            Longitude.
        z : float/int
            depth.

        Returns
        -------
        vp : float
            P-wave velocity.
        vs : float
            S-wave velocity.

        """
        # m = np.where(round(lat, self.ndec) == self.lat)[0][0]
        # n = np.where(round(lon, self.ndec) == self.lon)[0][0]
        # m = np.nonzero(np.isclose(round(lat, self.ndec), self.lat))[0][0]
        # n = np.nonzero(np.isclose(round(lon, self.ndec), self.lon))[0][0]
        xs, ys, zs = geo2cart(R_EARTH, lat, lon)
        d, i = self.tree.query([xs, ys, zs])

        if d > (self.lat[1]-self.lat[0]) * DEG2KM * 1.25:
            raise ValueError(""" The chosen velocity model does not cover
                             the queried area.""")

        m = np.where(self.lat == self.coords[0][i])[0][0]
        n = np.where(self.lon == self.coords[1][i])[0][0]
        p = np.where(round(z) == self.z)[0][0]

        vp = self.vpf[m, n, p]
        vs = self.vsf[m, n, p]
        return vp, vs

    def write(self, filename='gypsum', folder='data/velocity_models'):
        """
        Save the model.

        Parameters
        ----------
        filename : str, optional
            Filename. The default is 'avvmodel'.
        folder : str, optional
            Direcotry to save the model. The default is 'data/velocity_models'.

        Returns
        -------
        None.

        """
        # Remove filetype identifier if provided
        x = filename.split('.')
        if len(x) > 1:
            if x[-1] == 'pkl':
                filename = ''.join(x[:-1])
        oloc = os.path.join(folder, filename)
        with open(oloc + ".pkl", 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def flatten(self, vp, vs):
        """
        Creates a flat-earth approximated velocity model down to maxz as in
        Peter M. Shearer

        Parameters
        ----------
        vp : np.array (3D)
            P wave velocities [km/s].
        vs : np.array (3D)
            S wave velocities [km/s].

        Returns
        -------
        zf : np.array
            Depths in a Cartesian coordinate system.
        vpf : np.array
            P wave velocities in a Cartesian coordinate system.
        vsf : np.array
            S wave velocities in a Cartesian coordinate system.
        """

        r = R_EARTH  # Earth's radius
        vpf = np.multiply((r/(r-self.z)), vp)
        vsf = np.multiply((r/(r-self.z)), vs)
        zf = -r*np.log((r-self.z)/r)

        return vpf, vsf, zf

    def submodel(self, lat, lon):
        """
        Creates a submodel from the current velocity model within the
        defined geographic boundaries. Can save a lot of RAM.

        Parameters
        ----------
        lat : tuple
            Tuple in the form (minimum latitude, maximum latitude).
        lon : tuple
            Tuple in the form (minimum longitude, maximum longitude).
            Note that this can cause troubles if the defined area is
            between >-180 and <0 (which should usually not be the case).

        Returns
        -------
        subm : ComplexModel
            ComplexModel daughter object.

        """
        # latv = np.arange(np.floor(lat[0]), np.ceil(lat[1])+1, 1)
        # lonv = np.arange(np.floor(lon[0]), np.ceil(lon[1])+1, 1)

        m0 = np.where(self.lat == np.floor(lat[0]))[0][0]
        m1 = np.where(self.lat == np.ceil(lat[1]))[0][0] + 1
        n0 = np.where(self.lon == np.floor(lon[0]))[0][0]
        n1 = np.where(self.lon == np.ceil(lon[1]))[0][0] + 1

        # Define new model
        subm = ComplexModel(
            self.z, self.vpf[m0:m1, n0:n1, :], self.vsf[m0:m1, n0:n1, :],
            self.lat[m0:m1], self.lon[n0:n1], flatten=False, zf=self.zf)

        return subm


def load_avvmodel():
    """
    Creates a model over the average P and S-wave velocities in the upper
    crust. These are used by the P-SV-SH rotation algorithm. Model data
    is extracted from the Litho1.0 model (location provided above).
    The package is distributed with a readily compiled model.

    Litho1.0 must be installed and location must be set correct for a
    complete compilation! However, function will first look in RAM and then
    in data for a pickle file.

    Compiling takes up to one hour!

    Returns
    -------
    Model containing average velocities for upper crust.

    """
    try:
        return _MODEL_CACHE['avv']
    except KeyError:
        pass

    try:
        with open('data/velocity_models/avvmodel.pkl', 'rb') as infile:
            _MODEL_CACHE['avv'] = model = pickle.load(infile)
            return model
    except FileNotFoundError:
        pass

    # latitude and longitude vector
    latv = np.arange(-90, 91)
    lonv = np.arange(-180, 181)
    # self.depth = np.arange(-10, 801)

    # Create grid, spacing .5 deg, 1km
    # self.vp, self.vs, _ = np.mgrid[-180:181, -360:361, -10:801]

    # Grid of average P and S-wave velocities, used for P-SV-SH rotation
    # self.avpS, self.avsS = np.mgrid[-90:90.5:.5, -180:180.5:.5]
    # self.avpP, self.avsP = np.mgrid[-90:90.5:.5, -180:180.5:.5]
    avpS, avsS = np.mgrid[-90:91, -180:181]
    avpP, avsP = np.mgrid[-90:91, -180:181]

    # populate velocity grid
    for m, lat in enumerate(latv):
        for n, lon in enumerate(lonv):
            # Call Litho1.0
            try:
                x = subprocess.Popen(
                    [lith1, "-p", str(lat), str(lon)],
                    stdout=subprocess.PIPE)
                ls = str(x.stdout.read()).split("\\n")  # save the output

                # Close file or it will remain open forever!
                x.stdout.close()

                for ii, item in enumerate(ls):
                    ls[ii] = item.split()

                # clean list
                del ls[-1]
                del ls[0][0]

            except IndexError:
                # There are some points, on which the model is not defined
                lat = lat + .1
                x = subprocess.Popen(
                    [lith1, "-p", str(lat), str(lon)],
                    stdout=subprocess.PIPE)
                ls = str(x.stdout.read()).split("\\n")  # save the output

                # Close file or it will remain open forever!
                x.stdout.close()

                for ii, item in enumerate(ls):
                    ls[ii] = item.split()

                # clean list
                del ls[-1]
                del ls[0][0]

                pass

            # reorder items
            depth = []
            vp = []
            vs = []
            name = []

            for item in ls:
                depth.append(float(item[0]))  # in m
                vp.append(float(item[2]))  # m/s
                vs.append(float(item[3]))  # m/
                name.append(item[-1])  # name of the boundary

            # Interpolate and populate
            vp = np.interp(
                np.arange(min(depth), 15.5e3 + min(depth), .5e3),
                np.flip(depth), np.flip(vp))
            vs = np.interp(
                np.arange(min(depth), 15.5e3 + min(depth), .5e3),
                np.flip(depth), np.flip(vs))

            # build weighted average for upper ~15km (S-phases)
            # and the upper 6 km (P phases)
            avpS[m, n] = np.average(vp)
            avsS[m, n] = np.average(vs)

            # For P-wave as primary phase (higher frequencies and shorter
            # wavelength)
            avpP[m, n] = np.average(vp[:-18])
            avsP[m, n] = np.average(vs[:-18])

    _MODEL_CACHE['avv'] = model = AverageVelModel(
        latv, lonv, avpP, avsP, avpS, avsS)

    # Dump pickle file
    model.write()

    return model


class AverageVelModel(object):
    def __init__(self, lat, lon, avpP, avsP, avpS, avsS):
        """
        Creates a model over the average P and S-wave velocities in the upper
        crust. These are used by the P-SV-SH rotation algorithm. Model data
        is extracted from the Litho1.0 model (location provided above).
        The package is distributed with a readily compiled model.

        Compiling takes up to one hour!

        Returns
        -------
        None.

        """
        # latitude and longitude vector
        self.latv = lat
        self.lonv = lon

        self.avpP = avpP
        self.avsP = avsP
        self.avpS = avpS
        self.avsS = avsS

    def query(self, lat, lon, phase):
        """
        Query average P- and S-Wave velocity in the upper 15 km (phase=S) or
        6 km (phase=P)

        Parameters
        ----------
        lat : float
            Latitude.
        lon : TYPE
            Longitude.
        phase : str
            Primary phase (has to be provided due to different frequency
                           content)

        Returns
        -------
        avp : float
            Average P-wave veocity in m/s.
        avs : float
            Average S-wave velocity in m/s.

        """
        # lat = roundhalf(lat)
        # lon = roundhalf(lon)

        lat = round(lat)
        lon = round(lon)

        m = np.where(self.latv == lat)[0][0]
        n = np.where(self.lonv == lon)[0][0]

        if phase == 'P':
            avp = self.avpP[m, n]
            avs = self.avsP[m, n]
        elif phase == 'S':
            avp = self.avpS[m, n]
            avs = self.avsS[m, n]

        return avp, avs

    def write(self, filename='avvmodel', folder='data/velocity_models'):
        """
        Save the model.

        Parameters
        ----------
        filename : str, optional
            Filename. The default is 'avvmodel'.
        folder : str, optional
            Direcotry to save the model. The default is 'data/velocity_models'.

        Returns
        -------
        None.

        """
        # Remove filetype identifier if provided
        x = filename.split('.')
        if len(x) > 1:
            filename = filename + '.' - x[-1]
        oloc = os.path.join(folder, filename)

        if not Path(oloc).is_dir:
            subprocess.call(['mkdir', '-p', oloc])

        with open(oloc + ".pkl", 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
