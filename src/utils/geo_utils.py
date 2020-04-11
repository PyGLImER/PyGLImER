"""

Geographical utilities.

Author:
    Lucas Sawade (lsawade@princeton.edu)


Last Update: November 2019


"""

import numpy as np
from ..constants import R_EARTH


def reckon(lat, lon, distance, bearing):
    """ Computes new latitude and longitude from bearing and distance.

    Parameters
    ----------
    lat: in degrees
    lon: in degrees
    bearing: in degrees
    distance: in degrees

    Returns
    -------
    lat, lon


    lat1 = math.radians(52.20472)  # Current lat point converted to radians
    lon1 = math.radians(0.14056)  # Current long point converted to radians
    bearing = np.pi/2 # 90 degrees
    # lat2  52.20444 - the lat result I'm hoping for
    # lon2  0.36056 - the long result I'm hoping for.

    """

    # Convert degrees to radians for numpy
    lat1 = lat/180*np.pi
    lon1 = lon/180 * np.pi
    brng = bearing/180*np.pi
    d = distance/180*np.pi

    # Compute latitude
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d)
                     + np.cos(lat1) * np.sin(d) * np.cos(brng))

    # Compute longitude
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))

    # Convert back
    lat2 = lat2/np.pi*180
    lon2 = lon2/np.pi*180

    return lat2, lon2


def geo2cart(r, latitude, longitude):
    """Computes cartesian coordinates from geographical coordinates

    Parameters
    ----------
    r (float or `numpy.array` corresponding to lat/lon arrays): Radius of the
    location
    latitude (`numpy.array`): Latitude
    longitude (`numpy.array`): Longitude

    Returns
    -------
    xyz coordinates coordinates of same size as latitude and longitude arrays


    """

    # Convert to radians
    latrad = latitude/180*np.pi
    lonrad = longitude / 180 * np.pi

    # Convert to geolocations
    x = r * np.cos(latrad) * np.cos(lonrad)
    y = r * np.cos(latrad) * np.sin(lonrad)
    z = r * np.sin(latrad)

    return x, y, z


def cart2geo(x, y, z):
    """Computes geographical coordinates from cartesian coordinates

    Parameters
    ----------
    x (`numpy.array` corresponding to y/z arrays): x coordinate
    y (`numpy.array`): y coordinate
    z (`numpy.array`):  z coordinate

    Returns
    -------
    r, latitude, longitude tuple


    """

    # Compute r
    r = np.sqrt(x**2 + y**2 + z**2)

    # Compute theta using r
    latitude = 180/np.pi * np.arctan2(z, np.sqrt(x**2 + y**2))

    # Compute phi using atan
    longitude = np.arctan2(y, x) * 180/np.pi

    return r, latitude, longitude


def epi2euc(epi):
    """Converts epicentral distance in to a euclidean distance along the
    corresponding chord."""
    return 2 * R_EARTH * np.sin(np.pi * epi / (360))


def euc2epi(euc):
    """Converts euclidean distance to epicentral distance"""
    return 360 * np.arcsin(euc/(2*R_EARTH)) / np.pi
