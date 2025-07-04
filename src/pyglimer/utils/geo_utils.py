"""

Geographical utilities.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Lucas Sawade (lsawade@princeton.edu)


Last Update: November 2019


"""

from typing import Tuple
import numpy as np
from cartopy.geodesic import Geodesic
from ..constants import R_EARTH


def reckon(
    lat: float, lon: float, distance: float, bearing: float) -> Tuple[
        float, float]:
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


def geodiff(lat, lon):
    """Computes Azimuths and distances between geographical points."""

    # Create Geodesic class
    G = Geodesic(flattening=0.0)

    mat = np.asarray(
        G.inverse(
            np.array((lon[0:-1], lat[0:-1])).T,
            np.array((lon[1:], lat[1:])).T
        )
    )
    dists = mat[:, 0] / R_EARTH / 1000.0 / np.pi * 180
    az = mat[:, 1]

    return dists, az


def geodist(lat, lon):
    """Computes Azimuths and distances between geographical points."""

    # Compute distances
    dists, _ = geodiff(lat, lon)

    M = len(lat)
    cdists = np.zeros(M)
    cdists[1:] = np.cumsum(dists)

    return cdists


def gctrack(
        lat, lon, dist: float = 1.0, constantdist: bool = True) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
    """Given waypoints and a point distance, this function computes evenly
    spaced points along the great circle and the given waypoints.

    .. warning:: This function is not very accurate, but it is fast. You may
                 want to resort to more accurate functions if that is of
                 importance for your work. Here this function is only used for
                 the purpose of making cross sections.

    .. warning::

        **``constantdist``** The reason for this option is the fact that two
        waypoints are most likely not a multiple of ``dist`` apart from each
        other. A previous implementation included an interpolation between, but
        the interpolation of a parametric curve in 3D space is not as simple as
        I had hoped. The following explanation is for ``N`` segments defined by
        ``N+1`` waypoints.
        ``constantdist`` sets the spacing constant, but the waypoint is most
        likely not exactly hit making the distance between the point before
        the (n+1)-th waypoint smaller than the the rest. This will be the
        same in all segments between n and (n+1)th waypoints. The alternative
        is to create linspace between the waypoints with
        ``N = round(total_dist/dist) + 1`` points including ``n`` and
        ``(n+1)`` waypoints. This is essentially the more accurate way of
        defining the segments, but each segment has a different spacing that
        is not exactly ``dist``.



    Parameters
    ----------
    lat : np.ndarray
        waypoint latitudes
    lon : np.ndarray
        waypoint longitudes
    dist : float
        distance in degrees
    constantdist: bool
        Not really important for the enduser, but there are two ways of
        computing a gctrack along a curve. for 2 points only the most accurate
        way is to set ``constant_dist`` to ``False``.


    Returns
    -------
    tuple(np.ndarray, np.ndarray)


    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.10.10 02.17

    """

    # First get distances between points
    N = len(lon)
    # dists = np.zeros(N-1)
    # az = np.zeros(N-1)

    # Create Geodesic class
    # G = Geodesic(flattening=0.0)

    # Get tracks between segments that are far apart
    dists, az = geodiff(lat, lon)

    if np.any(dists/100 < dist):
        raise ValueError(
            "Final deltadeg between points in tracks should be at least "
            "100 times less smallest distance between waypoints.")

    if not constantdist:
        updated_dist = []

    # Cumulative station distances
    sdists = np.zeros(N)
    sdists[1:] = np.cumsum(dists)

    tracks = []
    for _i in range(N-1):

        # Choice of created equally space vector a long the track
        if constantdist:
            # New dist vector
            trackdists = np.arange(0, dists[_i], dist)
        else:
            # Get the length of the linspace
            N = int(np.round((dists[_i])/dist))

            # Get the closest distance measure
            updist = dists[_i]/N

            # Attached to list.
            updated_dist.append(updist)

            trackdists = np.linspace(0, dists[_i] - updist, N)

        track = np.array(reckon(lat[_i], lon[_i], trackdists, az[_i]))

        # Remove geolocations that overshot
        tmpdist = geodist(track[0, :], track[1, :])
        track = track[:, np.where(tmpdist < dists[_i])[0]]
        tracks.append(track)

    # Add last point because usually not added
    tracks.append(np.array(((lat[-1], lon[-1]),)).T)

    # Get tracks
    utrack = np.hstack(tracks).T

    # Remove duplicates if there are any
    _, idx = np.unique(utrack, return_index=True, axis=0)
    utrack = utrack[np.sort(idx), :]

    # Get distances along the new track
    udists, _ = geodiff(utrack[:, 0], utrack[:, 1])

    # Compute cumulative distance
    M = len(utrack[:, 0])
    cdists = np.zeros(M)
    cdists[1:] = np.cumsum(udists)

    if constantdist:
        return utrack[:, 0], utrack[:, 1], cdists, sdists
    else:
        return utrack[:, 0], utrack[:, 1], cdists, updated_dist


def geo2cart(
    r: float or np.ndarray, latitude: np.ndarray,
        longitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    latrad = latitude / 180*np.pi
    lonrad = longitude / 180 * np.pi

    # Convert to geolocations
    x = r * np.cos(latrad) * np.cos(lonrad)
    y = r * np.cos(latrad) * np.sin(lonrad)
    z = r * np.sin(latrad)

    return x, y, z


def cart2geo(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
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


def epi2euc(epi: float) -> float:
    """Converts epicentral distance in to a euclidean distance along the
    corresponding chord."""
    return 2 * R_EARTH * np.sin(np.pi * epi / 360)


def euc2epi(euc: float) -> float:
    """Converts euclidean distance to epicentral distance"""
    return 360 * np.arcsin(euc/(2*R_EARTH)) / np.pi


def fix_map_extent(extent: float, fraction=0.05):

    # Get extent values and fix them
    minlon, maxlon, minlat, maxlat = extent

    latb = (maxlat - minlat) * fraction
    lonb = (maxlon - minlon) * fraction

    # Max lat
    if maxlat + latb > 90.0:
        maxlat = 90.0
    else:
        maxlat = maxlat + latb

    # Min lat
    if minlat - latb < -90.0:
        minlat = -90.0
    else:
        minlat = minlat - latb

    # Max lon
    if maxlon + lonb > 180.0:
        maxlon = 180.0
    else:
        maxlon = maxlon + lonb

    # Minlon
    if minlon - lonb < -180.0:
        minlon = -180.0
    else:
        minlon = minlon - lonb

    return [minlon, maxlon, minlat, maxlat]
