"""
This module will handle file IO. The problematic functions here are the
incoming `.mat` files that are produced by the GLImER database and the
serialisation of the :class:`~pyglimer.ccp.ccp.CCPStack` object.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Lucas Sawade (lsawade@princeton.edu)

"""
import os
import pickle
from copy import deepcopy
from typing import List, Tuple

import h5py
import numpy as np

from pyglimer.ccp.compute.bin import BinGrid


def load_rawrf(filename):
    """Loads a matlab raw RF file.

    Parameters
    ----------
    filename : str
        filename of the receiver function ``.mat``-file to be loaded. The
        ``.mat``-file is expected to contain following parameters. See Notes
        below for required content of matlab rawRf file.


    Returns
    -------
    rfs, baz, rayp, lat, lon, elev, dt
        Set of RF variables


    Notes
    -----

    File content

    :crfs:
        MxN matrix of M Receiver Functions and N elements
    :cbaz:
        M elements long vector with back azimuth values corresponding to
        Receiver functions
    :crayp:
        M elements long vector with ray parameter values corresponding to
        Receiver functions
    :cslat:
        M elements long vector with station latitude corresponding to
        Receiver functions
    :cslon:
        M elements long vector with station longitude corresponding to
        Receiver functions
    :celev:
        M elements long vector with station elevation
    :dt:
        sampling interval


    """

    with h5py.File(filename, 'r') as f:
        baz = f['cbaz'][()].flatten()
        elev = f['celev'][()].flatten()
        rayp = f['crayp'][()].flatten()
        rfs = f['crfs'][()].transpose()
        lat = f['cslat'][()].flatten()
        lon = f['cslon'][()].flatten()
        dt = f['dt'][()]

    return rfs, baz, rayp, lat, lon, elev, dt


def load_velocity_model(filename):
    """ Loads velocity model and outputs velocities and corresponding
    parameters.

    Parameters
    ----------
    filename

    Returns
    -------
    vlat, vlon, vdep, vp, vs

    """

    # Errors (missing dataset, unreadable file, ...) propagate to the
    # caller. The previous broad except printed the error and then raised
    # a NameError on the undefined return values, hiding the real cause.
    with h5py.File(filename, 'r') as f:
        vlat = f['lat'][()].flatten()
        vlon = f['lon'][()].flatten()
        vdep = f['dep'][()].flatten()
        vp = f['vp'][()].transpose()
        vs = f['vs'][()].transpose()

    return vlat, vlon, vdep, vp, vs


def load_tracefile(filename):
    """Loading an RF file that has been raytraced."""

    # Load file
    npzfile = np.load(filename)

    # Allocate variables
    rf = npzfile['rf']
    lat = npzfile['lat']
    lon = npzfile['lon']
    baz = npzfile['baz']
    rayp = npzfile['rayp']
    elev = npzfile['elev']
    clat = npzfile['clat']
    clon = npzfile['clon']
    depth = npzfile['depth']
    dtimes = npzfile['dtimes']

    return rf, lat, lon, baz, rayp, elev, clat, clon, depth, dtimes


def read_ccp(filename: str, fmt: str = None):
    """
    Read CCP-Stack class file from input folder.

    :param filename: Filename of the input file with file ending.
        The default is 'ccp.pkl'.
    :type filename: str, optional
    :param fmt: File format, can be none if the filename has an ending,
        possible options are `"pickle"` or `"npz"`. The default is None.
    :type fmt: str, optional
    :raises ValueError: For unknown formats
    :return: CCPStack object
    :rtype: :class:`~pyglimer.ccp.ccp.CCPStack`

    .. note::

        Loading Files from numpy format will take a while as some
        recomputations are necessary.
    """
    # Imported here to avoid a circular import at module load time
    # (pyglimer.ccp.ccp imports this module)
    from pyglimer.ccp.ccp import CCPStack

    # Trying to identify filetype from ending:
    if not fmt:
        x = filename.split('.')
        if len(x) == 1:
            raise ValueError(
                "Could not determine format, please provide a valid format")
        if x[-1] == 'pkl':
            fmt = 'pickle'
        elif x[-1] == 'npz':
            fmt = 'npz'
        else:
            raise ValueError(
                "Could not determine format, please provide a valid format")
    else:
        fmt = fmt.lower()

    # Open provided file
    if fmt == "pickle":
        with open(filename, 'rb') as infile:
            ccp = pickle.load(infile)
    elif fmt == 'npz':
        ccp = CCPStack(None, None, None, None, _load=filename)
    else:
        raise ValueError("Unknown format ", fmt)

    return ccp


def _load_ccp_npz(ccp, filename: str):
    """
    Load a file from numpy format. For that We will do some recomputations
    and will also have to unpack the used arrays.

    :param ccp: The initialised :class:`~pyglimer.ccp.ccp.CCPStack` object.
    :type ccp: CCPStack
    :param filename: path to load from.
    :type filename: str
    :return: The loaded :class:`~pyglimer.ccp.ccp.CCPStack` object.
    :rtype: CCPStack
    """
    loaded = dict(np.load(filename))
    for k in loaded.pop('attrs_pkl'):
        loaded[k] = loaded[k][0]
    ccp.bingrid = BinGrid(
        loaded.pop('statlat'), loaded.pop('statlon'), loaded.pop('edist'),
        loaded.pop('phase'))
    # Compute bins
    ccp.bingrid.compute_bins()

    # Initialize kdTree for Bins
    ccp.bingrid.bin_tree()
    ccp.coords = ccp.bingrid.bins
    for k in loaded:
        ccp.__dict__[k] = loaded[k]
    return ccp


def _save_ccp_npz(ccp, oloc: str):
    """
    The point of this function is that, in contrast to pickling,
    the data is stil accessible after PyGLImER has been upgraded.
    For this, we will have to recompute some objects when loading.
    (in particular, the bingrid object)
    """
    # Save the important things from the bingrid object
    d = deepcopy(ccp.__dict__)
    d['phase'] = deepcopy(ccp.bingrid.phase)
    d['statlat'] = deepcopy(ccp.bingrid.latitude)
    d['statlon'] = deepcopy(ccp.bingrid.longitude)
    d['edist'] = deepcopy(ccp.bingrid.edist)
    d.pop('coords')
    d.pop('bingrid')

    # Attributes that need to be changed to arrays
    apkl = []
    for k in d.keys():
        if isinstance(d[k], np.ndarray):
            continue
        elif isinstance(d[k], list):
            d[k] = np.array(d[k])
            continue
        d[k] = np.array([d[k]])
        apkl.append(k)
    d['attrs_pkl'] = np.array(apkl)
    np.savez_compressed(oloc, **d)
