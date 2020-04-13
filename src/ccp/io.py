"""

This module will handle file IO. The problematic functions here are the
incoming `.mat` files that are produced by the GLImER database.

"""
import h5py
import numpy as np

def load_rawrf(filename):
    """

    Parameters
    ----------
    filename (str): filename of the receiver function `.mat`-file to be loaded.
                    The `.mat`-file is expected to contain following parameters:

                        crfs: MxN matrix of M Receiver Functions and N elements
                        cbaz: M elements long vector with back azimuth values
                              corresponding to Receiver functions
                        crayp: M elements long vector with ray parameter values
                               corresponding to Receiver functions
                        cslat: M elements long vector with station latitude
                               corresponding to Receiver functions
                        cslon: M elements long vector with station longitude
                               corresponding to Receiver functions
                        celev: M elements long vector with station elevation
                        dt: sampling interval

    Returns
    -------
    rfs, baz, rayp, lat, lon, elev, dt

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

    try:
        with h5py.File(filename, 'r') as f:
            vlat = f['lat'][()].flatten()
            vlon = f['lon'][()].flatten()
            vdep = f['dep'][()].flatten()
            vp = f['vp'][()].transpose()
            vs = f['vs'][()].transpose()
    except Exception as e:
        print(e)
        print()

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
