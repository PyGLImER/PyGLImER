"""

Plotting utilities for the rays

"""

import matplotlib.pyplot as plt


def plot_rays(clat, clon, depth, dtimes):
    """ Plots the ray for a given

    Parameters
    ----------
    clat
    clon
    depth
    dtimes

    Returns
    -------

    """
    fig = plt.figure(figsize=(10, 5))

    # as a function of depth
    ax1 = fig.add_subplot(121, projection='3d')
    for _i in range(clon.shape[0]):
        ax1.plot(clon[_i, :], clat[_i, :], depth[1:], 'k')
    ax1.set_xlabel('Latitude in [deg]')
    ax1.set_ylabel('Longitude in [deg]')
    ax1.set_zlabel('Depth in [km]')
    ax1.invert_zaxis()

    # as a function of time
    ax2 = fig.add_subplot(122, projection='3d')
    for _i in range(clon.shape[0]):
        ax2.plot(clon[_i, :], clat[_i, :], dtimes[_i, 1:, 0], 'k')
    ax2.set_xlabel('Latitude in [deg]')
    ax2.set_ylabel('Longitude in [deg]')
    ax2.set_zlabel('Time in [s]')
    ax2.invert_zaxis()

    plt.show(block=True)
