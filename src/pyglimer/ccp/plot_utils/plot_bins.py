"""
Module for plotting bins

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Lucas Sawade (lsawade@princeton.edu)


Last Update: November 2019

"""

from cartopy.crs import PlateCarree
import matplotlib.pyplot as plt
import numpy as np
from .plot_map import plot_map


def plot_bins(stations, bins, ax=None, kwargsS=None, kwargsB=None):
    """ Takes in original stations and bins and plots them on a map.

    Parameters
    ----------
    stations: tuple of latitude and longitude lists of the stations.
    bins: tuple of latitude and longitude lists of the bin centers.
    ax: pyplot axis to place the figure in. If none the function will create
        figure and axes to plot the map in. Default is None.
    kwargsS: dictionary of keyword arguments to be passed to the station
             scatterplot function

    Returns
    -------

    """

    # Avoid using mutable arguments create empty kwargsets
    if kwargsS is None:
        kwargsS = {}
    if kwargsB is None:
        kwargsB = {}

    # Create figure and axis if none were given
    if ax is None:
        show = True
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111, projection=PlateCarree(0.0))

    # Plot map
    plot_map(ax)

    # Standard marker settings
    markerkwarg = {"marker": 'v',
                   "s": 10,
                   "edgecolor": "k"}
    # Plot bins
    ax.scatter(bins[1], bins[0],
               **markerkwarg, c=((1, 1, 1),), **kwargsB)

    # Plot stations
    ax.scatter(stations[:, 1], stations[:, 0],
               **markerkwarg, c=((0.8, 0.2, 0.2),), **kwargsS)

    # Set extent
    minX, maxX = np.min(bins[1]), np.max(bins[1])
    minY, maxY = np.min(bins[0]), np.max(bins[0])
    buffer = 0.1  # in fraction
    xbuffer = (maxX - minX) * buffer
    ybuffer = (maxY - minY) * buffer
    mapextent = [minX - xbuffer, maxX + xbuffer,
                 minY - ybuffer, maxY + ybuffer]
    ax.set_extent(mapextent)

    # Get location on map in textbox
    tex = plt.text(1.025, 1, "lat:         \nlon:        ",
                   horizontalalignment='left',
                   verticalalignment='top', transform=ax.transAxes)

    def update_tex(x, y):
        if x is None or y is None:
            tex.set_text("lat:         \nlon:        ")
        else:
            tex.set_text(r"lat: %3.4f$^\circ$ \nlon: %3.4f$^\circ$" % (y, x))
        plt.draw()

    def mouse_move(event):
        x, y = event.xdata, event.ydata
        update_tex(x, y)

    plt.connect('motion_notify_event', mouse_move)

    if show:
        plt.show(block=False)
