# Basic
from typing import Optional, Union, Iterable

# External
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot

# Internal
from pyglimer.ccp.plot_utils.plot_map import plot_map
from pyglimer.ccp.plot_utils.plot_line_buffer import plot_line_buffer
from pyglimer.ccp.plot_utils.midpointcolornorm import MidpointNormalize


def get_ax_coor(ax, lat, lon):
    """Just a way to get axes coordinates

    Parameters
    ----------
    ax : axes
        Givenn axes
    lat : float
        Latitude
    lon : float
        Longitude

    Returns
    -------
    Tuple
        Axes coordinates for a given set of lat, lon.

    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.08.20 11.00 (Lucas Sawade)

    """

    # Geo point
    pgeo = np.array((lon, lat))

    # Geo to data
    pdata = ax.transData.transform(pgeo)

    # Data to disp
    pdisp = ax.transAxes.inverted().transform(pdata)

    return pdisp[0], pdisp[1]


def plot_cross_section(
        ccp, lat, lon,
        z0: Optional[float] = None,
        ax: Optional[Axes] = None,
        geoax: Optional[Union[GeoAxes, GeoAxesSubplot]] = None,
        mapplot: bool = True,
        minillum: int = 50,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        label: Optional[str] = None,
        rfcmap: str = "seismic",
        depthextent: Optional[Iterable] = None,
        mapextent: Optional[Iterable] = None):
    """Plots a cross section for given waypoints. If no axes are given, the
    function will also create figures for the map and the cross section.

    Parameters
    ----------
    ccp : CCPStack
        CCPStack as computed using the cpp module
    lat : Arraylike
        Latitudes of the waypoints defining the cross section
    lon : Arraylike
        Longitudes of the waypoints defining the cross section
    z0 : Optional[float], optional
        if given the map will be plotted with an illumination map at the given
        depth, by default None
    ax : Optional[Axes], optional
        Axes to plot the cross section in, by default None
    geoax : Optional[Union[GeoAxes, GeoAxesSubplot]], optional
        Axes to plot the waypoins in, by default None
    mapplot : bool, optional
        plot the map or not, by default True
    minillum : int, optional
        minimum illumination count to not be grayed out, by default 50
    vmin : Optional[float], optional
        minimum value of the cross section, by default None
    vmax : Optional[float], optional
        maximum value of the cross section, by default None
    label : Optional[str], optional
        label to put in the corner of the cross section plot and the
        cross section waypoints on the map, by default None
    rfcmap : str, optional
        cmap name for the plotting of the cross section, by default "seismic"
    depthextent : Optional[Iterable], optional
        List containing two entries defining min and max depth, by default None
    mapextent : Optional[Iterable], optional
        List of 4 entries defining [minlon,maxlon, minlat, maxlat],
        by default None

    Returns
    -------
    Tuple
        ax, geoax


    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.21 20.00 (Lucas Sawade)

    """

    # Get Cross section
    slat, slon, sdists, qlat, qlon, qdists, qz, qillum, qccp, epi_area = \
        ccp.get_profile(lat, lon)

    # Define norms
    if vmin is None:
        vmin = -np.quantile(np.abs(qccp[qccp < 0]), 0.98)
    if vmax is None:
        vmax = np.quantile(qccp[qccp > 0], 0.98)

    # Norm for the cross section
    rfnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.0)
    snorm = mcolors.Normalize(vmin=0, vmax=sdists[-1])

    # Set illumination boundaries for section plotting
    alpha = qillum/minillum
    alpha = np.where(alpha >= 1, 1.0, alpha)

    # ############### Plot map ###################
    if mapplot:

        # Create a map figure if None present
        if geoax is None:

            plt.figure()
            geoax = plt.axes(projection=ccrs.PlateCarree())
            if mapextent is not None:
                geoax.set_extent(mapextent)
            plot_map(geoax)
            geoax.tick_params(labelright=False, labeltop=False)

            # Plot illumination
            if z0 is not None:
                # Get depth slice (mainly for the illumination)
                zqlat, zqlon, zqill, zqccp, zextent, z0 = ccp.get_depth_slice(
                    z0=z0)
                zalpha = np.where(zqill == 0, 0, 0.5)

                # Get norm and cmap for the illumination map
                illumnorm = mcolors.LogNorm(vmin=1, vmax=zqill.max())
                illumcmap = 'magma_r'

                geoax.imshow(
                    zqill, alpha=zalpha,
                    extent=zextent, origin='lower',
                    transform=ccrs.PlateCarree(),
                    cmap=illumcmap, norm=illumnorm)

                # Create colorbar from artifical scalarmappable (alpha is
                # problematic)
                c = plt.colorbar(
                    ScalarMappable(cmap=plt.get_cmap(
                        illumcmap), norm=illumnorm),
                    orientation='vertical', aspect=40)
                c.set_label("Hitcount")

                # Set colormap alpha manually
                c.solids.set(alpha=0.5)

        # Plot cross section
        geoax.plot(qlon, qlat, 'k', zorder=10, transform=ccrs.PlateCarree())

        # Plot waypoints
        geoax.scatter(
            slon, slat, c=sdists, s=50,
            cmap='Greys', norm=snorm,
            marker='o', edgecolor='k',
            transform=ccrs.PlateCarree(),
            zorder=11)

        # Plot buffer that shows where we got cross section stuff from
        _ = plot_line_buffer(
            qlat, qlon, delta=epi_area, ax=geoax,
            linestyle='--', linewidth=1.0, alpha=1.0,
            facecolor='none', edgecolor='k',
            zorder=5)

        # This plots the ccp, locations, but I'm not sure if that's necesary
        # geoax.plot(ccp.coords_new[1], ccp.coords_new[0], 'k',
        #            zorder=0, transform=ccrs.PlateCarree())

        # Plot label
        if label is not None:
            x, y = get_ax_coor(geoax, qlat[0], qlon[0])

            geoax.text(
                x, y + 0.05,
                label,
                horizontalalignment='center',
                verticalalignment='bottom',
                fontdict=dict(fontsize='small'),
                bbox=dict(facecolor='w', edgecolor='k'),
                transform=geoax.transAxes, zorder=100)

    # ############### Plot section ###################

    # Plot section
    if ax is None:
        plt.figure()
        ax = plt.axes(facecolor=(0.8, 0.8, 0.8))
    else:
        ax.set_facecolor((0.8, 0.8, 0.8))

    if depthextent is not None:
        ax.set_ylim(depthextent[::-1])
        minz = depthextent[0]
    else:
        minz = np.min(qz)

    # Plot section
    plt.imshow(
        qccp,
        cmap=rfcmap, norm=rfnorm,
        extent=[0, np.max(qdists), np.max(qz), np.min(qz)],
        aspect='auto', rasterized=True)  # , alpha=alpha )

    # Plot waypoints

    ax.scatter(
        sdists, minz * np.ones_like(sdists), c=sdists, s=50,
        cmap='Greys', norm=snorm,
        marker='o', edgecolor='k',
        zorder=10, clip_on=False)

    plt.xlabel('Offset [$^\circ$]')
    plt.ylabel('Depth [km]')

    if label is not None:
        ax.text(
            qdists[0], 1.025,
            label,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontdict=dict(fontsize='small'),
            bbox=dict(facecolor='none', edgecolor='none'),
            transform=ax.transAxes, zorder=100)

    return ax, geoax
