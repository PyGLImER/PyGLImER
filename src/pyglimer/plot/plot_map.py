'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 4th August 2020 11:02:52 am
Last Modified: Friday, 20th January 2023 03:51:52 pm
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from cartopy.crs import PlateCarree
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy


def set_mpl_params():
    params = {
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.labelsize': 10,
        'xtick.labelsize': 7,
        'xtick.direction': 'in',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'ytick.labelsize': 7,
        'ytick.direction': 'in',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
    }
    matplotlib.rcParams.update(params)


def plot_map(
        cl=0.0, lat=None, lon=None, profile=None, p_direct=True, geology=False,
        states=False):
    """plot a map"""
    ax = plt.gca()
    if lat and lon:
        ax.set_extent((lon[0], lon[1], lat[0], lat[1]))
    else:
        ax.set_global()
    ax.frameon = True
    ax.outline_patch.set_linewidth(0.75)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=PlateCarree(central_longitude=0.0),
                      draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-', zorder=-1.5)
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = True

    # Add Coastline
    ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
                   linewidth=0.5, facecolor=(0.9, 0.9, 0.9))
    # Add the biggest waterbodies
    ax.add_feature(
        cartopy.feature.NaturalEarthFeature('physical', 'lakes', '110m'),
        zorder=-2, edgecolor='black',
        linewidth=0.5, facecolor='w')

    if lat and lon:
        # dy = np.floor((lat[1] - lat[0])/6)
        dx = np.floor((lat[1] - lat[0])/6)
        xt = np.arange(lon[0], lon[1], dx)
        yt = np.arange(lat[0], lat[1], dx)
    else:
        xt = np.linspace(-180, 180, 13)
        yt = np.linspace(-90, 90, 13)
    ax.set_xticks(xt, crs=ccrs.PlateCarree())
    ax.set_yticks(yt, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    if geology:
        ax.add_wms(
            wms='https://mrdata.usgs.gov/services/worldgeol?',
            layers=['geology'], zorder=-2)
    if states:
        ax.add_feature(feature=cartopy.feature.STATES,
                       linewidth=0.25, zorder=-2)
    if profile:
        if type(profile) == tuple:
            if p_direct:
                plt.plot((profile[0], profile[1]), (profile[2], profile[3]),
                         color='blue', linewidth=1.5, transform=PlateCarree(),
                         label='profile 1')
            else:
                p = profile
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        xy=(p[0], p[2]), height=p[3]-p[2], width=p[1]-p[0],
                        alpha=.2, facecolor='blue'))
        else:
            # colorlist
            if p_direct:
                for ii, p in enumerate(profile):
                    plt.plot((p[0], p[1]), (p[2], p[3]),
                             linewidth=1.5, transform=PlateCarree(),
                             label='profile ' + str(ii+1))
            else:
                cl = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey', 'tab:purple']
                for ii, p in enumerate(profile):
                    ax.add_patch(
                        matplotlib.patches.Rectangle(
                            xy=(p[0], p[2]), height=p[3]-p[2], width=p[1]-p[0],
                            alpha=.2, label=ii, facecolor=cl[ii]))
            # ax.legend()

    return ax


def plot_stations(slat, slon, cl=0.0):
    """Plots stations into a map
    """

    # Weird fix because cartopy is weird
    if cl == 180.0:
        slon = [sl + cl if sl <= 0
                else sl - cl
                for sl in slon]
    ax = plt.gca()
    ax.scatter(slon, slat, s=13, marker='v', c=((0.7, 0.2, 0.2),),
               edgecolors='k', linewidths=0.25, zorder=-1, label='stations')


def plot_bins(binlat, binlon, cl=0.0):
    """
    Plots bins into a map
    """

    # Weird fix because cartopy is weird
    if cl == 180.0:
        binlon = [bl + cl if bl <= 0
                  else bl - cl
                  for bl in binlon]

    ax = plt.gca()
    ax.scatter(binlon, binlat, s=3, marker='.', c='b',
               edgecolors=None, linewidths=0.25, zorder=-1,
               label='bin centres')


def plot_illum(binlat, binlon, dbin, illum, cl=0.0):
    if cl == 180.0:
        binlon = [bl + cl if bl <= 0
                  else bl - cl
                  for bl in binlon]

    ax = plt.gca()
    # Create histogram
    illumflat = np.log10(np.sum(illum, axis=1))

    il = ax.scatter(
        binlon, binlat, c=illumflat, cmap='plasma', s=10, edgecolors=None,
        label='bin centres', zorder=-1)
    cb = plt.colorbar(il, ax=ax)

    cb.set_label('log10(hits)')


def plot_scattered_colormap(
        binlat: np.ndarray, binlon: np.ndarray, vals: np.ndarray,
        amplitude: bool = False,
        cmap: str = 'gist_rainbow'):
    ax = plt.gca()

    pltfig = ax.scatter(
        binlon, binlat, c=vals, cmap=cmap, s=5, edgecolors=None,
        label='bin centres', zorder=-1)  # 'viridris', 'gist_rainbow
    cb = plt.colorbar(pltfig, ax=ax)
    plt.tight_layout()
    if amplitude:
        cb.set_label('Amplitude')
    else:
        cb.set_label('Depth [km]')


def plot_station_db(
        slat, slon, lat: tuple or None = None, lon: tuple or None = None,
        profile=None, p_direct=True, outputfile=None, format='pdf',
        dpi=300, geology=False):
    cl = 0.0
    set_mpl_params()
    plt.figure(figsize=(9, 4.5))
    plt.subplot(projection=PlateCarree(central_longitude=cl))
    plot_map(cl=cl, lat=lat, lon=lon, profile=profile, p_direct=p_direct,
             geology=geology)
    plot_stations(slat, slon, cl=cl)
    plt.tight_layout()
    if outputfile is None:
        plt.show()
    else:
        if format in ["pdf", "epsg", "svg", "ps"]:
            dpi = None
        plt.savefig(outputfile, format=format, dpi=dpi, bbox_inches='tight')


def plot_map_ccp(
        lat: tuple or None, lon: tuple or None, stations: bool,
        slat: list or np.ndarray, slon: list or np.ndarray,
        bins: bool, bincoords: tuple, dbin, illum: bool,
        illummatrix: np.ndarray, profile: list or None,
        p_direct=True, outputfile=None, format='pdf', dpi=300, geology=False,
        title=None):
    """
        Create a map plot of the CCP Stack containing user-defined information.

        Parameters
        ----------
        lat: tuple, optional
            boundaries for the map view
        lon: tuple, optional
            boundaries for the map view
        stations : bool, optional
            Plot the station locations, by default False

        illum : bool, optional
            Plot bin location with colour depending on the depth-cumulative
            illumination at bin b, by default False
        profile : list or tupleor None, optional
            Plot locations of cross sections into the plot. Information about
            each cross section is given as a tuple (lon1, lon2, lat1, lat2),
            several cross sections are given as a list of tuples in said
            format, by default None.
        p_direct : bool, optional
            If true the list in profile decribes areas with coordinates of the
            lower left and upper right corner, by default True
        outputfile : str or None, optional
            Save plot to file, by default None
        format : str, optional
            Format for the file to be saved as, by default 'pdf'
        dpi : int, optional
            DPI for none vector format plots, by default 300
        geology : bool, optional
            Plot a geological map. Requires internet connection
        title : str, optional
            Set title for plot.
    """
    cl = 0.0
    set_mpl_params()
    plt.figure(figsize=(9, 4.5))
    plt.subplot(projection=PlateCarree(central_longitude=cl))
    _ = plot_map(
        cl=cl, lat=lat, lon=lon, profile=profile, p_direct=p_direct,
        geology=geology, states=True)
    if bins:
        plot_bins(bincoords[0], bincoords[1], cl=cl)
    if illum:
        plot_illum(bincoords[0][0], bincoords[1][0], dbin, illummatrix, cl=cl)
    if stations:
        plot_stations(slat, slon, cl=cl)
    if bins or illum or stations or profile:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=4, mode="expand", borderaxespad=0.)

    if title:
        plt.title(title, fontdict={'fontweight': 'bold'}, y=1.1)
    plt.tight_layout()
    if outputfile is None:
        plt.show()
    else:
        if format in ["pdf", "epsg", "svg", "ps"]:
            dpi = None
        plt.savefig(outputfile, format=format, dpi=dpi, bbox_inches='tight')


def plot_vel_grad(
        coords, a, z, plot_amplitude: bool,
        lat: tuple or None, lon: tuple or None,
        outputfile=None, format='pdf', dpi=300, cmap: str = 'gist_rainbow',
        geology=False, title=None):
    """
    Plot velocity gradient. Use method implemented into object!

    Parameters
    ----------
    coords : np.ndarray
        lats and lons
    a : np.ndarray
        picked amplitudes
    z : np.ndarray
        picked depths
    plot_amplitude : bool
        Plot Amplitude or depth?
    lat : tuple or None
        Latitude boarder for map
    lon : tuple o rNone
        Longitude boarder for map
    outputfile : str, optional
        write plot to file, by default None
    format : str, optional
        file format, by default 'pdf'
    dpi : int, optional
        resolution for none-vector plots, by default 300
    """
    cl = 0.0
    set_mpl_params()
    _ = plt.figure(figsize=(9, 4.5))

    plt.subplot(projection=PlateCarree(central_longitude=cl))

    _ = plot_map(cl=cl, lat=lat, lon=lon, geology=geology, states=True)
    # plot depth distribution or amplitude?
    if plot_amplitude:
        data = a
    else:
        data = z

    plot_scattered_colormap(
        coords[0][0], coords[1][0], data, amplitude=plot_amplitude, cmap=cmap)
    plt.legend()
    if title:
        plt.title(title, fontdict={'fontweight': 'bold'})
    plt.tight_layout()

    # Write file
    if outputfile is None:
        plt.show()
    else:
        if format in ["pdf", "epsg", "svg", "ps"]:
            dpi = None
        plt.savefig(outputfile, format=format, dpi=dpi, bbox_inches='tight')
