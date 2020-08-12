'''
Author: Peter Makus (peter.makus@student.uib.no)

Created: Tuesday, 4th August 2020 11:02:52 am
Last Modified: Wednesday, 12th August 2020 12:11:51 pm
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


def plot_map(cl=0.0, lat=None, lon=None, profile=None, p_direct=True):
        """plot a map"""
        ax = plt.gca()
        if lat and lon:
            ax.set_extent((lon[0],lon[1],lat[0],lat[1]))
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
        
        if lat and lon:
            dy = np.floor((lat[1] - lat[0])/6)
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
        # ax.set_xticks(xt, crs=ccrs.Robinson())
        # ax.set_yticks(yt, crs=ccrs.Robinson())

        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        # ax.yaxis.set_major_formatter(LatitudeFormatter())
        if profile:
            if type(profile) == tuple:
                if p_direct:
                    plt.plot((profile[0], profile[1]), (profile[2], profile[3]),
                    color='blue', linewidth=1, transform=PlateCarree())
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
                        linewidth=1, transform=PlateCarree(), label=ii)
                else:
                    cl = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey', 'tab:purple']
                    for ii, p in enumerate(profile):
                        ax.add_patch(
                            matplotlib.patches.Rectangle(
                                xy=(p[0], p[2]), height=p[3]-p[2], width=p[1]-p[0],
                                alpha=.2, label=ii, facecolor=cl[ii]))
                ax.legend()
            
        return ax


def plot_stations(slat, slon, cl=0.0):
        """Plots stations into a map
        """

        # slat = [station[0] for station in stations]
        # Weird fix because cartopy is weird
        if cl == 180.0:
            slon = [sl + cl if sl <=0
                    else sl - cl
                    for sl in slon]
        # else:
        #     slon = [station[1] for station in stations]
        #ax = plot_map()
        ax = plt.gca()
        ax.scatter(slon, slat, s=13, marker='v', c=((0.7, 0.2, 0.2),),
                   edgecolors='k', linewidths=0.25, zorder=-1)

def plot_station_db(slat, slon, lat:tuple or None=None, lon:tuple or None=None,
                    profile=None, p_direct=True, outputfile=None, format='pdf',
                    dpi=300):
    cl = 0.0
    set_mpl_params()
    plt.figure(figsize=(9,4.5))
    plt.subplot(projection=PlateCarree(central_longitude=cl))
    plot_map(cl=cl, lat=lat, lon=lon, profile=profile, p_direct=p_direct)
    plot_stations(slat, slon, cl=cl)
    if outputfile is None:
        plt.show()
    else:
        if format in ["pdf", "epsg", "svg", "ps"]:
            dpi = None
        plt.savefig(outputfile, format=format, dpi=dpi)