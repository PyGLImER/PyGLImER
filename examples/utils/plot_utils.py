"""

Plot utilities not to modify plots or base plots.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 19, 2020


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy
from cartopy.crs import PlateCarree

def set_mpl_params():
    params = {
        'font.family': 'Avenir Next',
        'pdf.fonttype': 42,
        'font.weight': 'bold',
        'figure.dpi': 150,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.5,
        'axes.labelsize': 15,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 14,
        'xtick.direction': 'in',
#         'xtick.top': True,  # draw label on the top
#         'xtick.bottom': True,  # draw label on the bottom
#         'xtick.minor.visible': True,
#         'xtick.major.top': True,  # draw x axis top major ticks
#         'xtick.major.bottom': True,  # draw x axis bottom major ticks
#         'xtick.minor.top': True,  # draw x axis top minor ticks
#         'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'ytick.labelsize': 14,
#         'ytick.direction': 'in',
#         'ytick.left': True,  # draw label on the top
#         'ytick.right': True,  # draw label on the bottom
#         'ytick.minor.visible': True,
#         'ytick.major.left': True,  # draw x axis top major ticks
#         'ytick.major.right': True,  # draw x axis bottom major ticks
#         'ytick.minor.left': True,  # draw x axis top minor ticks
#         'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'legend.fancybox': False,
        'legend.frameon': False,
        'legend.loc': 'upper left',
        'legend.numpoints': 2,
        'legend.fontsize': 'large',
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit'
    }
    matplotlib.rcParams.update(params)
    matplotlib.font_manager._rebuild()

    


def plot_catalog(catalog):
    """ Takes in event catalog and plots events as a function of location and moment
    magnitude."""
    
    plt.figure(figsize=(20,7.5))
    ax = plt.subplot(111, projection=PlateCarree())
    
    size = 1
    mags = []
    lats = []
    lons = []
    
    for event in catalog:
        # Get mag
        mags.append(event.preferred_magnitude().mag)
        
        # Get location
        origin = event.preferred_origin()
        lats.append(origin.latitude)
        lons.append(origin.longitude)
    
    # Add coastline
    ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
                   linewidth=0.5, facecolor=(0.9, 0.9, 0.9))

    # Plot events
    c = ax.scatter(np.array(lons), np.array(lats),  c=np.array(mags), s=size*np.array(mags)**3,
               marker="o", cmap="magma", vmin=3, vmax=7.5,
               edgecolor="k", linewidth=0.75, zorder=201)
    cbar = plt.colorbar(c, pad=0.005, shrink=1)    
    cbar.ax.set_ylabel(r"       $M_w$", rotation=0)
    