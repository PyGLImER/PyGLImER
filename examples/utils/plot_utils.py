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
from pyglimer import RFTrace


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
    
    
def plot_single_rf(rf: RFTrace, outputdir=None)
    """Creates plot of a single receiver function
    
    """
    
    
    
def plot_section(st, timelimits=[0, 7200], epilimits=[0, 180], 
                 scalingfactor: float = 2.0, linewidth: float = 0.25,
                 outputdir: str or None = None, title: str or None = None,
                 show: bool = True):
    
    # Setup
    components = ["R", "T", "Z"]
    colors = [(0.7, 0.1, 0.1), (0.1, 0.5, 0.1), (0.1, 0.1, 0.5)]
    
    # 
    for _color, _comp in zip(colors, components):
        
        plt.figure(figsize=(10,15))
        
        for tr in st.select(component=_comp):
            plt.plot(scalingfactor * tr.data/np.max(np.abs(tr.data)) + tr.stats.distance, 
                     tr.times(), c=_color, lw=linewidth)
        
        plt.xlim(epilimits[0], epilimits[1])
        plt.ylim(timelimits[0], timelimits[1])
        plt.xlabel(r"$\Delta$ [$^{\circ}$]")
        plt.ylabel(r"Time [s]")
        
        if title is not None:
            plt.title(title + " - %s" % _comp)
        else:
            plt.title("%s component" % _comp)
        
        if outputdir is None:
            plt.show()
        else:
            outputfilename = os.path.join(outputdir, "component_%s.pdf" % _comp)
            plt.savefig(outputfilename, format="pdf")
            


def baz_hist(baz, nbins):
    ax = plt.gca()
    bin_edges = np.linspace(0,360, nbins+1)
    cts, edges = np.histogram(baz, bins=bin_edges)
    xbaz = edges[:-1] + 0.5 * np.diff(edges)
    wbaz = np.diff(edges)# * 0.8
    bars = plt.bar(xbaz/180*np.pi, cts, wbaz/180*np.pi, bottom=0.0)
    for r, bar in zip(cts, bars):
        bar.set_facecolor(plt.cm.magma_r(r / np.max(cts)))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.invert_yaxis()
    ax.set_xticklabels(['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'])

def rayp_hist(rayp, nbins, v=5.8):
    # Get Incidence angle p = sin i/v <--> v sin i / p <--> i = asin(vp)
    # 5.8 taken from PREM
    shift= 180
    barshift = shift/180*np.pi
    angle = np.arcsin(rayp*v)
    ax = plt.gca()
    bin_edges = np.linspace(0,np.pi/2, nbins+1)
    cts, edges = np.histogram(angle, bins=bin_edges)
    xbaz = edges[:-1] + 0.5 * np.diff(edges)
    wbaz = np.diff(edges)# * 0.8
    bars = plt.bar(xbaz, cts, wbaz, bottom=0.0)
    bars = plt.bar(barshift-xbaz, cts, wbaz, bottom=0.0)
    for r, bar in zip(cts, bars):
        bar.set_facecolor(plt.cm.magma_r(r / np.max(cts)))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(shift-0)
    ax.set_thetamax(shift-50)
    labelvals = np.array([130, 140, 150,160, 170, 180])
    ax.set_xticklabels(["%d$^\circ$" % np.round(shift-x) for x in labelvals])
    ax.invert_yaxis()
    ax.set_rlabel_position(10)


def rfstream_baz_dist()