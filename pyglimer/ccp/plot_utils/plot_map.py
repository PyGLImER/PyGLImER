"""

Simple map plot to be use by other

"""
import cartopy
from cartopy.crs import PlateCarree


def plot_map(ax):
    ax.set_global()
    ax.frameon = True
    # ax.outline_patch.set_visible(False)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=PlateCarree(), draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = True

    # Change fontsize
    fontsize = 12
    font_dict = {"fontsize": fontsize,
                 "weight": "bold"}
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=font_dict)

    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black',
                   facecolor=(0.85, 0.85, 0.85))
