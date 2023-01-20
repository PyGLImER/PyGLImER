"""

Simple map plot to be use by other. This might be obsolete

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Lucas Sawade (lsawade@princeton.edu)

"""
import cartopy
from cartopy.crs import PlateCarree


def plot_map(ax):
    # ax.set_global()
    ax.frameon = True
    # ax.outline_patch.set_visible(False)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=PlateCarree(), draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = True

    # Change fontsize
    # font_dict = {"fontsize": "small"}
    # ax.set_xticklabels(ax.get_xticklabels(), fontdict=font_dict)
    # ax.set_yticklabels(ax.get_yticklabels(), fontdict=font_dict)

    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black',
                   facecolor=(0.85, 0.85, 0.85))
