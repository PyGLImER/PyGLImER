"""

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Lucas Sawade (lsawade@princeton.edu)

Last Update: June 19, 2020

"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors as matcolors
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import numpy as np
from cartopy.crs import PlateCarree
import cartopy.feature as cfeature
from .ui_utils import set_sliderval_no_callback


class VolumePlot:

    def __init__(self, X, Y, Z, V,
                 xl: float or None = None,
                 yl: float or None = None,
                 zl: float or None = None,
                 nancolor='w', show=True,
                 cmap='seismic'):
        """

        Parameters:
        -----------
        X : `numpy.ndarray`
            1d vector with x data
        Y : `numpy.ndarray`
            1d vector with y data
        Z : `numpy.ndarray`
            1d vector with z data
        V : `numpy.ndarray`
            3D array with volumetric data
        xl : float
            X slice location. No slice is plotted if None.
            Default None.
        yl : float
            Y slice location. No slice is plotted if None.
            Default None.
        zl : float
            Z slice location. No slice is plotted if None.
            Default None.

        Notes
        -----

        :Authors:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.04.21 20.00 (Lucas Sawade)

        """

        # Input allocation
        self.X = X
        self.Y = Y
        self.Z = Z
        self.V = V
        self.nx = len(X)
        self.ny = len(Y)
        self.nz = len(Z)

        if xl is None:
            self.xl = np.mean(X)
        else:
            self.xl = xl
        if yl is None:
            self.yl = np.mean(Y)
        else:
            self.yl = yl
        if zl is None:
            self.zl = np.mean(Z)
        else:
            self.zl = zl

        # For reset
        self.xl0 = xl
        self.yl0 = yl
        self.zl0 = zl

        # Allocate stuff that is going to be used.
        # Slices
        self.xsl = None
        self.ysl = None
        self.zsl = None

        # Cross section lines
        self.xlines = None
        self.ylines = None
        self.zlines = None

        # Slice index in Respective vector
        self.xli = None
        self.yli = None
        self.zli = None
        self.xlp = None
        self.ylp = None
        self.zlp = None

        # Figure stuff
        self.ax = {'x': None,
                   'y': None,
                   'z': None,
                   'm': {'main': None,
                         'inset': None}}
        self.slice = {'x': None,
                      'y': None,
                      'z': None}
        self.lines = {'x': {'y': None, 'z': None},
                      'y': {'x': None, 'z': None},
                      'z': {'x': None, 'y': None},
                      'm': {'x': None, 'y': None}}
        self.axtopright = None
        self.mapax = None
        self.fig = None
        self.minX, self.maxX = np.min(X), np.max(X)
        self.minY, self.maxY = np.min(Y), np.max(Y)
        self.minZ, self.maxZ = np.min(Z), np.max(Z)
        self.minV, self.maxV = np.nanmin(V), np.nanmax(V)
        self.buffer = 0.1  # in fraction
        self.xbuffer = (self.maxX - self.minX) * self.buffer
        self.ybuffer = (self.maxY - self.minY) * self.buffer
        self.mapextent = [self.minX - self.xbuffer, self.maxX + self.xbuffer,
                          self.minY - self.ybuffer, self.maxY + self.ybuffer]
        self.linewidth = 1.5

        # Colormap settings
        self.vmin0, self.vmax0 = [-0.05, 0.05]
        # Due to the interpolation the above valuers are not easily
        # identifiable by distribution
        self.vmin, self.vmax = self.vmin0, self.vmax0
        self.norm = matcolors.TwoSlopeNorm(
            vmin=self.vmin, vcenter=0.0, vmax=self.vmax)
        self.cmapname = cmap
        self.nancolor = nancolor
        self.cmap = plt.get_cmap(self.cmapname)
        self.cmap.set_bad(color=self.nancolor)
        self.mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.cbar = None

        # First initialization
        self.show = show
        self.get_locations()
        self.init_plot()

    def reset(self):

        # Colorbar reset
        self.norm = matcolors.TwoSlopeNorm(
            vmin=self.vmin0, vcenter=0.0, vmax=self.vmax0)
        self.cmap = plt.get_cmap(self.cmapname)
        self.cmap.set_bad(color=self.nancolor)
        self.mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        # Update slices and lines
        self.update_xslice(self.xl0)
        self.update_yslice(self.yl0)
        self.update_zslice(self.zl0)

    def init_plot(self):

        self.fig = plt.figure(figsize=(8, 6))
        self.ax["x"] = plt.subplot(224)
        self.ax["y"] = plt.subplot(223)  # , sharey=self.ax['x'])
        self.ax["z"] = plt.subplot(221)  # , sharex=self.ax['y'])
        self.ax["m"].update({"main": plt.subplot(222),
                             "inset": None})

        self.plot_xsl()
        self.plot_ysl()
        self.plot_zsl()
        self.plot_map()
        self.plot_xlines()
        self.plot_ylines()
        self.plot_zlines()

        self.fig.subplots_adjust(hspace=0.0, wspace=0.0)
        self.cbar = self.fig.colorbar(
            self.mappable, shrink=1., aspect=50, orientation="horizontal",
            ax=[self.ax['z'], self.ax['m']['main'],
                self.ax['y'], self.ax['x']])
        if self.show:
            plt.show()

    def plot_xsl(self):
        self.slice['x'] = self.ax['x'].imshow(
            self.V[self.xli, :, :].T,
            extent=[self.minY, self.maxY, self.minZ, self.maxZ],
            interpolation='nearest', origin='lower', aspect='auto',
            cmap=self.cmap, norm=self.norm, rasterized=True)
        self.ax['x'].set_xlabel("y")
        plt.setp(self.ax['x'].get_yticklabels(), visible=False)
        self.ax['x'].invert_yaxis()

    def plot_ysl(self):
        self.slice['y'] = self.ax['y'].imshow(
            self.V[:, self.yli, :].T,
            extent=[self.minX, self.maxX, self.minZ, self.maxZ],
            interpolation='nearest', origin='lower', aspect='auto',
            cmap=self.cmap, norm=self.norm, rasterized=True)
        self.ax['y'].set_xlabel("x")
        self.ax['y'].set_ylabel("z")
        self.ax['y'].invert_yaxis()

    def plot_zsl(self):
        self.slice['z'] = self.ax['z'].imshow(
            self.V[:, :, self.zli].T,
            extent=[self.minX, self.maxX, self.minY, self.maxY],
            interpolation='nearest', origin='lower', aspect='auto',
            cmap=self.cmap, norm=self.norm, rasterized=True)
        self.ax['z'].set_ylabel("y")
        plt.setp(self.ax['z'].get_xticklabels(), visible=False)

    def plot_map(self):
        self.ax['m']['main'].axis("off")
        # inset location relative to main plot (ax) in normalized units
        inset_x = 0.5
        inset_y = 0.5
        inset_size = 0.8
        inset_dim = [inset_x - inset_size / 2,
                     inset_y - inset_size / 2,
                     inset_size, inset_size]

        self.ax['m']['inset'] = plt.axes([0, 0, 1, 1],
                                         projection=PlateCarree())
        self.ax['m']['inset'].add_feature(cfeature.LAND)
        self.ax['m']['inset'].add_feature(cfeature.OCEAN)
        self.ax['m']['inset'].add_feature(cfeature.COASTLINE)
        ip = InsetPosition(self.ax['m']['main'], inset_dim)
        self.ax['m']['inset'].set_axes_locator(ip)
        self.ax['m']['inset'].set_extent(self.mapextent)

    def plot_xlines(self):
        self.lines['y']['x'] = self.ax['y'].plot(
            [self.xlp, self.xlp], [self.minZ, self.maxZ], "k",
            lw=self.linewidth)[0]
        self.lines['z']['x'] = self.ax['z'].plot(
            [self.xlp, self.xlp], [self.minY, self.maxY], "k",
            lw=self.linewidth)[0]
        self.lines['m']['x'] = self.ax['m']['inset'].plot(
            [self.xlp, self.xlp], [self.minY, self.maxY], "k",
            lw=self.linewidth)[0]

    def plot_ylines(self):
        self.lines['x']['y'] = self.ax['x'].plot(
            [self.ylp, self.ylp], [self.minZ, self.maxZ], "k",
            lw=self.linewidth)[0]
        self.lines['z']['y'] = self.ax['z'].plot(
            [self.minX, self.maxX], [self.ylp, self.ylp], "k",
            lw=self.linewidth)[0]
        self.lines['m']['y'] = self.ax['m']['inset'].plot(
            [self.minX, self.maxX], [self.ylp, self.ylp], "k",
            lw=self.linewidth)[0]

    def plot_zlines(self):
        self.lines['x']['z'] = self.ax['x'].plot(
            [self.minY, self.maxY], [self.zlp, self.zlp], "k",
            lw=self.linewidth)[0]
        self.lines['y']['z'] = self.ax['y'].plot(
            [self.minX, self.maxX], [self.zlp, self.zlp], "k",
            lw=self.linewidth)[0]

    def update_xslice(self, val):
        # Get Slice location
        self.xli = np.argmin(np.abs(self.X - val))
        self.xlp = self.X[self.xli]

        # Set Color
        self.slice['x'].set_data(self.V[self.xli, :, :].T)

        # Update line locations
        self.lines['y']['x'].set_xdata([self.xlp, self.xlp])
        self.lines['z']['x'].set_xdata([self.xlp, self.xlp])
        self.lines['m']['x'].set_xdata([self.xlp, self.xlp])

        # Set lines on the other slices and the map.
        self.fig.canvas.draw_idle()

    def update_yslice(self, val):
        # Get Slice location
        self.yli = np.argmin(np.abs(self.Y - val))
        self.ylp = self.Y[self.yli]

        # Set Color
        self.slice['y'].set_data(self.V[:, self.yli, :].T)

        # Update line locations
        self.lines['x']['y'].set_xdata([self.ylp, self.ylp])
        self.lines['z']['y'].set_ydata([self.ylp, self.ylp])
        self.lines['m']['y'].set_ydata([self.ylp, self.ylp])

        # Set lines on the other slices and the map.
        self.fig.canvas.draw_idle()

    def update_zslice(self, val):
        # Get Slice location
        self.zli = np.argmin(np.abs(self.Z - val))
        self.zlp = self.Z[self.zli]

        # Set Color
        self.slice['z'].set_data(self.V[:, :, self.zli].T)

        # Update line locations
        self.lines['x']['z'].set_ydata([self.zlp, self.zlp])
        self.lines['y']['z'].set_ydata([self.zlp, self.zlp])

        # Set lines on the other slices and the map.
        self.fig.canvas.draw_idle()

    def update_cmap(self, vmin, vcenter, vmax):

        # Colormap settings
        self.vmin, self.vmax = [vmin, vmax]
        self.norm = matcolors.TwoSlopeNorm(
            vmin=vmin, vcenter=vcenter, vmax=vmax)
        self.cmap = plt.cm.coolwarm
        self.mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        self.slice['x'].norm = self.norm
        self.slice['x'].set_data(self.V[self.xli, :, :].T)
        self.slice['y'].norm = self.norm
        self.slice['y'].set_data(self.V[:, self.yli, :].T)
        self.slice['z'].norm = self.norm
        self.slice['z'].set_data(self.V[:, :, self.zli].T)
        self.fig.canvas.draw_idle()

    def get_locations(self):

        if self.xl is not None:
            self.xli = np.argmin(np.abs(self.X - self.xl))
            self.xlp = self.X[self.xli]
        if self.yl is not None:
            self.yli = np.argmin(np.abs(self.Y - self.yl))
            self.ylp = self.Y[self.yli]
        if self.zl is not None:
            self.zli = np.argmin(np.abs(self.Z - self.zl))
            self.zlp = self.Z[self.zli]


class VolumeExploration:

    def __init__(self, X, Y, Z, V, xl=None, yl=None, zl=None):
        """ Control Panel that controls a volume plot defined above.

        Parameters
        ----------
        X : Arraylike
            X (Longitude) Offset
        Y : Arraylike
            Y (Latitude) Offset
        Z : Arraylike
            Depth
        V : Arraylike
            Input Volume
        xl : float, optional
            Slice location. Defaults to 135.
        yl :float, optional
            slice location in y direction. Defaults to 37.5.
        zl : float, optional
            zslice location. Defaults to -50.

        Notes
        -----

        :Authors:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.04.21 20.00 (Lucas Sawade)

        """
        plt.ion()

        self.vp = VolumePlot(X, Y, Z, V, xl=xl, yl=yl, zl=zl)
        self.max = np.max(abs(V))

        self.slices = {
            'x': {'slider': None, 'ax': None},
            'y': {'slider': None, 'ax': None},
            'z': {'slider': None, 'ax': None}
        }

        self.cmap = {
            'pos': {'slider': None, 'ax': None},
            'neg': {'slider': None, 'ax': None},
            'checkbox': {'box': None, 'ax': None},
        }

        self.controlpanel()
        self.activate_slicers()
        print(self.vp.V[self.vp.xli, :, :].shape)
        plt.show(block=True)

    def controlpanel(self):

        self.cp = plt.figure(figsize=(5, 10))
        self.gs = gridspec.GridSpec(ncols=1, nrows=10)

        self.slices['x']['ax'] = self.cp.add_subplot(self.gs[0, :])
        self.slices['x']['slider'] = Slider(self.slices['x']['ax'], r'x_l',
                                            self.vp.minX, self.vp.maxX,
                                            valinit=np.mean(self.vp.X),
                                            valfmt='%f',
                                            valstep=np.diff(self.vp.X)[0])
        self.slices['y']['ax'] = self.cp.add_subplot(self.gs[1, :])
        self.slices['y']['slider'] = Slider(self.slices['y']['ax'], r'y_l',
                                            self.vp.minY, self.vp.maxY,
                                            valinit=np.mean(self.vp.Y),
                                            valfmt='%d',
                                            valstep=np.diff(self.vp.Y)[0])

        self.slices['z']['ax'] = self.cp.add_subplot(self.gs[2, :])
        self.slices['z']['slider'] = Slider(self.slices['z']['ax'], r'y_l',
                                            self.vp.minZ, self.vp.maxZ,
                                            valinit=np.mean(self.vp.Z),
                                            valfmt='%d',
                                            valstep=np.diff(self.vp.Z)[0])

        self.cmap['checkbox']['ax'] = self.cp.add_subplot(self.gs[3, :])
        self.cmap['checkbox']['box'] = CheckButtons(
            self.cmap['checkbox']['ax'], ["Symmetric"], [False])
        self.cmap['pos']['ax'] = self.cp.add_subplot(self.gs[4, :])
        self.cmap['pos']['slider'] = Slider(self.cmap['pos']['ax'], r'+',
                                            1e-10, self.vp.maxV,
                                            valinit=self.vp.maxV)
        self.cmap['neg']['ax'] = self.cp.add_subplot(self.gs[5, :])
        self.cmap['neg']['slider'] = Slider(self.cmap['neg']['ax'], r'-',
                                            self.vp.minV, -1e-10,
                                            valinit=self.vp.minV)
        plt.subplots_adjust(hspace=2)

        # Slider(slidercmin, 'ColorMin', -3 * np.max(np.abs(RF_PLOT)),
        # 0, valinit=vmin, valfmt='%1.2E')
        # Slider(slidercmin, 'ColorMin', -3 * np.max(np.abs(RF_PLOT)),
        # 0, valinit=vmin, valfmt='%1.2E')
        # self.cmap['z']['ax'] = self.cp.add_subplot(self.gs[6, :])

    def activate_slicers(self):
        self.slices['x']['slider'].on_changed(self.vp.update_xslice)
        self.slices['y']['slider'].on_changed(self.vp.update_yslice)
        self.slices['z']['slider'].on_changed(self.vp.update_zslice)
        self.cmap['pos']['slider'].on_changed(self.update_pos)
        self.cmap['neg']['slider'].on_changed(self.update_neg)
        self.cmap['checkbox']['box'].on_clicked(self.update_sym)

    def update_pos(self, val):
        if self.cmap['checkbox']['box'].get_status()[0]:
            vmin = -val
            set_sliderval_no_callback(self.cmap['neg']['slider'], vmin)
        else:
            vmin = self.vp.vmin
        vmax = val
        vcenter = 0
        self.vp.update_cmap(vmin, vcenter, vmax)

    def update_neg(self, val):
        if self.cmap['checkbox']['box'].get_status()[0]:
            vmax = np.abs(val)
            set_sliderval_no_callback(self.cmap['pos']['slider'], vmax)
        else:
            vmax = self.vp.vmax
        vmin = val
        vcenter = 0
        self.vp.update_cmap(vmin, vcenter, vmax)

    def update_sym(self, val):
        if val[0]:
            val = np.mean([np.abs(self.vp.vmin), np.abs(self.vp.vmax)])
            vmin = -val
            vmax = val
            self.cmap['pos']['slider'].set_val(vmax)
            self.cmap['neg']['slider'].set_val(vmin)

# Sample volume
# X = np.linspace(126.588, 152.46, 100)
# Y = np.linspace(30.063, 46.862, 100)
# Z = np.linspace(0, -100, 200)
# YY, XX, ZZ = np.meshgrid(Y, X, Z)
# V = np.sqrt(((XX - np.mean(X))/np.mean(np.abs(X)))**2
#             + ((YY - np.mean(Y))/np.mean(np.abs(Y)))**2
#             + ((ZZ - np.mean(Z))/np.mean(np.abs(Z)))**2) - 0.5 \
#     + 0.5 * np.sin(8*np.pi*((XX - np.mean(XX))/np.mean(np.abs(XX)))) \
#     + 0.5 * np.cos(8*np.pi*((YY - np.mean(YY))/np.mean(np.abs(YY))))


# vp = VolumeExploration(X, Y, Z, V, xl=135, yl=37.5, zl=-50)
