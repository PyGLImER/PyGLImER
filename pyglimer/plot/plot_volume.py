import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors as matcolors
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import numpy as np
from cartopy.crs import PlateCarree
import cartopy.feature as cfeature


class VolumePlot:

    def __init__(self, X, Y, Z, V,
                 xl: float or None = None,
                 yl: float or None = None,
                 zl: float or None = None):
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
        yl ( float ) : 
        Y slice location. No slice is plotted if None.
        Default None.
        zl : float
        Z slice location. No slice is plotted if None.
        Default None.
        """

        self.X = X
        self.Y = Y
        self.Z = Z
        self.V = V
        self.xl = xl
        self.yl = yl
        self.zl = zl
        self.nx = len(X)
        self.ny = len(Y)
        self.nz = len(Z)

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
        self.minV, self.maxV = np.min(V), np.max(V)
        self.buffer = 0.1  # in fraction
        self.xbuffer = (self.maxX - self.minX) * self.buffer
        self.ybuffer = (self.maxY - self.minY) * self.buffer
        self.mapextent = [self.minX - self.xbuffer, self.maxX + self.xbuffer,
                          self.minY - self.ybuffer, self.maxY + self.ybuffer]
        self.linewidth = 1.5

        # Colormap settings
        self.vmin, self.vmax = [np.min(V), np.max(V)]
        self.norm = matcolors.TwoSlopeNorm(
            vmin=self.vmin, vcenter=0.0, vmax=self.vmax)
        self.cmap = plt.cm.coolwarm
        self.mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        # First initialization
        self.get_locations()
        self.init_plot()

    def reset(self):
        pass

    def init_plot(self):

        self.fig = plt.figure(figsize=(10, 10))
        self.ax["x"] = plt.subplot(224)
        self.ax["y"] = plt.subplot(223)  # , sharey=self.axx)
        self.ax["z"] = plt.subplot(221)  # , sharex=self.axy)
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
        print(self.ax)
        self.fig.colorbar(self.mappable, shrink=1., aspect=50,
                          orientation="horizontal",
                          ax=[self.ax['z'], self.ax['m']['main'], self.ax['y'], self.ax['x']])

        plt.show()

    def plot_xsl(self):
        # self.slice['x'] = self.ax['x'].pcolor(self.Y, self.Z, 
        #                                       self.V[self.xli, :, :].T,
        #                                       cmap=self.cmap, norm=self.norm)
        self.slice['x'] = self.ax['x'].imshow(self.V[self.xli, :, :].T,
                                              extent=[self.minY, self.maxY,
                                                      self.minZ, self.maxZ],
                                              interpolation='nearest', 
                                              origin='bottom', 
                                              aspect='auto',
                                              cmap=self.cmap, norm=self.norm)
        self.ax['x'].set_xlabel("y")
        plt.setp(self.ax['x'].get_yticklabels(), visible=False)

    def plot_ysl(self):
        self.slice['y'] = self.ax['y'].pcolor(self.X, self.Z, 
                                              self.V[:, self.yli, :].T,
                                              cmap=self.cmap, norm=self.norm)
        self.ax['y'].set_xlabel("x")
        self.ax['y'].set_ylabel("z")

    def plot_zsl(self):
        self.slice['z'] = self.ax['z'].pcolor(self.X, self.Y,
                                              self.V[:, :, self.zli].T,
                                              cmap=self.cmap, norm=self.norm)
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
            lw=self.linewidth)
        self.lines['z']['x'] = self.ax['z'].plot(
            [self.xlp, self.xlp], [self.minY, self.maxY], "k",
            lw=self.linewidth)
        self.lines['m']['x'] = self.ax['m']['inset'].plot(
            [self.xlp, self.xlp], [self.minY, self.maxY], "k", 
            lw=self.linewidth)

    def plot_ylines(self):
        self.lines['x']['y'] = self.ax['x'].plot(
            [self.ylp, self.ylp], [self.minZ, self.maxZ], "k",
            lw=self.linewidth)
        self.lines['z']['y'] = self.ax['z'].plot(
            [self.minX, self.maxX], [self.ylp, self.ylp], "k",
            lw=self.linewidth)
        self.lines['m']['y'] = self.ax['m']['inset'].plot(
            [self.minX, self.maxX], [self.ylp, self.ylp], "k", 
            lw=self.linewidth)

    def plot_zlines(self):
        self.lines['x']['z'] = self.ax['x'].plot(
            [self.minY, self.maxY], [self.zlp, self.zlp], "k",
            lw=self.linewidth)
        self.lines['y']['z'] = self.ax['y'].plot(
            [self.minX, self.maxX], [self.zlp, self.zlp], "k",
            lw=self.linewidth)

    def compute_slices(self):
        if self.xli is not None:
            self.compute_xl()
        if self.yli is not None:
            self.compute_yl()
        if self.zli is not None:
            self.compute_zl()

    def update_xl(self, value):
        pass
    def update_yl(self, value):
        pass
    def update_zl(self, value):
        pass

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
    
    def __init__(self, X, Y, Z, V, xl=135, yl=37.5, zl=-50):
        """ Control Panel that controls a volume plot.

        Args:
            X (`numpy.ndarray`): X (Longitude) Offset
            Y ([type]): Y (Latitude) Offset
            Z ([type]): Depth
            V ([type]): Input Volume
            xl (int, optional): [description]. Defaults to 135.
            yl (float, optional): [description]. Defaults to 37.5.
            zl (int, optional): [description]. Defaults to -50.
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
                                            valinit=np.mean(X),
                                            valfmt='%f', valstep=np.diff(X)[0])
        self.slices['y']['ax'] = self.cp.add_subplot(self.gs[1, :])
        self.slices['y']['slider'] = Slider(self.slices['y']['ax'], r'y_l',
                                            0, self.vp.ny,
                                            valinit=int(len(Y/2)), valfmt='%d')
        self.slices['z']['ax'] = self.cp.add_subplot(self.gs[2, :])
        self.slices['z']['slider'] = Slider(self.slices['z']['ax'], r'z_l',
                                            0, self.vp.nz,
                                            valinit=int(len(Z/2)), valfmt='%d')

        self.cmap['checkbox']['ax'] = self.cp.add_subplot(self.gs[3, :])
        self.cmap['checkbox']['box'] = CheckButtons(
            self.cmap['checkbox']['ax'], ["Symmetric"], [False])
        self.cmap['pos']['ax'] = self.cp.add_subplot(self.gs[4, :])
        self.cmap['pos']['slider'] = Slider(self.cmap['pos']['ax'], r'+',
                                            0, self.vp.maxV,
                                            valinit=self.vp.maxV)
        self.cmap['neg']['ax'] = self.cp.add_subplot(self.gs[5, :])
        self.cmap['neg']['slider'] = Slider(self.cmap['neg']['ax'], r'-',
                                            self.vp.minV, 0, 
                                            valinit=self.vp.minV)
        plt.subplots_adjust(hspace=2)
        
        # Slider(slidercmin, 'ColorMin', -3 * np.max(np.abs(RF_PLOT)), 0, valinit=vmin, valfmt='%1.2E')
        # Slider(slidercmin, 'ColorMin', -3 * np.max(np.abs(RF_PLOT)), 0, valinit=vmin, valfmt='%1.2E')
        # self.cmap['z']['ax'] = self.cp.add_subplot(self.gs[6, :])

    def activate_slicers(self):
        self.slices['x']['slider'].on_changed(self.xslice)
        print("hello")

    def xslice(self, val):
        print("Hello")
        # Set new min color
        self.vp.xli = np.argmin(np.abs(self.vp.X - val))
        self.vp.xlp = self.vp.X[self.vp.xli]

        self.vp.slice['x'].set_data(self.vp.V[self.vp.xli, :, :].T)
        # Update colors
        # plt.setp(self.vp.slice['x'], 'facecolors',
        #          self.vp.cmap(self.vp.norm(self.vp.V[self.vp.xli, :, :].reshape(-1, order='C'))))
        print(self.vp.xli)
        # Update figure

        self.vp.fig.canvas.draw_idle()

    def _activate_cmap(self):
        pass
    

# Sample volume                
X = np.linspace(126.588, 152.46, 100)
Y = np.linspace(30.063, 46.862, 100)
Z = np.linspace(0, -100, 200)
YY, XX, ZZ = np.meshgrid(Y, X, Z)
V = np.sqrt(((XX - np.mean(X))/np.mean(np.abs(X)))**2
            + ((YY - np.mean(Y))/np.mean(np.abs(Y)))**2
            + ((ZZ - np.mean(Z))/np.mean(np.abs(Z)))**2) - 0.5 \
    + 0.5 * np.sin(8*np.pi*((XX - np.mean(XX))/np.mean(np.abs(XX)))) \
    + 0.5 * np.cos(8*np.pi*((YY - np.mean(YY))/np.mean(np.abs(YY))))


vp = VolumeExploration(X, Y, Z, V, xl=135, yl=37.5, zl=-50)
