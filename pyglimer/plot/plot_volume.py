# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as matcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import numpy as np
from cartopy.crs import PlateCarree
import cartopy.feature as cfeature

class slice:
    
    def __init__(self, X, Y, Z, C):
    
        if ((X.shape != Y.shape)
            or (Y.shape != Z.shape) 
            or (Z.shape != C.shape) 
            or (C.shape != X.shape)):
            print("X", X.shape)
            print("Y", Y.shape)
            print("Z", Z.shape)
            print("C", C.shape)
            raise ValueError("Shapes not equal.")
        self.X = X
        self.Y = Y
        self.Z = Z
        self.C = C
        self.pl = None
    """
    def plot(self, cmap="coolwarm", norm=None, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.pcolor(self.Y, self.Z, scmap=self.cmap, norm=self.norm)
    """

class volume_plot:

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
        self.axx = None
        self.axy = None
        self.axz = None
        self.axtopright = None
        self.mapax = None
        self.fig = None
        self.view = None
        self.minX, self.maxX = np.min(X), np.max(X)
        self.minY, self.maxY = np.min(Y), np.max(Y)
        self.minZ, self.maxZ = np.min(Z), np.max(Z)
        self.buffer = 0.1 # in fraction
        self.xbuffer = (self.maxX - self.minX) * self.buffer
        self.ybuffer = (self.maxY - self.minY) * self.buffer
        self.mapextent = [self.minX - self.xbuffer, self.maxX + self.xbuffer,
                          self.minY - self.ybuffer, self.maxY + self.ybuffer]
        self.linewidth = 1.5

        # Colormap settings
        self.vmin, self.vmax = [np.min(V), np.max(V)];
        self.norm = matcolors.Normalize(self.vmin, self.vmax);
        self.cmap = plt.cm.coolwarm
        self.mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        # First initialization
        self.get_locations()
        self.init_plot()

    def reset(self):
        pass

    def init_plot(self):

        self.compute_slices()

        self.fig = plt.figure(figsize=(10, 10))
        self.axx = plt.subplot(224)
        self.axtopright = plt.subplot(222)
        self.axy = plt.subplot(223)  #, sharey=self.axx)
        self.axz = plt.subplot(221)  #, sharex=self.axy)

        self.plot_xsl()
        self.plot_ysl()
        self.plot_zsl()
        self.plot_map()
        self.plot_xlines()
        self.plot_ylines()
        self.plot_zlines()

        self.fig.subplots_adjust(hspace=0.0, wspace=0.0)
        self.fig.colorbar(self.mappable, shrink=1., aspect=50,
                          orientation="horizontal",
                          ax=[self.axz, self.axtopright, self.axy, self.axx])

        plt.show(block=False)

    def plot_xsl(self):
        self.axx.pcolor(self.Y, self.Z, self.V[self.xli, :, :].T,
                        cmap=self.cmap, norm=self.norm)
        self.axx.set_xlabel("y")
        plt.setp(self.axx.get_yticklabels(), visible=False)

    def plot_ysl(self):
        self.axy.pcolor(self.X, self.Z, self.V[:, self.yli, :].T,
                        cmap=self.cmap, norm=self.norm)
        self.axy.set_xlabel("x")
        self.axy.set_ylabel("z")

    def plot_zsl(self):
        self.axz.pcolor(self.X, self.Y, self.V[:, :, self.zli].T,
                        cmap=self.cmap, norm=self.norm)
        self.axz.set_ylabel("y")
        plt.setp(self.axz.get_xticklabels(), visible=False)

    def plot_map(self):
        self.axtopright.axis("off")
        # inset location relative to main plot (ax) in normalized units
        inset_x = 0.5
        inset_y = 0.5
        inset_size = 0.8
        inset_dim = [inset_x - inset_size / 2,
                     inset_y - inset_size / 2,
                     inset_size, inset_size]

        self.mapax = plt.axes([0, 0, 1, 1], projection=PlateCarree())
        self.mapax.add_feature(cfeature.LAND)
        self.mapax.add_feature(cfeature.OCEAN)
        self.mapax.add_feature(cfeature.COASTLINE)
        ip = InsetPosition(self.axtopright, inset_dim)
        self.mapax.set_axes_locator(ip)
        self.mapax.set_extent(self.mapextent)

    def plot_xlines(self):
        xliney = self.axy.plot([self.xlp, self.xlp], [self.minZ, self.maxZ], "k",
                               lw=self.linewidth)
        xlinez = self.axz.plot([self.xlp, self.xlp], [self.minY, self.maxY], "k",
                               lw=self.linewidth)
        xlinem = self.mapax.plot([self.xlp, self.xlp], [self.minY, self.maxY], "k",
                                 lw=self.linewidth)

    def plot_ylines(self):
        ylinex = self.axx.plot([self.ylp, self.ylp], [self.minZ, self.maxZ], "k",
                               lw=self.linewidth)
        ylinez = self.axz.plot([self.minX, self.maxX], [self.ylp, self.ylp], "k",
                               lw=self.linewidth)
        ylinem = self.mapax.plot([self.minX, self.maxX], [self.ylp, self.ylp], "k",
                                 lw=self.linewidth)

    def plot_zlines(self):
        zlinex = self.axx.plot([self.minY, self.maxY], [self.zlp, self.zlp], "k",
                               lw=self.linewidth)
        zliney = self.axy.plot([self.minX, self.maxX], [self.zlp, self.zlp], "k",
                               lw=self.linewidth)

    def compute_slices(self):
        if self.xli is not None:
            self.compute_xl()
        if self.yli is not None:
            self.compute_yl()
        if self.zli is not None:
            self.compute_zl()

    def compute_xl(self):
        # Get coords
        YY, ZZ = np.meshgrid(self.Y, self.Z)
        XX = self.X[self.xli] * np.ones_like(YY)
        self.xsl = slice(XX, YY, ZZ, self.V[self.xli, :, :].T)

    def compute_yl(self):
        # Get coords
        XX, ZZ = np.meshgrid(self.X, self.Z)
        YY = self.Y[self.yli] * np.ones_like(XX)
        self.ysl = slice(XX, YY, ZZ, self.V[:, self.yli, :].T)

    def compute_zl(self):
        # Get coords
        XX, YY = np.meshgrid(self.X, self.Y)
        ZZ = self.Z[self.zli] * np.ones_like(XX)
        self.zsl = slice(XX, YY, ZZ, self.V[:, :, self.xli].T)

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

    def view(self, elevation, angle):
        self.vax.view_init(elevation, angle)



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


vp = volume_plot(X, Y, Z, V,
                 xl=135, yl=37.5, zl=-50)
