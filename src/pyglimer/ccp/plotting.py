'''
Plotting functionality for the CCP stack: phase picking and the plotting
methods of :class:`~pyglimer.ccp.ccp.CCPStack` (implemented as a mixin so
that the stacking core stays free of plotting imports).

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import numpy as np

from pyglimer.ccp.plot_utils.plot_bins import plot_bins
from pyglimer.ccp.plot_utils.plot_cross_section import plot_cross_section
from pyglimer.plot.plot_map import plot_map_ccp, plot_vel_grad


def _pad_extent(coord):
    """Floor/ceil a coordinate sequence to integer-padded map bounds."""
    return (np.floor(min(coord)) - 1, np.ceil(max(coord)) + 1)


class PhasePick(object):
    """
    A phasepick object, just created for more convenient plotting.
    """

    def __init__(
        self, coords: np.ndarray, amplitudes: np.ndarray, polarity: str,
            z: np.ndarray, depthrange: list = [None, None]):
        """
        Initialise object

        Parameters
        ----------
        coords : np.ndarray
            2-d array containing latitudes and longitudes.
        amplitudes : np.ndarray
            1-D array containg amplitude values of each bin
        polarity : str
            Either '+' for positive velocity gradients or '-' for negative
        z : np.ndarray
            1D array containing depth of maxima/minima per bin
        depthrange : list, optional
            list containing depth restrictions in form of [zmin, zmax],
            by default [None, None]
        """
        # assign vars
        self.coords = coords
        self.a = amplitudes
        self.z = z
        self.pol = polarity
        self.depthr = depthrange

    def plot(
        self, plot_amplitude: bool = False, outputfile: str or None = None,
        format: str = 'pdf', dpi: int = 300, cmap: str = 'gist_rainbow',
            geology: bool = False, title: str or None = None):
        """
        Plot heatmap containing depth or amplitude of picked phase.

        Parameters
        ----------
        plot_amplitude : bool, optional
            If True amplitude instead of depths is plotted, by default False
        outputfile : str or None, optional
            Write Figure to file, by default None
        format : str, optional
            File format, by default 'pdf'
        dpi : int, optional
            Resolution for non-vector graphics, by default 300
        cmap : str, optional
            Colormap
        geology : bool, optional
            Plot geological map.
        title
        """
        lat = _pad_extent(self.coords[0][0])
        lon = _pad_extent(self.coords[1][0])

        plot_vel_grad(
            self.coords, self.a, self.z, plot_amplitude, lat, lon, outputfile,
            dpi=dpi, format=format, cmap=cmap, geology=geology, title=title)


class CCPStackPlotMixin(object):
    """Plotting methods for the
    :class:`~pyglimer.ccp.ccp.CCPStack`. Only to be used as a mixin."""

    def plot_cross_section(self, *args, **kwargs):
        """
        See documentation for
        :func:`~pyglimer.ccp.plot_utils.plot_cross_section.plot_cross_section`
        """
        return plot_cross_section(self, *args, **kwargs)

    def plot_bins(self):
        """
        A simple map view of the bins.
        """
        if hasattr(self, 'coords_new'):
            coords = self.coords_new
        else:
            coords = self.coords
        plot_bins(self.bingrid.stations, coords)

    def map_plot(
        self, plot_stations: bool = False, plot_bins: bool = False,
        plot_illum: bool = False, profile: list or tuple or None = None,
        p_direct: bool = True, outputfile: str or None = None,
        format: str = 'pdf', dpi: int = 300, geology: bool = False,
            title: str or None = None):
        """
        Create a map plot of the CCP Stack containing user-defined information.

        Parameters
        ----------
        plot_stations : bool, optional
            Plot the station locations, by default False
        plot_bins : bool, optional
            Plot bin location, by default False
        plot_illum : bool, optional
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
        if hasattr(self, 'coords_new') and self.coords_new[0].size:
            bincoords = self.coords_new
        else:
            bincoords = self.coords
        # Set boundaries for map:
        lat = _pad_extent(bincoords[0][0])
        lon = _pad_extent(bincoords[1][0])
        plot_map_ccp(
            lat, lon, plot_stations, self.bingrid.latitude,
            self.bingrid.longitude, plot_bins, bincoords, self.bingrid.edist,
            plot_illum, self.hits, profile, p_direct, outputfile=outputfile,
            format=format, dpi=dpi, geology=geology, title=title)

    def pick_phase(
            self, pol: str = '+', depth: list = [None, None]) -> PhasePick:
        """
        Pick the strongest negative or strongest positive gradient from a
        predefined depth-range.

        Parameters
        ----------
        pol : str, optional
            Either '+' for positive velocity gradient or '-' for negative
            gradient, by default '+'
        depth : list, optional
            List with two elements [minz, maxz], defines in which depth window
            the phase picker is gonna look for maxima and minima,
            by default [None, None]

        Returns
        -------
        PhasePick
            A phasepick object, which subsequently can be plotted.

        Raises
        ------
        ValueError
            For errandeous inputs.
        """
        self.conclude_ccp(r=3)
        # Find indices
        for ii, d in enumerate(depth):
            if d or d == 0:
                depth[ii] = np.abs(self.z-d).argmin()
        # Find minimum in vector
        ccp_short = self.ccp[:, depth[0]:depth[1]]
        # There should be a line here excluding insufficiently illuminated bins
        illum_flat = np.sum(self.hits[:, depth[0]:depth[1]], axis=1)
        id = np.where(illum_flat < ccp_short.shape[1]*10)  # delete those bins

        ccp_short = np.delete(ccp_short, id, 0)
        coords = (
            np.delete(self.coords_new[0], id, 1),
            np.delete(self.coords_new[1], id, 1))

        if pol == '+':
            # Amplitudes
            a = ccp_short.max(axis=1)
            # Depths
            z = self.z[depth[0]:depth[1]][ccp_short.argmax(axis=1)]
        elif pol == '-':
            a = ccp_short.min(axis=1)
            z = self.z[depth[0]:depth[1]][ccp_short.argmin(axis=1)]
        else:
            raise ValueError(
                'Choose either \'+\' to return the highest '
                + 'positive velocity gradient or \'-\' to return the '
                + 'highest negative velocity gradient.')

        p_phase = PhasePick(coords, a, pol, z, depth)
        return p_phase
