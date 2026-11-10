'''
Plotting methods for the receiver function stream and trace objects
(implemented as mixins so that the RF creation core stays free of
plotting imports).

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import os

from matplotlib import pyplot as plt
import numpy as np

from pyglimer.constants import DEG2KM
from pyglimer.plot.plot_utils import plot_section, plot_single_rf, stream_dist


class RFStreamPlotMixin(object):
    """Plotting methods for
    :class:`~pyglimer.rf.create.RFStream`. Only to be used as a mixin."""

    def plot(
        self, channel: str = "PRF", lim: list or tuple or None = None,
        epilimits: list or tuple or None = None, scalingfactor: float = 2.0,
        ax: plt.Axes = None, line: bool = True, linewidth: float = 0.25,
        outputdir: str or None = None, title: str or None = None,
        show: bool = True, format: str = 'pdf', color: str = 'seismic',
            bold: bool = False):
        """Creates plot of a receiver function section as a function
        of epicentral distance or single plot if len(RFStream)==1.

        Parameters
        ----------
        lim : list or tuple or None
            y axis time limits in seconds (if self.stats.type==time)
            or depth in km (if self.stats==depth) (len(list)==2).
            If `None` full traces is plotted.
            Default None.
        epilimits : list or tuple or None = None,
            y axis time limits in seconds (len(list)==2).
            If `None` from 30 to 90 degrees plotted.
            Default None.
        scalingfactor : float
            sets the scale for the traces. Could be automated in
            future functions(Something like mean distance between
            traces)
            Defaults to 2.0
        line : bool
            plots black line of the actual RF
            Defaults to True
        linewidth: float
            sets linewidth of individual traces
        ax : `matplotlib.pyplot.Axes`, optional
            Can define an axes to plot the RF into. Defaults to None.
            If None, new figure is created.
        outputdir : str, optional
            If set, saves a pdf of the plot to the directory.
            If None, plot will be shown instantly. Defaults to None.
        clean: bool
            If True, clears out all axes and plots RF only.
            Defaults to False.
        color: str, optional
            Color-scale to use. Options are 'seismic', 'pyglimer', or 'mono'.
            Defaults to 'seismic'.
        bold: bool, optional
            Print titles and labels larger

        Returns
        -------
        ax : `matplotlib.pyplot.Axes`

        """
        if self.count() == 1:
            # Do single plot
            ax = plot_single_rf(
                self[0], tlim=lim, ax=ax, outputdir=outputdir, format=format,
                color=color, bold=bold)
        else:
            if outputdir:
                outputfile = os.path.join(outputdir, '%s%s' % (
                    self[0].stats.network, self[0].stats.station
                ))
            else:
                outputfile = None
            ax = plot_section(
                self, timelimits=lim, epilimits=epilimits,
                scalingfactor=scalingfactor, line=line, linewidth=linewidth,
                ax=ax, outputfile=outputfile, channel=channel, format=format,
                color=color, bold=bold)
        return ax

    def plot_distribution(
        self, nbins=50, phase="P", outputfile=None, format="pdf", dpi=300,
            title: str = None, bold: bool = False):
        """
        Plot back azimuth and rayparameter distributions.

        Parameters
        ----------
        nbins : int
            Number of bins. Default
        v : float
            assummed surface velocity for the computation of the
            incidence angle. Default 5.8 km/s.
        outputfile : str, optional
            Path to savefile. If None plot is not saved just shown.
            Defaults to None.
        format : str, optional
            outputfile format
        dpi : int, optional
            only used if file format is none vector.
        bold: bool, optional
            Print titles and labels larger


        Returns
        -------
        None

        """

        # Define velocity depending on the incident phase
        if phase[-1] == "P":
            v = 5.793  # PREM vp velocity at the surface
        else:
            v = 3.191  # PREM vs velocity at the surface

        # Get rayparameter and backazimuth from stream.
        rayp = []
        baz = []

        for rf in self:
            if rf.stats.phase != phase:
                continue
            rayp.append(rf.stats.slowness / DEG2KM)
            baz.append(rf.stats.back_azimuth)

        if not len(baz):
            raise ValueError(
                'No Receiver Functions of Phase %s found,\
                    did you choose the right phase?' % phase)

        stream_dist(
            np.array(rayp), np.array(baz), nbins=nbins, v=v,
            outputfile=outputfile, format=format,  dpi=dpi, title=title,
            bold=bold)


class RFTracePlotMixin(object):
    """Plotting methods for
    :class:`~pyglimer.rf.create.RFTrace`. Only to be used as a mixin."""

    def plot(
        self, lim: list or tuple or None = None,
        depth: np.ndarray or None = None, ax: plt.Axes = None,
        outputdir: str = None, format: str = 'pdf', clean: bool = False,
        std: np.ndarray = None, flipxy: bool = False,
            color: str = 'seismic', bold: bool = False):
        """Creates plot of a single receiver function

        Parameters
        ----------
        lim: list or tuple or None
            x axis time limits in seconds or km (depth) (len(list)==2).
            If `None` full trace is plotted.
            Default None.
        depth: :class:`numpy.ndarray`
            1D array of depths
        ax : `matplotlib.pyplot.Axes`, optional
            Can define an axes to plot the RF into. Defaults to None.
            If None, new figure is created.
        outputdir : str, optional
            If set, saves a pdf of the plot to the directory.
            If None, plot will be shown instantly. Defaults to None.
        clean: bool
            If True, clears out all axes and plots RF only.
            Defaults to False.
        std: np.ndarray, optional
            **Only if self.type == stastack**. Plots the upper and lower
            limit of the standard deviation in the plot. Provide the std
            as a numpy array (can be easily computed from the output of
            :meth:`~pyglimer.rf.create.RFStream.bootstrap`)
        flipxy: bool, optional
            Plot Depth/Time on the Y-Axis and amplitude on the x-axis. Defaults
            to False.
        color: str, optional
            Color-scale to use. Options are 'seismic', 'pyglimer', or 'mono'.
            Defaults to 'seismic'.
        bold: bool, optional
            Print titles and labels larger

        Returns
        -------
        ax : `matplotlib.pyplot.Axes`
        """
        ax = plot_single_rf(
            self, lim, depth=depth, ax=ax, outputdir=outputdir, clean=clean,
            format=format, std=std, flipxy=flipxy, color=color, bold=bold)
        return ax
