Common Conversion Point Stacking
--------------------------------

Provided that we have an extensive RF database for a certain geographic region,
we can create a volume of those receiver functions depending on piercing points
with varying grades of spatial averaging. Those volumes are referred to as Common
Conversion Point (CCP) stacks.

CCPStack objects
++++++++++++++++
In PyGLImER, the base class enabling CCP stacking is the :py:class:`~pyglimer.ccp.ccp.CCPStack` class.
However, to understand how CCP-Stacking in PyGLImER is implemented,
we will have to have a small intermezzo about binning.

CCP bins
========
In PyGLImER CCP bins are round in mapview (cylindric in 3D). If a piercing point of a receiver function is inside
of a bin, it will be added to this bin (i.e. there is one "station" stack in each bin). As such the shape of a bingrid is controlled
by two parameters: The bin's radius R and the interbin distance d.

The bin radius decides, whether a piercing point is close enough (smaller
than R) to be added to the bin. High bin radii result in strong spatial averaging, which mitigates noise and can be helpful if the
illumination is low, whereas small radii have the potential of revealing small scale details.

Concerning the bin distance, we face a tradeoff between resolution and computational efficiency. Dense grids will take significantly
longer to compute and to plot than coarse grids. However, finely gridded bingrids have a higher resolution (much like pixels in a
digital image). It should be mentioned though that the resolution that can be reached this way is limited and depending on the bin radius.
In a way, higher bin radii introduce a blur to our image, which we cannot remove by decreasing the bin distance (much like we can upscale
our digital image, but the resolving power will not increase). Therefore, PyGLImER has a threshold for the bin radius.

.. warning::
    
    The maximal bin radius allowed by PyGLImER equals 4 times the bin distance (to limit computational expense).
    The minimal bin radius one should use is cos(30deg)*bin distance. For lower values the bingrid will not cover the whole surface area.

Creating a CCP stack in PyGLImER
================================

CCP stacks in PyGLImER are usually not created by initialising the :py:class:`~pyglimer.ccp.ccp.CCPStack` class but by calling the
:py:func:`~pyglimer.ccp.ccp.init_ccp()` function, which, itself, initialises the aforementioned object.

Using :py:func:`~pyglimer.ccp.ccp.init_ccp()`, there are two ways to create a :py:class:`~pyglimer.ccp.ccp.CCPStack` object:

    1. By Creating a bingrid tailored to be used with data from predefined networks and stations (i.e. one
       has to know network and station codes for the desired stack.
    2. More commonly, one would like to create a CCP stack containing all available data for a given area.
       In order to that, :py:func:`~pyglimer.ccp.ccp.init_ccp()` can be called with the ``geocoords`` parameter.
       Then, PyGLImER will automatically find all stations available in the given area and create a bingrid around those data.

**Populating the CCP object** is done by :py:func:`~pyglimer.ccp.ccp.init_ccp()` if called using the ``compute_stack=True``
argument (recommended), else one will have to call :py:meth:`~pyglimer.ccp.ccp.CCPStack.compute_stack()`.

**Finalising the CCP object.** Up to now data was only saved in the object and we have not created an actual CCP stack, yet.
We can do just that by calling :py:meth:`~pyglimer.ccp.ccp.CCPStack.conclude_ccp()`. Here, we can decide **1.** if we want to keep
empty bins **2.** what should be the threshold for a minimum amount of data per bin **3.** Whether we want to discards bins
that are on water surfaces or keep them.

**Saving our CCP object.** The standard format to save these objects is pickle `.pkl`, which provides the fastest
reading speed. However, if you want to archive your data we recommend using the **numpy npz** format as older pickle
files can sometimes lead to compatibility issues after updating PyGLImER.
For compatibility with older plotting programs in Matlab (a legacy from GLImER), we can also save our CCP object as `.mat`.

.. warning::
    
    A ccp object saved as `.mat` will not save all data - only the finalised stack!

.. warning::

    If you want to store your CCP Stack for a longer period of time, we recommend using the *npz* format to do so.