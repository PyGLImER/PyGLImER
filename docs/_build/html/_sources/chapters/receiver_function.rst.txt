Handling Receiver Functions
---------------------------

Supposing that you have created receiver functions and have them available in the directory ``output/RF``,
we can now start imaging!

Handling of receiver functions is generally done by using the :class:`~pyglimer.rf.create.RFTrace` and :class:`~pyglimer.rf.create.RFStream` classes. These
classes are built upon a modified version of the `rf <https://rf.readthedocs.io/en/latest/index.html>`_ project
by Tom Eulenfeld.

.. note::  Just as in the obspy base classes, an :class:`~pyglimer.rf.create.RFStream` can hold several receiver functions,
    while an :class:`~pyglimer.rf.create.RFTrace` object can only hold one receiver function. As all rfs are saved in .sac format,
    saving :class:`~pyglimer.rf.create.RFStream` will lead to the creation of several files.


Receiver functions in .sac format can be read using the :func:`~pyglimer.rf.create.read_rf` function. By doing so, we
obtain an :class:`~pyglimer.rf.create.RFStream` object, on which we can apply a number of functions.

Methods available for both RFStream and RFTrace objects
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. hlist::
    :columns: 1

    * :func:`~pyglimer.rf.create.RFTrace.write()`
        to write receiver function(s) to SAC file(s) (in time domain).
    * :func:`~pyglimer.rf.create.RFTrace.moveout()`
        to migrate the receiver function(s) to depth domain using one of the provided depth models (either '3D' for GyPsum or 'iasp91.dat' for iasp91).
        Piercing points will be appended to the object.
    * :func:`~pyglimer.rf.create.RFTrace.ppoint()`
        to compute the piercing points in depth without migrating
        the receiver function
    * :func:`~pyglimer.rf.create.RFTrace.plot()`
        to plot the receiver function(s). The plot will be different
        depending on the type of receiver function: **1.** For `depth-migrated` RFs, the plot will be against
        depth. **2.** For an :class:`~pyglimer.rf.create.RFTrace` in time domain, the plot will be against time.
        **3.** For an :class:`~pyglimer.rf.create.RFStream` in time domain, the plot will show the receiver functions
        depending on epicentral distance.

Methods for RFStream objects
++++++++++++++++++++++++++++
.. hlist::
    :columns: 1

    * :func:`~pyglimer.rf.create.RFStream.station_stack()`
        to create a station specific stack of all receiver function in the object.
        For that to work, all RFs, have to be from the same station.