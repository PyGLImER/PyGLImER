Methods available for both RFStream and RFTrace objects
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

No matter whether your data was stored as *mseed* or *hdf5*, after reading it,
you will receive an :class:`~pyglimer.rf.create.RFStream` object. Below you can
find a list of the available public methods. For examples on how to use them,
please consult the example Jupyter notebooks.

Methods for RFTrace objects
############################

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.rf.create.RFTrace.write()`
        to write receiver function to a SAC file(in time domain)
    * :py:meth:`~pyglimer.rf.create.RFTrace.moveout()`
        to migrate the receiver function(s) to depth domain using one of the
        provided velocity-depth models (either '3D' for GyPsum or 'iasp91.dat'
        for iasp91). Piercing points will be appended to the object.
    * :py:meth:`~pyglimer.rf.create.RFTrace.ppoint()`
        to compute the piercing points in depth without migrating the receiver
        function using the provided velocity model.
    * :py:meth:`~pyglimer.rf.create.RFTrace.plot()`
        to plot the receiver function(s). The plot will be different depending
        on the type of receiver function: **1.** For *depth-migrated* RFs, the
        plot will be against depth. **2.** For an
        :py:class:`~pyglimer.rf.create.RFTrace` in time domain, the plot will be
        against time.
        
Methods for RFStream objects
############################

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.rf.create.RFStream.write()`
        to write receiver function(s) to SAC file(s) (in time domain). Creates
        one file per receiver function.
    * :py:meth:`~pyglimer.rf.create.RFStream.plot()`
        This plot will show the receiver functions depending on epicentral
        distance (i.e., a section)
    * :py:meth:`~pyglimer.rf.create.RFStream.plot_distribution()`
        Plot the azimuthal and ray-parameter distribution of all traces in the
        stream in a rose diagram.
    * :py:meth:`~pyglimer.rf.create.RFStream.station_stack()`
        to create a station specific stack of all receiver function in the
        object. For that to work, all RFs have to be from the same station.
    * :py:meth:`~pyglimer.rf.create.RFStream.dirty_ccp_stack()`
        Create a simple CCP Stack. For more on CCP stacking, see `the later part
        of this tutorial <./ccp>`_ **This method is experimental!**