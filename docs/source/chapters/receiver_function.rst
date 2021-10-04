Handling Receiver Functions
---------------------------

Supposing that you have created receiver functions and have them available in the directory ``output/RF``,
we can now start imaging!

Handling of receiver functions is generally done by using the :py:class:`~pyglimer.rf.create.RFTrace` and :py:class:`~pyglimer.rf.create.RFStream` classes. These
classes are built upon a modified version of the `rf <https://rf.readthedocs.io/en/latest/index.html>`_ project
by Tom Eulenfeld.

.. note::
    
    Just as in the obspy base classes, an :py:class:`~pyglimer.rf.create.RFStream` can hold several receiver functions,
    while an :py:class:`~pyglimer.rf.create.RFTrace` object can only hold one receiver function. As all rfs are saved in .sac format,
    saving :py:class:`~pyglimer.rf.create.RFStream` will lead to the creation of several files.

Reading receiver functions that were saved in sac format
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Receiver functions in .sac format can be read using the :py:func:`~pyglimer.rf.create.read_rf` function. By doing so, we
obtain an :py:class:`~pyglimer.rf.create.RFStream` object, on which we can apply a number of functions.

Receiver functions stored in hdf5 format
++++++++++++++++++++++++++++++++++++++++

If you saved your receiver functions in *hdf5* format, you can use the :py:class:`~pyglimer.database.rfh5.RFDataBase` class
to access and manipulate your database.

As a user, you will only ever be calling the :class:`~pyglimer.database.rfh5.RFDataBase` class.
The only function of this class is to return a :class:`~pyglimer.database.rfh5.DBHandler`, which hold all the
"useful" functions. To call :class:`~pyglimer.database.rfh5.RFDataBase`, use a context manager like so:

>>> from pyglimer.database.rfh5 import RFDataBase
>>> with RFDataBase('/path/to/myfile.h5') as rfdb:
>>>     type(rfdb)  # This is a DBHandler
<class 'pyglimer.database.rfh5.DBHandler'>

.. warning::

    Do not call :class:`~pyglimer.database.rfh5.DBHandler` directly! This might lead to unexpected behaviour or
    even dataloss due to corrupted hdf5 files.

.. warning::

    If you should for some reason decide to not use the context manager, you will have to close the hdf5 file
    with :meth:`pyglimer.database.rfh5.DBHandler._close` to avoid corrupting your files!

:py:class:`~pyglimer.database.rfh5.DBHandler` has the following public methods:

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.database.rfh5.RFDataBase.add_rf` to add receiver functions to the database
    * :py:meth:`~pyglimer.database.rfh5.RFDataBase.get_data` to read data from this file
    * :py:meth:`~pyglimer.database.rfh5.RFDataBase.get_coords` to get the coordinates of the associated station
    * :py:meth:`~pyglimer.database.rfh5.RFDataBase.walk` to iterate over all receiver functions in a subset defined by the provided arguments

Reading data
############

The most common usecase is probably that you will want to access receciver functions that **PyGLImER** computed
for you (as shown earlier). To do so, you can use the :py:meth:`~pyglimer.database.rfh5.RFDataBase.get_data`
method:

>>> from pyglimer.database.rfh5 import RFDataBase
>>> with RFDataBase('/path/to/myfile.h5') as rfdb:
>>>     rfst = rfdb.get_data(
>>>         tag='rf', network='IU', station='*', phase='P', evt_time='*', pol='v')
>>> # rfst is a RFStream object on that we can use methods (more later)
>>> print(type(cst))
<class 'pyglimer.rf.create.CorrStream'>
>>> #rfst.count()
289

As you can see, we use seed station codes to identify data. All arguments accept wildcards.
The data we are loading are receiver functions from waveforms recorded at every station of the *IU* network
caused by events with any origin time (``evt_time='*'``).

.. seealso::
    
    If you want to create your own function to :py:meth:`~pyglimer.database.rfh5.RFDataBase.walk`
    might come in handy.

Tags
####

**PyGLImER** uses tags to identify your data. You could for example use different tags for differently processed data. ``rf`` is the standard
tag for receiver function data.


Getting an overview over available data
#######################################

You can **Access the DBHandler like a dictionary**: Just like in h5py, it is possible to access the :class:`~pyglimer.database.rfh5.DBHandler` like a dictionary. The logic works as follows:
    dbh[tag][netcomb][statcomb][chacomb][corr_start][corr_end]

Following the logic of the structure above, we can get a list of all available tags as follows:

>>> print(list(dbh.keys()))
['rf', 'rf_with_my_funny_processing_idea']

Writing data to hdf5
++++++++++++++++++++

If you postprocess your receiver functions (e.g., stacking), you might want to save the data afterwards.
You can do that like below:

.. code-block:: python
    :linenos:

    from pyglimer.database.rfh5 import RFDataBase

    # Suppose you have a RFStream or RFTrace object rf
    # that has a header with all the station information

    with RFDataBase('/path/to/myfile.h5') as rfdb:
        rfst = rfdb.get_data(
            rf, tag='rf_with_my_funny_processing_idea')

We can retrieve the :class:`~pyglimer.rf.create.RFStream` as shown above.
Network, station, and channel information are determined automatically from the header and used to identify and locate the data.

Methods available for both RFStream and RFTrace objects
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

No matter whether your data was stored as *mseed* or *hdf5*, after reading it, you will receive an :class:`~pyglimer.rf.create.RFStream`
object. Below you can find a list of the available public methods. For examples on how to use them, please
consult the example Jupyter notebooks.

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.rf.create.RFTrace.write()`
        to write receiver function to a SAC file(in time domain)
    * :py:meth:`~pyglimer.rf.create.RFTrace.moveout()`
        to migrate the receiver function(s) to depth domain using one of the provided velocity-depth models
        (either '3D' for GyPsum or 'iasp91.dat' for iasp91). Piercing points will be appended to the object.
    * :py:meth:`~pyglimer.rf.create.RFTrace.ppoint()`
        to compute the piercing points in depth without migrating the receiver function using the provided velocity
        model.
    * :py:meth:`~pyglimer.rf.create.RFTrace.plot()`
        to plot the receiver function(s). The plot will be different
        depending on the type of receiver function: **1.** For *depth-migrated* RFs, the plot will be against
        depth. **2.** For an :py:class:`~pyglimer.rf.create.RFTrace` in time domain, the plot will be against time.
        
    * :py:meth:`~pyglimer.rf.create.RFTrace.

Methods for RFStream objects
############################

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.rf.create.RFStream.write()`
        to write receiver function(s) to SAC file(s) (in time domain). Creates one file per receiver function.
    * :py:meth:`~pyglimer.rf.create.RFStream.plot()`
        This plot will show the receiver functions depending on epicentral distance (i.e., a section)
    * :py:meth:`~pyglimer.rf.create.RFStream.plot_distribution()`
        Plot the azimuthal and ray-parameter distribution of all traces in the stream in a rose diagram.
    * :py:meth:`~pyglimer.rf.create.RFStream.station_stack()`
        to create a station specific stack of all receiver function in the object.
        For that to work, all RFs have to be from the same station.
    * :py:meth:`~pyglimer.rf.create.RFStream.dirty_ccp_stack()`
        Create a simple CCP Stack. For more on CCP stacking, see
        `the later part of this tutorial <../ccp>`_ :warning: ***This method is experimental!***