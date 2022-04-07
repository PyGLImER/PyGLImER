Receiver functions stored in hdf5 format
++++++++++++++++++++++++++++++++++++++++

If you saved your receiver functions in *hdf5* format, you can use the
:py:class:`~pyglimer.database.rfh5.RFDataBase` class to access and manipulate
your database.

As a user, you will only ever be calling the
:py:class:`~pyglimer.database.rfh5.RFDataBase` class. The only function of this
class is to return a :py:class:`~pyglimer.database.rfh5.DBHandler`, which hold
all the "useful" functions. To call
:py:class:`~pyglimer.database.rfh5.RFDataBase`, use a context manager like so:

>>> from pyglimer.database.rfh5 import RFDataBase
>>> with RFDataBase('/path/to/myfile.h5') as rfdb:
>>>     type(rfdb)  # This is a DBHandler
<class 'pyglimer.database.rfh5.DBHandler'>

.. warning::

    Do not call :py:class:`~pyglimer.database.rfh5.DBHandler` directly! This
    might lead to unexpected behaviour or even dataloss due to corrupted hdf5
    files.

.. warning::

    If you should for some reason decide to not use the context manager, you
    will have to close the hdf5 file with
    :py:meth:`~pyglimer.database.rfh5.DBHandler._close()` to avoid corrupting
    your files!

:py:class:`~pyglimer.database.rfh5.DBHandler` has the following public methods:

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.database.rfh5.DBHandler.add_rf()` to add receiver
      functions to the database 
    * :py:meth:`~pyglimer.database.rfh5.DBHandler.get_data()` to read data from
      this file 
    * :py:meth:`~pyglimer.database.rfh5.DBHandler.get_coords()` to get
      the coordinates of the associated station 
    * :py:meth:`~pyglimer.database.rfh5.DBHandler.walk()` to iterate over all
      receiver functions in a subset defined by the provided arguments

Reading data
############

The most common usecase is probably that you will want to access receciver
functions that **PyGLImER** computed for you (as shown earlier). To do so, you
can use the :py:meth:`~pyglimer.database.rfh5.DBHandler.get_data()` method:

>>> from pyglimer.database.rfh5 import RFDataBase
>>> with RFDataBase('/path/to/myfile.h5') as rfdb:
>>>     rfst = rfdb.get_data(
>>>         tag='rf', network='IU', station='*', phase='P', evt_time='*', pol='v')
>>> # rfst is a RFStream object on that we can use methods (more later)
>>> print(type(cst))
<class 'pyglimer.rf.create.CorrStream'>
>>> #rfst.count()
289

As you can see, we use seed station codes to identify data. All arguments accept
wildcards. The data we are loading are receiver functions from waveforms
recorded at every station of the *IU* network caused by events with any origin
time (``evt_time='*'``).

.. seealso::
    
    If you want to create your own function to
    :py:meth:`~pyglimer.database.rfh5.DBHandler.walk()` might come in handy.

Tags
####

**PyGLImER** uses tags to identify your data. You could for example use
different tags for differently processed data. ``rf`` is the standard tag for
receiver function data.


Getting an overview over available data
#######################################

You can **Access the DBHandler like a dictionary**: Just like in h5py, it is
possible to access the :class:`~pyglimer.database.rfh5.DBHandler` like a
dictionary. The logic works as follows:

    dbh[tag][network][station][phase][pol][evt_time]

Following the logic of the structure above, we can get a list of all available
tags as follows:

>>> print(list(dbh.keys()))
['rf', 'rf_with_my_funny_processing_idea']

Writing data to hdf5
++++++++++++++++++++

If you postprocess your receiver functions (e.g., stacking), you might want to
save the data afterwards. You can do that like below:

.. code-block:: python
    :linenos:

    from pyglimer.database.rfh5 import RFDataBase

    # Suppose you have a RFStream or RFTrace object rf
    # that has a header with all the station information

    with RFDataBase('/path/to/myfile.h5') as rfdb:
        rfst = rfdb.add_rf(
            rf, tag='rf_with_my_funny_processing_idea')

We can retrieve the :class:`~pyglimer.rf.create.RFStream` as shown above.
Network, station, and channel information are determined automatically from the
header and used to identify and locate the data.