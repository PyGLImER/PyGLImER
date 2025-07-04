Raw waveforms stored in hdf5 format
+++++++++++++++++++++++++++++++++++

If you saved your raw wavforms *hdf5* format, you can use the
:py:class:`~pyglimer.database.raw.RawDataBase` class to access and manipulate
your database.

As a user, you will only ever be calling the
:py:class:`~pyglimer.database.raw.RawataBase` class. The only function of this
class is to return a :py:class:`~pyglimer.database.raw.DBHandler`, which hold
all the "useful" functions. To call
:py:class:`~pyglimer.database.raw.RawDataBase`, use a context manager like so:

>>> from pyglimer.database.raw import RawDataBase
>>> with RawDataBase('/path/to/myfile.h5') as rdb:
>>>     type(rdb)  # This is a DBHandler
<class 'pyglimer.database.raw.DBHandler'>

.. warning::

    Do not call :py:class:`~pyglimer.database.raw.DBHandler` directly! This
    might lead to unexpected behaviour or even dataloss due to corrupted hdf5
    files.

.. warning::

    If you should for some reason decide to not use the context manager, you
    will have to close the hdf5 file with
    :py:meth:`~pyglimer.database.raw.DBHandler._close()` to avoid corrupting
    your files!

:py:class:`~pyglimer.database.raw.DBHandler` has the following public methods:

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.database.raw.DBHandler.add_waveform()` to add waveforms to the database 
    * :py:meth:`~pyglimer.database.raw.DBHandler.add_response()` to add stations
      response data to the database 
    * :py:meth:`~pyglimer.database.raw.DBHandler.get_data()` to read data from
      this file 
    * :py:meth:`~pyglimer.database.raw.DBHandler.get_response()` to get
      the station inventory of the associated station 
    * :py:meth:`~pyglimer.database.raw.DBHandler.walk()` to iterate over all
      waveforms in a subset defined by the provided arguments

Reading data
############

You can access downloaded waveform data using the
:py:meth:`~pyglimer.database.raw.DBHandler.get_data()` method:

>>> from pyglimer.database.raw import RawDataBase
>>> with RawDataBase('/path/to/myfile.h5') as rdb:
>>>     st = rdb.get_data(
>>>         network='IU', station='*', evt_id='*', tag='raw')
>>>     # st is an ObsPy Stream object
>>>     inv = rdb.get_response(network='IU', station='HRV')
>>>     # inv is an ObsPy inventory object.

As you can see, we use seed station codes to identify data. All arguments accept
wildcards. Each waveform is associated to an event identified by its origin time
(``evt_id='*'``, this can be a ``UTCDateTime``).

.. seealso::
    
    To iterate over raw waveforms use
    :py:meth:`~pyglimer.database.raw.DBHandler.walk()`.

Tags
####

**PyGLImER** uses tags to identify your data. You could for example use
different tags for differently processed data. ``raw`` is the standard tag for
raw waveform data.


Writing data to hdf5
++++++++++++++++++++

If you postprocess your receiver functions (e.g., stacking), you might want to
save the data afterwards. You can do that like below:

.. code-block:: python
    :linenos:

    from pyglimer.database.raw import RawDataBase

    # Suppose you have a obspy Stream or Trace object st
    # event is an obspy Event that st is associated to

    with RawDataBase('/path/to/myfile.h5') as rdb:
        rfst = rdb.add_data(
            st, event.preferred_origin().time)


.. seealso::

    :py:class:`~pyglimer.database.raw.DBHandler` (raw data specific) handles very similarly to
    :py:class:`~pyglimer.database.rfdb.DBHandler` (receiver function specific).