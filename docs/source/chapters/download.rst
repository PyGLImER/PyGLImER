Waveform Download, preprocessing, and receiver function creation
----------------------------------------------------------------

Our first step will be to download the data from FDSN webservices for a certain time window.

The Request Class
+++++++++++++++++
The :class:`~pyglimer.waveform.request.Request` class handles all steps from download, over preprocessing,
to deconvolution (i.e. the creation of a time domain receiver function).
Since PyGLImER-v0.1.0 the database can be saved in two different formats - *MSEED* (raw waveforms), *SAC*
(receiver functions), and *XML* (station response and event database) **OR** in *hdf5* (hierachical data format).
Both options have advantages and downsides (see below).

MSEED/SAC Database
##################
If the user chooses this option, the :class:`~pyglimer.waveform.request.Request` object will create
a folder structure as defined by the user. Those contain raw (i.e. unprocessed, but downsampled)
waveform files in miniseed format, preprocessed (3 component in RTZ coordinate system, and filtered/discarded
by signal to noise ratio) waveforms in miniseed format together with info-files (shelve format),
and receiver functions in time domain and in .SAC format, respectively). Additionally,
a directory with station response files will be created.

.. note::

    Saving your database in this format might be the better option if you create a relatively
    small database. Then, computational times tend to be shorter.
    However, for large databases (e.g., world-wide) this will create millions of files and
    potentially overload your file system.

HDF5 Database
#############
Your second option is to save data in *hdf5* format. In this case, only two directories will be created;
one for the downsampled raw-data and one for the final time-domain receiver functions. The raw data
management is based on `PyASDF <https://seismicdata.github.io/pyasdf/>` (head there to
learn how to access waveforms, event catalogues, and station response files), whereas receiver
functions are saved in an hdf5 variant specific to PyGLImER (you can learn here, how to use it).

.. note::

    Use this format if you plan to create a large database as it will both save some disk space
    and, more importantly, create only two files per station.

.. note::
    Once a database is created,
    a new Request object will always update existing raw-data if the same
    rawdir is chosen (i.e. download new data, if any available). This is valid for both
    formats!

A Request object has four public methods:

.. hlist::
    :columns: 1

    * :func:`~pyglimer.waveform.request.Request.download_evtcat()`
    * :func:`~pyglimer.waveform.request.Request.download_waveforms()`
    * :func:`~pyglimer.waveform.request.Request.download_waveforms_small_db()`
    * :func:`~pyglimer.waveform.request.Request.preprocess()`

The functions are responsible for:

.. hlist::
    :columns: 1

    * Downloading the event catalogue - for which waveforms should be downloaded
    * (2+3) Downloading station information - such as response data - and raw waveform data
    * Downsampling the raw data, preprocessing the raw data and saving the filtered data in a different directory, and creating receiver functions.

However, all parameters are already set, when initialising the :class:`~pyglimer.waveform.request.Request` object.