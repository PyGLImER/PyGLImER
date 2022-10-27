The Request Class
+++++++++++++++++
The :py:class:`~pyglimer.waveform.request.Request` class handles all steps from download, over preprocessing,
to deconvolution (i.e. the creation of a time domain receiver function).
Since PyGLImER-v0.1.0 the database can be saved in two different formats - *MSEED* (raw waveforms), *SAC*
(receiver functions), and *XML* (station response and event database) **OR** in *hdf5* (hierachical data format).
Both options have advantages and downsides (see below).

MSEED/SAC Database
##################
If the user chooses this option, the :py:class:`~pyglimer.waveform.request.Request` object will create
a folder structure as defined by the user. Those contain raw (i.e. unprocessed, but downsampled)
waveform files in miniseed format, preprocessed (3 component in RTZ coordinate system, and filtered/discarded
by signal to noise ratio) waveforms in miniseed format together with info-files (shelve format),
and receiver functions in time domain and in *.SAC* format, respectively). Additionally,
a directory with station response files will be created.

.. note::

    Saving your database in this format might be the better option if you create a relatively
    small database. Then, computational times tend to be shorter.
    However, for large databases (e.g., world-wide) this will create millions of files and
    potentially overload your file system.

HDF5 Database
#############
Your second option is to save data in *hdf5* format. In this case, only two directories will be created;
one for the downsampled raw-data and another one for the final time-domain receiver functions. The raw data
and receiver functions are saved in an hdf5 variant specific to PyGLImER (in the following, we will learn how to use it).

.. note::

    Use this format if you plan to create a large database as it will both save some disk space
    and, more importantly, create only two files per station.

.. note::
    Once a database is created,
    a new Request object will always update existing raw-data if the same
    rawdir is chosen (i.e. download new data, if any available). This is valid for both
    formats!

Methods of the Request class
############################

A Request object has four public methods:

.. hlist::
    :columns: 1

    * :py:meth:`~pyglimer.waveform.request.Request.download_eventcat()`
    * :py:meth:`~pyglimer.waveform.request.Request.download_waveforms()`
    * :py:meth:`~pyglimer.waveform.request.Request.download_waveforms_small_db()`
    * :py:meth:`~pyglimer.waveform.request.Request.preprocess()`

The functions are responsible for:

.. hlist::
    :columns: 1

    * Downloading the event catalogue - for which waveforms should be downloaded
    * (2+3) Downloading station information - such as response data - and raw waveform data
    * Downsampling the raw data, preprocessing the raw data and saving the filtered data in a different directory, and creating receiver functions.

However, all parameters are already set, when initialising the :py:class:`~pyglimer.waveform.request.Request` object.

.. note::

    As you surely have noticed, there are two functions to download data. As the name suggests
    :py:meth:`~pyglimer.waveform.request.Request.download_waveforms_small_db()`, is the tool
    of choice if you wish to download smaller databases, consisting of data from only few stations
    or networks. This function does also have the advantage that you can
    **download data from defined lists of networks and stations** instead of having to rely only on wildcards.
    For smaller databases :py:meth:`~pyglimer.waveform.request.Request.download_waveforms_small_db()` will
    download data up to twice faster than :py:meth:`~pyglimer.waveform.request.Request.download_waveforms()`.
    The latter will be the better choice if you create databases on continental or even global scales. It
    utilises the `Obspy Mass Downloader <https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html>`_.

Setting the parameters for your request
#######################################

The parameters for preprocessing and download are set when initialising the
:py:class:`~pyglimer.waveform.request.Request` object. Probably the most convenient way to define them
is to create a *yaml* file with the parameters. An example comes with this repository in `params.yaml`:


.. code-block:: yaml
    :linenos:

  # This file is used to define the parameters used for PyGLImER
  # ### Project wide parameters ###
  # lowest level project directory
  proj_dir : 'database'
  # raw waveforms
  raw_subdir: 'waveforms/raw'
  # preprocessed subdir, only in use if fileformat = 'mseed'
  prepro_subdir: 'waveforms/preprocessed'
  # receiver function subdir
  rf_subdir: 'waveforms/RF'
  # statxml subdir
  statloc_subdir: 'stations'
  # subdir for event catalogues
  evt_subdir: 'event_catalogs'
  # directory for logging information
  log_subdir : 'log'
  # levels:
  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
  loglvl: 'WARNING'
  # format, either mseed or hdf5
  format: 'hdf5'

  # The teleseismic phase to use (P or S or also more exotic ones like SKS, PKP, ScS)
  phase: 'S'

  ### Request parameters
  ## First, everything concerning the download
  # waveform client, list of strings
  # use None if you want to download from all available FDSN servers
  waveform_client: ['IRIS']
  # Use an already downloaded event catalog
  # If so insert path+filename here.
  evtcat: None
  # earliest event
  starttime: '2009-06-1 00:00:00.0'
  # latest event
  endtime: '2011-12-31 00:00:00.0'
  # Minumum Magnitude
  minmag: 5.5
  # Network and station to use, unix-style wildcards are allowed
  # if you use the Request.download_waveforms_small_db method,
  # you can also provide a list of networks and/or a list of stations
  network: 'YP'
  station: '*'

  ## concerning preprocessing
  # Coordinate system to rotate the seismogram to before deconvolution
  # RTZ, LQT, or PSS
  rot: 'PSS'
  # Polarisation, use v for v/q receiver functions
  # and h for transverse (SH)
  pol: 'v'
  # Deconvolution method to use
  # Iterative time domain: 'it'
  # Waterlevel Spectral Division: 'waterlevel'
  deconmeth: 'it'
  # Remove the station response. Set to False if you don't have access to the response
  remove_response: False

You can then read the yaml file using *pyyaml* like so:

.. code-block:: python

    import yaml

    from pyglimer.waveform.request import Request

    with open('/path/to/my/params.yaml') as pfile:
        kwargs = yaml.load(pfile, Loader=yaml.FullLoader)
    
    r = Request(**kwargs)

Alternatively, you could of course just set the parameters while initialising the
object.