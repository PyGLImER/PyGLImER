Waveform Download, preprocessing, and receiver function creation
----------------------------------------------------------------

Our first step will be to download the data from FDSN webservices for a certain time window.

The Request Class
+++++++++++++++++
The :class:`~pyglimer.waveform.request.Request` class handles all steps from download, over preprocessing,
to deconvolution (i.e. the creation of a receiver function in time domain).
It will first create a folder structure as defined by the user. Those contain raw (i.e. unprocessed, but downsampled)
waveform files in miniseed format, preprocessed (3 component in RTZ coordinate system, and filtered/discarded
by signal to noise ratio) waveforms in miniseed format together with info-files (shelve format),
and receiver functions in time domain and in .SAC format, respectively).

.. note::  Once a database is created,
            a new Request object will always update existing raw-data if the same
            rawdir is chosen (i.e. download new data, if any available).

A Request object has three public methods:

.. hlist::
    :columns: 1

    * :func:`~pyglimer.waveform.request.Request.download_evtcat()`
    * :func:`~pyglimer.waveform.request.Request.download_waveforms()`
    * :func:`~pyglimer.waveform.request.Request.preprocess()`

The functions are responsible for:

.. hlist::
    :columns: 1

    * Downloading the event catalogue - for which waveforms should be downloaded
    * Downloading station information - such as response data - and raw waveform data
    * Downsampling the raw data, preprocessing the raw data and saving the filtered data in a different directory, and creating receiver functions.

However, all parameters are already set, when initialising the :class:`~pyglimer.waveform.request.Request` object.