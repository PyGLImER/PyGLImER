RF Handling
+++++++++++

Supposing that you have created receiver functions and have them available in
the directory ``output/RF``, we can now start imaging!

Handling of receiver functions is generally done by using the
:py:class:`~pyglimer.rf.create.RFTrace` and
:py:class:`~pyglimer.rf.create.RFStream` classes. These classes are built upon a
modified version of the `rf <https://rf.readthedocs.io/en/latest/index.html>`_
project by Tom Richter.

.. note::
    
    Just as in the obspy base classes, an
    :py:class:`~pyglimer.rf.create.RFStream` can hold several receiver
    functions, while an :py:class:`~pyglimer.rf.create.RFTrace` object can only
    hold one receiver function. In the *.sac* format, saving
    :py:class:`~pyglimer.rf.create.RFStream` will lead to the creation of
    several files.

.. toctree::
   :maxdepth: 4

   sac
   hdf5
   methods