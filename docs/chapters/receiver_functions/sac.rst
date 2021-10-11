Reading receiver functions that were saved in sac format
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Receiver functions in .sac format can be read using the
:py:func:`~pyglimer.rf.create.read_rf` function. By doing so, we obtain an
:py:class:`~pyglimer.rf.create.RFStream` object, on which we can apply a number
of functions.