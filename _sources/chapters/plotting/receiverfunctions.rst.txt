Plotting Plain Receiver Functions
+++++++++++++++++++++++++++++++++



Single Receiver Function
------------------------

Given the data downloaded using the example notebook for data collection, we
can simply read and plot them using

.. code-block:: python
    :linenos:

    from pyglimer.rf.create import read_rf
    set_mpl_params()

    # Read all RFs from Station IU/HRV
    rfst = read_rf("../database/waveforms/RF/P/IU/HRV/*.sac")

    # Some random RF from the 800 avalable one at IU.HRV
    N = 753

    # Plot RF and save its output.
    rfst[N].plot()

which results in the following image:

.. image:: figures/IU.HRV.00.PRF_raw.svg

We can time limit the figure as well

.. code:: python

    rfst[N].plot(tlim=[0, 20])

which cuts out the RF between 0 and 20 seconds

.. image:: figures/IU.HRV.00.PRF_timelimit.svg

If you feel artsy and only want the trace

.. code:: python

    rfst[N].plot(tlim=[0, 20], clean=True)

which removes all labels, axes etc.

.. image:: figures/IU.HRV.00.PRF_timelimit.svg

Refer to :py:meth:`~pyglimer.rf.create.RFStream.plot()` and :py:meth:`~pyglimer.rf.create.RFTrace.plot()`
for all possible plotting arguments.


Receiver Function Section
-------------------------

We can plot all receiver functions in an ``RFStream`` into a section depending 
on epicentral distance.

.. code-block:: python

    # Plot section
    rfst.plot(scalingfactor=1)

This plots all available RFs in the Stream into a section

.. image:: figures/section_raw.png

Also this plot can be limited using the right arguments

.. code-block:: python
    :linenos:

    # Plot section with limits
    timelimits = (0, 20)  # seconds
    epilimits = (32, 36)  # epicentral distance
    rfst.plot(
        scalingfactor=0.25, linewidth=0.75,
        timelimits=timelimits, epilimits=epilimits
        )

which provides a more detailed view of the receiver functions

.. image:: figures/section_limits.png

