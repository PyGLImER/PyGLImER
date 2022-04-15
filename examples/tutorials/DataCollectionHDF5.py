#!/bin/env python
"""
HDF5 Database
=============

This example is nearly identical to the MSEED tutorial except the fact that we
are specifying a different data format in the request dictionary.

Downloading Event & Station Metadata
------------------------------------

In this section, the only difference is the ``format`` ``key`` in the
``request_dict`` that is set to ``hdf5``.

"""
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_dummy_images = 1

# %%
# First let's get a path where to create the data.


# Some needed Imports
import os
from typing import NewType
from obspy import UTCDateTime
from pyglimer.waveform.request import Request

# Get notebook path for future reference of the database:
try: db_base_path = ipynb_path
except NameError: db_base_path = os.getcwd()

# Define file locations
proj_dir = os.path.join(db_base_path, 'database_hdf5')

# Define network and station to download RFs for
network = 'IU'
station = 'HRV'

request_dict = {
    # Necessary arguments
    'proj_dir': proj_dir,
    'raw_subdir': 'waveforms/raw', # Directory of the waveforms
    'prepro_subdir': 'waveforms/preprocessed',  # Directory of the preprocessed waveforms
    'rf_subdir': 'waveforms/RF',  # Directory of the receiver functions
    'statloc_subdir': 'stations', # Directory stations
    'evt_subdir': 'events',       # Directory of the events
    'log_subdir': 'log',          # Directory for the logs
    'loglvl': 'WARNING',          # logging level
    'format': 'hdf5',              # Format to save database in
    "phase": "P",                 # 'P' or 'S' receiver functions
    "rot": "RTZ",                 # Coordinate system to rotate to
    "deconmeth": "waterlevel",    # Deconvolution method
    "starttime": UTCDateTime(2021, 1, 1, 0, 0, 0), # Starttime of database.
                                                # Here, starttime of HRV
    "endtime": UTCDateTime(2021, 7, 1, 0, 0, 0), # Endtimetime of database
    # kwargs below
    "pol": 'v',                   # Source wavelet polaristion. Def. "v" --> SV
    "minmag": 5.5,                # Earthquake minimum magnitude. Def. 5.5
    "event_coords": None,         # Specific event?. Def. None
    "network": network,              # Restricts networks. Def. None
    "station": station,             # Restricts stations. Def. None
    "waveform_client": ["IRIS"],  # FDSN server client (s. obspy). Def. None
    "evtcat": None,               # If you have already downloaded a set of
                                # events previously, you can use them here
    "loglvl": 'DEBUG'
}

# %%
# Now that all parameters are in place, let's initialize the 
# :class:`pyglimer.waveform.request.Request`

# Initializing the Request class and downloading the data
R = Request(**request_dict)

# %%
# The initialization will look for all events for which data is available. To see 
# whether the events make sense we plot a map of the events:

import matplotlib.pyplot as plt
from pyglimer.plot.plot_utils import plot_catalog
from pyglimer.plot.plot_utils import set_mpl_params

# Setting plotting parameters
set_mpl_params()

# Plotting the catalog
plot_catalog(R.evtcat)

# %%
# We can also quickly check how many events we gathered.

print(f"There are {len(R.evtcat)} available events")

# %% Downloading waveform data and station information
# -------------------------------------------------
#
# Again, this section does not really change, because the ``request_dict``
# parsed all needed information to :class:`pyglimer.waveform.request.Request`.

R.download_waveforms_small_db(channel='BH?')

# %% Let's have a quick look at how many traces we actually downloaded. So, now,
# there is indeed a change in the code because we saved the raw data in form of
# ``ASDFDataset``s and we need to actually access the ``hdf5`` file that we
# to count how many traces it contains.

# Import ASDFDataset
from pyasdf import ASDFDataSet

# Path to the where the miniseeds are stored
data_storage = os.path.join(
    proj_dir, 'waveforms', 'raw', 'P', f'{network}.{station}.h5')

# Read the data from the station ``h5`` file
with ASDFDataSet(data_storage, mode='r', mpi=False) as ds:

    # Perform waveform query on ASDFDataSet
    stream = ds.get_waveforms(
        network,
        station,
        '*', # Location
        '*', # Channel
        request_dict['starttime']-1000,  # Starttime 
        request_dict['endtime']+1000,    # Endtime
        'raw_recording')

# Print output
print(f"Number of downloaded waveforms: {len(stream)}")


# %%
# The final step to get you receiver function data is the preprocessing. 
# Although it is hidden in a single function, which
# is :func:`pyglimer.waveform.request.Request.preprocess`
# A lot of decisions are being made:
#
# Processing steps:
# 1. Clips waveform to the right length (tz before and ta after theorethical 
# arrival.)
# 2. Demean & Detrend
# 3. Tapering
# 4. Remove Instrument response, convert to velocity &
# simulate havard station
# 5. Rotation to NEZ and, subsequently, to RTZ.
# 6. Compute SNR for highpass filtered waveforms (highpass f defined in 
# qc.lowco) If SNR lower than in qc.SNR_criteria for all filters, rejects 
# waveform.
# 7. Write finished and filtered waveforms to folder
# specified in qc.outputloc.
# 8. Write info file with shelf containing station,
# event and waveform information.
# 9. (Optional) If we had chosen a different coordinate system in ``rot``
# than RTZ, it would now cast the preprocessed waveforms information
# that very coordinate system.
# 10. Deconvolution with method ``deconmeth`` from our dict is perfomed.
#

R.preprocess(hc_filt=1.5, client='single')

# %% Read the IU-HRV receiver functions as a Receiver function stream
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 
# Here, again we have a change between the ``SAC`` workflow and the ``HDF5``
# workflow because the RFs are saved in form of
# :class:`pyglimer.database.rfh5.RFDataBase`s. It is important to note that the
# ``RFDataBase`` is an abstraction of an ``HDF5`` file that contains RF specific 
# Header info that cannot be saved in ASDF. So, we import the database class
# and query for the P receiver functions we computed. and output a stream.

from pyglimer.database.rfh5 import RFDataBase

network = request_dict['network']
station = request_dict['station']

path_to_rf = os.path.join(proj_dir, 'waveforms','RF','P', f'{network}.{station}.h5')

# Open RF Database
with RFDataBase(path_to_rf, 'r') as rfdb:

    # Query database for specific RFs
    rfstream = rfdb.get_data('IU', 'HRV', 'P', '*', 'rf')

# Print number of RFs queried
print(f"Number of RFs: {len(rfstream)}")

# %% 
# PyGLImER is based on Obspy and hence, similar to ``obspy``, we can access
# the ``Stats`` of a ``RFTrace`` in a ``RFStream``. The abstraction from obspy
# follows a need for more (SAC) header information to handle RFs. Let's take
# a look at the attributes.

from pprint import pprint

rftrace = rfstream[0]
pprint(rftrace.stats)


# %%
# First Receiver function plots
# -----------------------------
# 
# If the Receiver functions haven't been further processed,
# they are plotted as a function of time. A single receiver
# function in the stream will be plotted as function of time
# only. A full stream can make use of the distance measure saved
# in the sac-header and plot an entire section as a function of
# time and epicentral distance.
#
# Plot single RF
# ++++++++++++++
#
# Below we show how to plot the receiver function
# as a function of time, and the clean option, which plots
# the receiver function without any axes or text.

from pyglimer.plot.plot_utils import set_mpl_params

# Plot RF
rftrace.plot()

# %% 
# Let's zoom into the first 20 seconds (~200km)

rftrace.plot(lim=[0,20])

# %% 
# Plot RF section
# +++++++++++++++
#
# Since we have an entire stream of receiver functions at hand,
# we can plot a section

rfstream.plot(scalingfactor=1)

# %% 
# Similar to the single RF plot we can provide time and 
# epicentral distance limits:

timelimits = (0, 20)  # seconds  
epilimits = (32, 36)  # epicentral distance
rfstream.plot(
    scalingfactor=0.25, lim=timelimits, epilimits=epilimits,
    linewidth=0.75)

# %%
# By increasing the scaling factor and removing the plotted lines, we can
# already see trends:

rfstream.plot(
    scalingfactor=0.5, lim=timelimits, epilimits=epilimits, 
    line=False)


# %% 
# As simple as that you can create your own receiver functions with 
# just a single smalle script.