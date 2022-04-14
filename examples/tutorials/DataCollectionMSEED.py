"""
Collection Data in the MSEED 
============================

In this Tutorial we are going to get all good receiver functions for the year
2018 for station IU-HRV ([Adam Dziewonski
Observatory](http://www.seismology.harvard.edu/hrv.html)).

To Compute the receiver function we need to download and organize the observed
data. PyGLImER will automatically take car of these things for you in the
background, but do check out the 'Database Docs' if you aren't familiar.

Downloading Event & Station Metadata
------------------------------------

To download the data we use the :class:`pyglimer.waveform.request.Request`
class. The first method from this class that we are going to use is the download
event catalog public method
:func:`pyglimer.waveform.request.Request.download_evtcat`, to get a set
of events that contains all wanted earthquakes. This method is launched
automatically upon initialization.

To initialize said `class` we setup a parameter dictionary, with all the needed
information. Let's look at the expected information:

"""
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_dummy_images = 1

# %%
# First let's get a path where to create the data.

# Some needed Imports
import os
from obspy import UTCDateTime
from pyglimer.waveform.request import Request

# Get notebook path for future reference of the database:
try: db_base_path = ipynb_path
except NameError: db_base_path = os.getcwd()

# Define file locations
proj_dir = os.path.join(db_base_path, 'database')



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
    'format': 'sac',              # Format to save database in
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
    "network": "IU",              # Restricts networks. Def. None
    "station": "HRV",             # Restricts stations. Def. None
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

# %%
# Downloading waveform data and station information
# -------------------------------------------------
#
# The next step on our journey to receiver functions is retrieving
# the corresponding waveform data.
# To retrieve the waveform data, we can two different public methods
# of `Request`: `download_waveforms()` or, as in this case `download_waveforms_small_db()`.
# 
# Both methods use the station and event locations to get viable
# records of receiver function depending on epicentral distance and
# traveltimes.
# 
# `download_waveforms()` relies on obspy's massdownloader and is best suited for
# extremely large databases (i.e., from many different networks and stations).
# `download_waveforms_small_db()` is faster for downloads from few stations and, additionally,
# has the advantage that the desired networks and stations can be defined as lists and not only
# as strings. Note that this method requires you to define the channels, you'd like to download
# (as a string, wildcards allowed).
# 
# ***NOTE:*** This might take a while.
# 


print('downloading')
R.download_waveforms_small_db(channel='BH?')

# %%
# Let's have a quick look at how many miniseeds we actually downloaded.

from glob import glob

# Path to the where the miniseeds are stored
data_storage = os.path.join(
    proj_dir, 'waveforms','raw','P','**', '*.mseed')

# Print output
print(f"Number of downloaded waveforms: {len(glob(data_storage))}")

# %%
# The final step to get you receiver function data is the preprocessing. Although it is hidden in a single function, which
# is :function:`:func:`pyglimer.waveform.request.Request.preprocess`
# A lot of decisions are being made:
#
# Processing steps:
# 1. Clips waveform to the right length
#    (tz before and ta after theorethical arrival.)
# 2. Demean & Detrend
# 3. Tapering
# 4. Remove Instrument response, convert to velocity &
#    simulate havard station
# 5. Rotation to NEZ and, subsequently, to RTZ.
# 6. Compute SNR for highpass filtered waveforms
#    (highpass f defined in qc.lowco)
#    If SNR lower than in qc.SNR_criteria for all filters,
#    rejects waveform.
# 7. Write finished and filtered waveforms to folder
#    specified in qc.outputloc.
# 8. Write info file with shelf containing station,
#    event and waveform information.
# 9. (Optional) If we had chosen a different coordinate system in ``rot``
#    than RTZ, it would now cast the preprocessed waveforms information
#    that very coordinate system.
# 10. Deconvolution with method ``deconmeth`` from our dict is perfomed.
#    
# It again uses the request class to perform this.

R.preprocess(hc_filt=1.5)

# %%
# First Receiver functions
# ------------------------
# 
# The following few section show how to plot 
# 
# 1. Single raw RFs
# 2. A set of raw RFs
# 3. A move-out corrected RF
# 4. A set of move-out corrected RFs
#
# 
# Read the IU-HRV receiver functions as a Receiver function stream
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 
# Let's read a receiver function set and see what it's all about! 
# (i.e. let's look at what data a Receiver function trace contains
# and how we can use it!)

from pyglimer.rf.create import read_rf

path_to_rf = os.path.join(proj_dir, 'waveforms','RF','P','IU','HRV','*.sac')
rfstream = read_rf(path_to_rf)

print(f"Number of RFs: {len(rfstream)}")

# %% 
# PyGLImER is based on Obspy, but to handle RFs we need some more attributes: 
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