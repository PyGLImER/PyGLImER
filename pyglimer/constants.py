"""

A file for regularly used constants. Such as the Earth's radius.


"""
from obspy.geodetics import degrees2kilometers


# Earth's radius in km
R_EARTH = 6371.

DEG2KM = degrees2kilometers(1)
maxz = 750  # maximum interpolation depth in km
res = 1  # vertical resolution in km for interpolation and ccp bins
