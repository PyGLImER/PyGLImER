"""

A file for regularly used constants. Such as the Earth's radius.


"""
from obspy.geodetics import degrees2kilometers


# Earth's radius in km
R_EARTH = 6371.

DEG2KM = degrees2kilometers(1)
KM2DEG = 1.0/DEG2KM
maxz = 750  # maximum interpolation depth in km
maxzm = 200  # maximum depth for multiple interpolation in km
res = 1  # vertical resolution in km for interpolation and ccp bins
