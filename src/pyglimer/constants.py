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
# Time Window in seconds that should be downloaded before theoretical arrival
onset = {'P': 30, 'S': 120}
onsetP = 30
onsetS = 120

# Set event depth and min/max epicentral distances
# according to phase
# for P, S see Wilson et. al., 2006
# for SKS see Yuan et. al., 2006
# for ScS see Zhang et. al, 2014
maxdepth = {'P': None, 'S': 300, 'SCS': 300, 'SKS': 300}  # km
min_epid = {'P': 28.1, 'S': 55, 'SCS': 50, 'SKS': 90}  # deg
max_epid = {'P': 95.8, 'S': 80, 'SCS': 75, 'SKS': 120}  # deg
