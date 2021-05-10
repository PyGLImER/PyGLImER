import os
import numpy as np
from pyglimer.ccp.ccp import read_ccp
from pyglimer.plot.plot_utils import set_mpl_params
import matplotlib.pyplot as plt
set_mpl_params()


# Get outdir
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Read the CCP Stack
ccpstack = read_ccp(filename='../ccp_IU_HRV.pkl', fmt=None)

# Create points waypoints for the cross section
lat0 = np.array([42.5, 42.5])
lon0 = np.array([-72.7, -69.5])
lat1 = np.array([41.5, 43.5])
lon1 = np.array([-71.45, -71.45])

# Set RF boundaries
mapextent = [-73.5, -69, 41, 44]
depthextent = [0, 200]
vmin = -0.1
vmax = 0.1

# Plot cross sections
ax1, geoax = ccpstack.plot_cross_section(
    lat0, lon0, z0=23, vmin=vmin, vmax=vmax,
    mapplot=True, minillum=1, label="A",
    depthextent=depthextent,
    mapextent=mapextent)
ax2, _ = ccpstack.plot_cross_section(
    lat1, lon1, vmin=vmin, vmax=vmax,
    geoax=geoax, mapplot=True,
    minillum=1, label="B",
    depthextent=depthextent
)
mapfig = geoax.figure
fig1 = ax1.figure
fig2 = ax2.figure

# Resize Map fig
mapfig.set_size_inches(8, 5, forward=True)

mapfig.savefig(
    os.path.join(outdir, "cross_section_map.png"), dpi=300)
fig1.savefig(
    os.path.join(outdir, "cross_section_A.png"), dpi=300
)
fig2.savefig(
    os.path.join(outdir, "cross_section_B.png"), dpi=300
)
plt.show(block=True)
