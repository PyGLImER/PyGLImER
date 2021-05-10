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

# Create  binning functions
lats = np.arange(41, 43.5, 0.05)
lons = np.arange(-72.7, -69.5, 0.05)
z = np.linspace(-10, 200, 211)

# Plot volume slices: vplot is a VolumePlot object
vplot = ccpstack.plot_volume_sections(
    lons, lats, zmax=211, lonsl=-71.45, latsl=42.5, zsl=23,
    show=False)
fig = vplot.ax['x'].figure
fig.set_size_inches(8, 8, forward=True)
plt.savefig(os.path.join(outdir, "ccp_volume.png"), dpi=300, transparent=True)

