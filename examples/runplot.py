"""
Example for running the CCP Stack exploration tool.
"""
import numpy as np
from pyglimer.ccp.ccp import read_ccp
from pyglimer.plot.plot_volume import VolumeExploration

# Change filename and folder to adapt to your database
filename = 'ccp_IU_HRV.pkl'
folder = 'output/ccps'

# Load and plot
ccpstack = read_ccp(filename=filename, folder=folder, fmt=None)

# 3D interpolation without all points has drawbacks! See Documentation
ccpstack.explore(maxz=200, minillum=None)
