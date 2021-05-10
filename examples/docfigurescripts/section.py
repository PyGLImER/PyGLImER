import os
from pyglimer.plot.plot_utils import set_mpl_params
from pyglimer.rf.create import read_rf
from pyglimer.plot.plot_utils import plot_section
set_mpl_params()

# Note that these section cannot be easily exported using
# PDF or SVG as a format, since the number of geometric objects in either
# would be too large

# Get outdir
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Read Rfs
rfst = read_rf("../database/waveforms/RF/P/IU/HRV/*.sac")

# Plot section
plot_section(
    rfst, scalingfactor=1,
    outputfile=os.path.join(outdir, "section_raw.png"))

# Plot section with limits
timelimits = (0, 20)  # seconds
epilimits = (32, 36)  # epicentral distance
plot_section(
    rfst, scalingfactor=0.25, linewidth=0.75,
    timelimits=timelimits, epilimits=epilimits,
    outputfile=os.path.join(outdir, "section_limits.png"))
