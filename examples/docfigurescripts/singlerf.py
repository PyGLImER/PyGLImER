from pyglimer.plot.plot_utils import set_mpl_params
from pyglimer.plot.plot_utils import plot_single_rf
from pyglimer.rf.create import read_rf
set_mpl_params()

# Read all RFs from Station IU/HRV
rfst = read_rf("../database/waveforms/RF/P/IU/HRV/*.sac")
N = 753
# Plot RF and save its output.
plot_single_rf(
    rfst[N],
    outputdir="./figures", post_fix="raw", format='svg')

# Plot a single RF using the time limits
plot_single_rf(
    rfst[N], tlim=[0, 20],
    outputdir="./figures", post_fix="timelimit", format='svg')

# Plot single RF using time limits and clean
plot_single_rf(
    rfst[N], tlim=[0, 20], clean=True,
    outputdir="./figures", post_fix="timelimit_clean", format='svg')
