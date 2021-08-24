from .line_buffer import line_buffer
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


def plot_line_buffer(lat, lon, delta: float = 1, ax=None, **kwargs):
    """Takes in lat, lon points and creates circular polygons around it. Merges
    the polygons of possible

    Parameters
    ----------
    lat : np.ndarray
        latitudes of a line
    lon : np.ndarray
        longitudes of a line
    delta : float, optional
        epicentral distance of the buffer, by default 1

    Returns
    -------
    tuple
        (patch, artist)


    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.21 20.00 (Lucas Sawade)

    """

    # Get axes
    if ax is None:
        ax = plt.gca()

    # Get buffer
    poly, circles = line_buffer(lat, lon, delta=delta)

    # Plot into figure
    mpolys = []
    artists = []
    for _poly in poly:
        mpoly = Polygon(_poly, **kwargs)
        mpolys.append(mpoly)
        artists.append(ax.add_patch(mpoly))

    return poly, circles, mpolys, artists
