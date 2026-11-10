'''
3D volume handling for the CCP stack: KDTree interpolation onto regular
grids, VTK mesh export, and the interactive/static volume plots
(implemented as a mixin so that the stacking core stays free of pyvista
and vtk imports).

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import sys
from functools import partial
from copy import deepcopy

import numpy as np
import pyvista as pv
import vtk
from scipy.interpolate import RegularGridInterpolator

from pyglimer.constants import DEG2KM, KM2DEG, R_EARTH
from pyglimer.plot.plot_volume import VolumePlot, VolumeExploration
from pyglimer.utils.geo_utils import fix_map_extent, geo2cart
from pyglimer.utils.SphericalNN import SphericalNN


class CCPStackVolumeMixin(object):
    """Volume computation and plotting methods for the
    :class:`~pyglimer.ccp.ccp.CCPStack`. Only to be used as a mixin."""

    def compute_kdtree_volume(
        self, qlon: np.ndarray or list = None,
        qlat: np.ndarray or list = None, zmax: float = None,
            verbose: bool = True):
        """Using the CCP kdtree, we get the closest few points and compute
        the weighting using a distance metric. if points are too far away,
        they aren't weighted


        Parameters
        ----------
        qlon : np.ndarray or list
            grid defining array for longitude
        qlat : np.ndarray or list
            grid defining array for latitude
        zmax : float, optional
            maximum depth to compute.
        gz : np.array or list
            grid defining array for z
        r : float or None
            outside r everything is nan
        minillum: int or None, optional
            Minimum number of illumation points use in the interpolation,
            everything below is downweighted by the square reciprocal

        Returns
        -------
        [type]
            [description]
        """

        def vprint(verbose, *args, **kwargs):
            print(*args, **kwargs)
        verboseprint = partial(vprint, verbose)

        # Area considered around profile points
        area = 2 * self.binrad

        # Get global bounds
        if qlat is None or qlon is None:

            minlat = np.min(self.coords_new[0])
            maxlat = np.max(self.coords_new[0])
            minlon = np.min(self.coords_new[1])
            maxlon = np.max(self.coords_new[1])
            minlon, maxlon, minlat, maxlat = fix_map_extent(
                [minlon, maxlon, minlat, maxlat, ])

            # Create mesh vectors
            qlat = np.arange(minlat, maxlat + self.binrad/2, self.binrad/2)
            qlon = np.arange(minlon, maxlon + self.binrad/2, self.binrad/2)

        else:

            minlat, maxlat = np.min(qlat), np.max(qlat)
            minlon, maxlon = np.min(qlon), np.max(qlon)
            minlon, maxlon, minlat, maxlat = fix_map_extent(
                [minlon, maxlon, minlat, maxlat])

        # Create mesh
        mlat, mlon = np.meshgrid(qlat, qlon)

        # smaller/larger is fine as the extent was fixed.
        cpos = np.where(
            (self.coords_new[0] > minlat)
            & (self.coords_new[0] < maxlat)
            & (self.coords_new[1] > minlon)
            & (self.coords_new[1] < maxlon)
        )[1]

        # Get interpolation weights and rows.
        verboseprint("Creating Spherical Nearest Neighbour class ...", end="")
        snn = SphericalNN(
            self.coords_new[0][:, cpos], self.coords_new[1][:, cpos])
        verboseprint(" done.")
        verboseprint("Creating interpolators ...", end="")
        ccp_interpolator = snn.interpolator(
            mlat, mlon, maximum_distance=area, k=10, p=2.0, no_weighting=False)
        ill_interpolator = snn.interpolator(
            mlat, mlon, maximum_distance=area, no_weighting=True)
        verboseprint(" done.")
        # Get coordinates array from CCPStack
        qz = deepcopy(self.z)

        if zmax is not None:
            pos = np.argmin(np.abs(qz-zmax))
            qz = qz[:pos]
        else:
            pos = len(qz)

        # Get data arrays
        Nz = pos
        Nlat, Nlon = len(qlat), len(qlon)

        # Interpolate
        qccp = np.zeros((Nlat, Nlon, Nz))
        qill = np.zeros((Nlat, Nlon, Nz))

        # Interpolate each depth
        maintext = "KDTree interpolation ... (intensive part)"
        verboseprint(maintext)
        for _i, (_ccpcol, _illumcol) in enumerate(
                zip(self.ccp[cpos, :pos].T, self.hits[cpos, :pos].T)):
            if _i % int((Nz/10)) == 0:
                sys.stdout.write("\033[F")  # back to previous line
                sys.stdout.write("\033[K")
                verboseprint(
                    maintext + f"--->  {_i+1:0{len(str(Nz))}d}/{Nz:d}")
            qccp[:, :, _i] = ccp_interpolator(_ccpcol).T
            qill[:, :, _i] = ill_interpolator(_illumcol).T
        verboseprint("... done.")

        # Interpolate onto regular depth grid for easy representation with
        # imshow
        verboseprint(
            "Regular Grid interpolation ... (less intensive part)", end="")
        ccp3D_interpolator = RegularGridInterpolator(
            (qlat, qlon, qz), np.where(np.isnan(qccp), 0, qccp))
        ill3D_interpolator = RegularGridInterpolator(
            (qlat, qlon, qz), qill)

        # Where to sample
        qqz = np.arange(np.min(qz), np.max(qz), 1)
        xqlat, xqlon, xqz = np.meshgrid(qlat, qlon, qqz)

        # Interpolate
        qccp = ccp3D_interpolator((xqlat, xqlon, xqz))
        qill = ill3D_interpolator((xqlat, xqlon, xqz))
        verboseprint(" done.")

        return qlat, qlon, qqz, qill, qccp, area

    def create_vtk_mesh(
            self, geo=True, bbox: list = None, filename: str = None):
        """Creates a mesh with given bounding box s

        Parameters
        ----------
        geo : bool, optional
            flag whether the output mesh is in geographical coordinates or
            meters, by default True
        bbox : list or None, optional
            bounding box [minlon, maxlon, minlat, maxlat]. If None
            No boundaries are taken, by default None
        filename : str or None, optional
            If set, the computed grid will be output as VTK file under the
            given filename. This file can then later be opened using either the
            plotting tool or, e.g., Paraview. If None, no file is written.
            By default, None.

        Returns
        -------
        VTK.UnstructuredGrid
            outputs a vtk mesh that can be opened in, e.g., Paraview.
        """

        # Get coordinates
        lat = np.squeeze(self.coords_new[0])
        lon = np.squeeze(self.coords_new[1])

        # Filter ccpstacks if not none
        if bbox is None:
            bbox = [-180, 180, -90, 90]
        pos = np.where(((
            bbox[0] <= lon) & (lon <= bbox[1])
            & (bbox[2] <= lat) & (lat <= bbox[3])))[0]
        lat = lat[pos]
        lon = lon[pos]

        # Create VTK point cloud at the surface to triangulate.
        r = R_EARTH * np.ones_like(lat)
        points = np.vstack((deepcopy(lon*DEG2KM).flatten(),
                            deepcopy(lat*DEG2KM).flatten(),
                            deepcopy(r).flatten())).T
        pc = pv.PolyData(points)

        # Triangulate 2D surface
        mesh = pc.delaunay_2d(alpha=self.binrad*1.5*DEG2KM)

        # Use triangles and their connectivity to create 3D Mesh of wedges
        points = deepcopy(mesh.points)
        n_points = mesh.n_points

        # Get cells and create first layer of wedges at the surface
        cells = mesh.faces.reshape(mesh.n_cells, 4)
        cells[:, 0] = 6
        cells = np.hstack((cells, n_points + cells[:, 1:]))
        newcells = deepcopy(cells)

        # Give second layer of points in the wedge the right depth!
        zpoints = deepcopy(np.array(points))
        zpoints[:, 2] = R_EARTH - self.z[1]
        newpoints = np.vstack((points, zpoints))

        # Loop over remaining depths to populated the mesh.
        for _z in self.z[2:]:

            # Add cells
            extra_cells = cells
            extra_cells[:, 1:] += n_points
            newcells = np.vstack((newcells, extra_cells))

            # Add points
            zpoints = deepcopy(np.array(points))
            zpoints[:, 2] = R_EARTH - _z
            newpoints = np.vstack((newpoints, zpoints))

        # Define Cell types
        newcelltypes = np.array(
            [vtk.VTK_WEDGE] * newcells.shape[0], dtype=np.uint8)

        # Redefine location of the points if! Geo location is wanted instead of
        # Cartesian(-ish)
        if geo:
            x, y, z = geo2cart(
                newpoints[:, 2],
                newpoints[:, 1] * KM2DEG,
                newpoints[:, 0] * KM2DEG)

            newpoints = np.vstack((x, y, z)).T

        # Create Unstructured Grid
        grid = pv.UnstructuredGrid(newcells, newcelltypes, newpoints)

        # Populate with RF and illumination values
        grid['RF'] = deepcopy(self.ccp[pos, :].T.ravel())
        grid['illumination'] = deepcopy(self.hits[pos, :].T.ravel())

        # If file name is set write unstructured grid to file!
        if filename is not None:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetInputData(grid)
            writer.SetFileName(filename)
            writer.Write()

        return grid

    def explore(
        self, qlon: np.ndarray or list = None, qlat: np.ndarray or list = None,
            zmax: float = None):
        """
        Creates a volume exploration window set. One window is for all
        plots, and the other window is generated with sliders such that one
        can explore how future plots should be generated. It technically does
        not require any inputs, as simply the binning size will be used to
        create a meshgrid fitting to the bin distance distribution.
        One can however fix the extent and size of the grid by defining the
        grids coordinate vectors.

        Parameters
        ----------

        qlon: np.ndarray, optional
            Longitude gridpoints for the Grid. Defines longitudinal extent
            and resolution. If None, it's determined automatically from
            the CCPStacks resolution and grid extent.
            Must be monotonically increasing.
        qlat: np.ndarray, optional
            Latitude gridpoints for the Grid. Defines latitudinal extent
            and resolution. If None, it's determined automatically from
            the CCPStacks resolution and grid extent.
            Must be monotonically increasing.
        zmax: float, optional
            Maximum depth (km) that the CCP-exploration extends to.

        Returns
        -------

        VolumeExploration

        """
        # Compute the volume max radius at depth is z*0.33
        qlat, qlon, qz, qill, qccp, area = self.compute_kdtree_volume(
            qlon, qlat, zmax=zmax)

        # Launch plotting tool
        return VolumeExploration(qlon, qlat, qz, qccp)

    def plot_volume_sections(
        self, qlon: np.ndarray, qlat: np.ndarray, zmax: float or None = None,
        lonsl: float or None = None, latsl: float or None = None,
        zsl: float or None = None, r: float or None = None,
            minillum: int or None = None, show: bool = True):
        """
        Creates the same plot as the `explore` tool, but(!) statically and
        with more options left to the user.

        !!! IT IS IMPORTANT for the plotting function that the input arrays
        are monotonically increasing!!!

        Parameters
        ----------
        qlon : np.ndarray
            Monotonically increasing ndarray. No failsafes implemented.
        qlat : np.ndarray
            Monotonically increasing ndarray. No failsafes implemented.
        z : np.ndarray
            Monotonically increasing ndarray. No failsafes implemented.
        lonsl : float or None, optional
            Slice location, if None, the center of lon is used,
            by default None
        latsl : float or None, optional
            Slice location, if None, the center of lat is used,
            by default None
        zsl : float or None, optional
            Slice location, if None, the center of depth is used,
            by default None
        r : float or None, optional
            Max radius for interpolation values taken into account [km],
            if None bingrid.edist * DEG2KM * 2.0, by default None
        minillum: int or None, optional
            Minimum number of illumation points use in the interpolation,
            everything below is downweighted by the square reciprocal
        show : bool, optional
            whether to make figure window visible, default True

        Returns
        -------
        VolumePlot
            [description]
        """

        # Change distance to delete things
        if r is None:
            r = self.bingrid.edist * DEG2KM * 2.0

        # Compute the volume max radius at depth is z*0.33
        qlat, qlon, qz, qill, qccp, area = self.compute_kdtree_volume(
            qlon, qlat, zmax=zmax)

        return VolumePlot(qlon, qlat, qz, qccp, xl=lonsl, yl=latsl, zl=zsl,
                          show=show)
