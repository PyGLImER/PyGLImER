'''
Receiver-function ingestion backends for the CCP stack.

This module hosts the :class:`CCPStackIngestMixin`, which implements
reading the hdf5 receiver-function database, the per-trace depth
migration/piercing-point worker, and the stacking backend dispatch
(joblib multiprocessing and MPI). The methods are mixed into
:class:`pyglimer.ccp.ccp.CCPStack`, so the public API is unchanged.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)
    Lucas Sawade (lsawade@princeton.edu)

Created: Thursday, 6th November 2025
'''
import os
from typing import List, Tuple

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from pyglimer.database.rfh5 import RFDataBase
from pyglimer.rf.create import RFStream
from pyglimer.rf.moveout import maxzm
from pyglimer.utils.createvmodel import ComplexModel
from pyglimer.utils.log import create_mpi_logger


class CCPStackIngestMixin(object):
    """Mixin that adds the RF database ingestion and stacking backends
    to :class:`pyglimer.ccp.ccp.CCPStack`."""

    def _ccp_process_rftr(
        self, rfst: RFStream, filt: Tuple[float, float], vmodel: str,
        latb: Tuple[float, float], lonb: Tuple[float, float], multiple: bool,
        append_pp: bool, n_closest_points: int) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """
        Do the depth migration and the piercing point computation for
        a single receiver function. Then, find the laterally closest bin
        to the ppoint.

        :param rfst: Input receiver function
        :type rfst: RFStream
        :param filt: Filter the receiver function? If so, provide high- and
            lowpass frequency as a Tuple (in Hz).
        :type filt: Tuple[float, float]
        :param vmodel: The velocity model to use.
        :type vmodel: str
        :param latb: boundaries for the moveout correction in the form
            (minlat, maxlat).
        :type latb: Tuple[float, float]
        :param lonb: boundaries for the moveout correction in the form
            (minlon, maxlon).
        :type lonb: Tuple[float, float]
        :param multiple: Should the moveout correction be done for crustal
            multiples as well?
        :type multiple: bool
        :param append_pp: Should the coordinates of the ppoint be saved?
        :type append_pp: bool
        :param n_closest_points: Number of closest bin points to return.
        :type n_closest_points: int
        :return: The indices of the closest bins k and the depth index j.
            the Receiver fnction data. If multiples also multiple data.
        :rtype: Tuple[np.ndarray[int], np.ndarray[int], np.ndarray]
        """
        if filt:
            rfst.filter(
                'bandpass', freqmin=filt[0], freqmax=filt[1],
                zerophase=True, corners=2)
        try:
            z, rf, rfm1, rfm2 = rfst[0].moveout(
                vmodel, latb=latb, lonb=lonb, taper=False,
                multiple=multiple)
        except ComplexModel.CoverageError as e:
            # Wrong stations codes can raise this
            self.logger.warning(e)
            return
        except (ValueError, IndexError, np.linalg.LinAlgError) as e:
            # Interpolation/migration failures on malformed RFs. Skip the
            # RF, but log with traceback so systematic failures remain
            # visible; anything else propagates and kills the stack loudly.
            self.logger.exception(
                'Failed to migrate RF %s.%s: %s',
                rfst[0].stats.network, rfst[0].stats.station, e)
            return

        lat = np.array(rf.stats.pp_latitude)
        lon = np.array(rf.stats.pp_longitude)
        # Above-surface depths have no piercing points (NaN); they can
        # neither be binned nor stacked.
        m = np.isfinite(lat) & np.isfinite(lon)
        lat = lat[m]
        lon = lon[m]
        if append_pp:
            plat = np.pad(
                lat, (0, len(self.z)-len(lat)), constant_values=np.nan)
            plon = np.pad(
                lon, (0, len(self.z)-len(lon)), constant_values=np.nan)
            self.pplat.append(plat)
            self.pplon.append(plon)
        k, j = self.query_bin_tree(lat, lon, n_closest_points)

        if multiple:
            depthi = np.where(z == maxzm)[0][0]
            try:
                lm1 = (rfm1.data[:depthi+1])
                lm2 = (rfm2.data[:depthi+1])
            except AttributeError as e:
                # for Interpolationerrors
                self.logger.exception(
                    'Error in multiple computation. Mostly caused by issue '
                    + 'during the interpolation. Origninal Error message '
                    + 'below.')
                self.logger.exception(e)
                lm1 = None
                lm2 = None
            return k, j, rf.data, lm1, lm2

        return k, j, rf.data

    def _unpack_joblib_output(self, multiple: bool, endi: int, out: Tuple):
        """
        Unpack the output that comes from joblib and add stack them in the CCP
        object.
        """
        if multiple:
            for kk, jj, datal, datalm1, datalm2, N in out:
                self._unpack_output_multiple(
                    kk, jj, datal, datalm1, datalm2, endi)
                self.N += N
        else:
            for kk, jj, datal, _, _, N in out:
                self._unpack_output(kk, jj, datal)
                self.N += N

    def _unpack_output_multiple(
        self, kk: List[int], jj: List[int], datal: List[float],
            datalm1: List[float], datalm2: List[float], endi: int):
        """
        Unpack the output of the kdtree and stack them into the CCP object.
        Use this function if `multiple=True`

        .. note::
            The RF count ``self.N`` is accounted by the callers
            (``_unpack_joblib_output`` and the MPI loop) from the RF count
            returned by ``_create_ccp_from_hdf5``. This method must not
            modify it.
        """
        for k, j, data, datam1, datam2 in zip(kk, jj, datal, datalm1, datalm2):
            self.bins[k, j] = self.bins[k, j] + data[j]

            # hit counter + 1
            self.illum[k, j] = self.illum[k, j] + 1

            # multiples
            iii = np.where(j <= endi-1)[0]
            jm = j[iii]
            km = k[iii]
            try:
                self.bins_m1[
                    km, jm] = self.bins_m1[km, jm] + datam1[jm]
                self.bins_m2[
                    km, jm] = self.bins_m2[km, jm] + datam2[jm]
                self.illumm[km, jm] = self.illum[km, jm] + 1
            except IndexError as e:
                if not len(datam1) or not len(datam2):
                    continue
                else:
                    raise IndexError(e)

    def _unpack_output(self, kk: List[int], jj: List[int], datal: List[float]):
        """
        Unpack the output of the kdtree and stack them into the CCP object.
        Use this function if `multiple=False`.

        .. note::
            The RF count ``self.N`` is accounted by the callers
            (``_unpack_joblib_output`` and the MPI loop) from the RF count
            returned by ``_create_ccp_from_hdf5``. This method must not
            modify it.
        """
        for k, j, data in zip(kk, jj, datal):
            self.bins[k, j] = self.bins[k, j] + data[j]

            # hit counter + 1
            self.illum[k, j] = self.illum[k, j] + 1

    def _create_ccp_from_hdf5(
        self, filename: str, multiple: bool, append_pp: bool,
        n_closest_points: int, vel_model: str, latb: Tuple[float, float],
        lonb: Tuple[float, float], filt: Tuple[float, float]) -> Tuple[
            List[np.ndarray], List[np.ndarray], List[np.ndarray],
            List[np.ndarray], List[np.ndarray]]:
        """
        Does ppoint calculation and depth migration for rf-functions in hdf5
        format. (per file based)
        Subsequently, computes the closest gridpoints in the ccp object.

        :param filename: HDF5 file to read.
        :type filename: str
        :param multiple: Regard crustal multiples for stack?
        :type multiple: bool
        :param append_pp: Append ppoints to the object?
        :type append_pp: bool
        :param n_closest_points: number of closest points to query from kd-tree
        :type n_closest_points: int
        :param vel_model: The velocity mdoel to use for depth migration.
        :type vel_model: str
        :param latb: Boundary for the CCP calculation in form
            `(minlat, maxlat)`.
        :type latb: Tuple[float, float]
        :param lonb: Boundary for the CCP calculation in form
            `(minlon, maxlon)`.
        :type lonb: Tuple[float, float]
        :param filt: Filter to impose on rfs before migration. Either False
            or tuple in form (minfreq, maxfreq).
        :type filt: Tuple[float, float]
        :return: (List of lateral indices of closest bins,  List of depth
            indices of closest gridpoint, list of rf data to stack for each
            of those indices, list of multiple data (pps), list of multiple
            data (pss))
        :rtype: Tuple[ List[np.ndarray], List[np.ndarray], List[np.ndarray],
            List[np.ndarray], List[np.ndarray]]
        """
        net, stat, _ = os.path.basename(filename).split('.')
        kk = []
        jj = []
        datal = []
        datalm1 = []
        datalm2 = []
        # number of rfs
        N = 0
        with RFDataBase(filename, mode='r') as rfdb:
            for rftr in rfdb.walk('rf', net, stat, self.bingrid.phase):
                rfst = RFStream([rftr])
                out = self._ccp_process_rftr(
                    rfst, filt, vel_model, latb, lonb, multiple, append_pp,
                    n_closest_points)
                if not out:
                    continue
                N += 1
                kk.append(out[0])
                jj.append(out[1])
                datal.append(out[2])
                if multiple:
                    datalm1.append(out[3])
                    datalm2.append(out[4])
        return kk, jj, datal, datalm1, datalm2, N

    def _create_ccp_from_hdf5_mc(
        self, files: List[str], multiple: bool, append_pp: bool,
        n_closest_points: int, vel_model: str, latb: Tuple[float, float],
        lonb: Tuple[float, float], filt: Tuple[float, float], endi,
            mc_backend: str, n_jobs: int = -1):
        """
        Computes a CCP stack by reading a rf database in hdf5. Dispatches
        to the joblib multiprocessing or the MPI backend; the per-file work
        is done by :meth:`_create_ccp_from_hdf5`.

        For a description of the parameters, see
        :meth:`_create_ccp_from_hdf5` (per-file parameters), and:

        :param files: List of hdf5 files to read.
        :type files: List[str]
        :param endi: Only relevant if `multiple=True`. Last depth index to
            consider for multiple computation.
        :type endi: int
        :param mc_backend: Multi-core backend to use, either `'joblib'`
            (multiprocessing) or `'MPI'` (has to be started via mpirun).
        :type mc_backend: str
        :param n_jobs: Number of parallel jobs for the joblib backend.
            -1 uses all cores, defaults to -1.
        :type n_jobs: int, optional
        """
        if mc_backend.lower() == 'joblib':
            out = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                delayed(self._create_ccp_from_hdf5)(
                    filename, multiple, append_pp, n_closest_points,
                    vel_model, latb, lonb, filt)
                for filename in tqdm(files))

            self._unpack_joblib_output(multiple, endi, out)

        elif mc_backend.upper() == 'MPI':
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            psize = comm.Get_size()
            rank = comm.Get_rank()
            pmap = (np.arange(len(files))*psize)/len(files)
            pmap = pmap.astype(np.int32)
            ind = pmap == rank
            ind = np.arange(len(files), dtype=int)[ind]

            # get new MPI compatible loggers
            self.logger = create_mpi_logger(self.logger, rank)
            for ii in ind:
                try:
                    kk, jj, datal, datalm1, datalm2, N =\
                        self._create_ccp_from_hdf5(
                            files[ii], multiple, append_pp, n_closest_points,
                            vel_model, latb, lonb, filt)
                except KeyError:
                    self.logger.warning(
                        f'No Receiver Functions in file {files[ii]}.')
                    continue
                self.N += N
                if multiple:
                    self._unpack_output_multiple(
                        kk, jj, datal, datalm1, datalm2, endi)
                else:
                    self._unpack_output(kk, jj, datal)

            # Gather results. mpi4py's buffer-map collectives infer the
            # datatype from the numpy array, which keeps the integer hit
            # counters intact (the former explicit MPI.DOUBLE declarations
            # mismatched the int dtype of illum/illumm).
            self.N = comm.allreduce(self.N, op=MPI.SUM)
            self.illum = comm.allreduce(self.illum, op=MPI.SUM)
            self.bins = comm.allreduce(self.bins, op=MPI.SUM)
            if multiple:
                self.illumm = comm.allreduce(self.illumm, op=MPI.SUM)
                self.bins_m1 = comm.allreduce(self.bins_m1, op=MPI.SUM)
                self.bins_m2 = comm.allreduce(self.bins_m2, op=MPI.SUM)

        self.logger.info('Number of receiver functions used: '+str(self.N))

