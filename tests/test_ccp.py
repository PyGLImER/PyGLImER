'''
Characterization tests for :mod:`pyglimer.ccp.ccp`.

These tests pin the current behaviour of the CCPStack machinery so that
the planned structural refactoring (Phase 4) cannot silently change
results. They run entirely offline on the packaged example data.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import os
import shutil
import tempfile
import unittest

import numpy as np

from pyglimer.ccp.ccp import CCPStack, read_ccp
from pyglimer.rf import create
from pyglimer.utils.geo_utils import epi2euc


class CCPStackTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.st = CCPStack(
            [42.6, 43.0], [-71.6, -71.0], 2, 'P', verbose=False,
            logdir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestInit(CCPStackTest):
    def test_attributes(self):
        st = self.st
        self.assertEqual(st.bingrid.phase, 'P')
        self.assertEqual(st.bingrid.edist, 2)
        self.assertEqual(st.N, 0)
        self.assertIsNone(st.binrad)
        # Depth axis: 0.1 km sampling above the surface (-10..0), then
        # 1 km sampling below (0..maxz inclusive)
        self.assertEqual(st.z.size, 851)
        self.assertAlmostEqual(st.z[0], -10)
        self.assertAlmostEqual(st.z[99], -0.1)
        self.assertAlmostEqual(st.z[100], 0)
        self.assertAlmostEqual(st.z[-1], 750)
        self.assertEqual(st.bins.shape, (st.coords[0].size, 851))
        self.assertEqual(st.illum.shape, st.bins.shape)
        self.assertEqual(st.illum.dtype, int)
        np.testing.assert_array_equal(st.bins, 0)
        np.testing.assert_array_equal(st.illum, 0)

    def test_logfile_created(self):
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, 'ccp.log')))

    def test_S_phase(self):
        st = CCPStack(
            [42.6, 43.0], [-71.6, -71.0], 2, 's', verbose=False,
            logdir=self.tmpdir)
        self.assertEqual(st.bingrid.phase, 'S')


class TestComputeStackValidation(CCPStackTest):
    def test_binrad_too_small(self):
        with self.assertRaises(ValueError):
            self.st.compute_stack(
                'iasp91.dat', self.tmpdir, in_format='sac', binrad=0.5)

    def test_binrad_too_big(self):
        with self.assertRaises(ValueError):
            self.st.compute_stack(
                'iasp91.dat', self.tmpdir, in_format='sac', binrad=5)


class TestQueryBinTree(CCPStackTest):
    def test_near_and_far(self):
        self.st.binrad = 2.3  # edist * full-overlap binrad
        self.st.binrad_eucl = epi2euc(self.st.binrad)
        # A bin close to the station -> at least one hit
        k, j = self.st.query_bin_tree(
            np.array([42.6]), np.array([-71.6]), 5)
        self.assertGreater(k.size, 0)
        # Far outside the bin grid -> no bins
        k_far, j_far = self.st.query_bin_tree(
            np.array([-45.]), np.array([170.]), 5)
        self.assertEqual(k_far.size, 0)
        self.assertEqual(j_far.size, 0)


class TestProcessRF(CCPStackTest):
    def test_process_rftr(self):
        st = self.st
        st.binrad = 2.3
        st.binrad_eucl = epi2euc(st.binrad)
        rfst = create.read_rf()
        k, j, data = st._ccp_process_rftr(
            rfst, None, 'iasp91.dat', None, None, False, False, 5)
        # Depth-migrated RF has the length of the stack's depth axis
        self.assertEqual(data.shape, (851,))
        # k holds one bin index per finite piercing point (one per depth
        # sample per RF, so it can exceed the depth-axis length); every
        # index must point at an existing bin
        self.assertGreater(k.size, 0)
        self.assertLessEqual(k.max(), st.bins.shape[0])

    def test_process_rftr_append_pp(self):
        st = self.st
        st.binrad = 2.3
        st.binrad_eucl = epi2euc(st.binrad)
        rfst = create.read_rf()
        k, j, data = st._ccp_process_rftr(
            rfst, None, 'iasp91.dat', None, None, False, True, 5)
        self.assertEqual(len(st.pplat), 1)
        self.assertEqual(len(st.pplon), 1)
        self.assertEqual(st.pplat[0].shape, (851,))
        self.assertEqual(st.pplon[0].shape, (851,))

    def test_process_rftr_matches_moveout(self):
        """The migrated data must equal an untapered moveout correction."""
        st = self.st
        st.binrad = 2.3
        st.binrad_eucl = epi2euc(st.binrad)
        rfst = create.read_rf()
        k, j, data = st._ccp_process_rftr(
            rfst, None, 'iasp91.dat', None, None, False, False, 5)
        _, rf2, _, _ = create.read_rf()[0].moveout(
            'iasp91.dat', taper=False)
        np.testing.assert_allclose(data, rf2.data, rtol=1e-12)


class TestConcludeCCP(CCPStackTest):
    def test_simple_weighting(self):
        st = self.st
        nb = st.bins.shape[0]
        rng = np.random.RandomState(42)
        st.bins = rng.rand(nb, 851)
        st.illum = rng.randint(0, 10, (nb, 851))
        st.conclude_ccp(keep_empty=True, keep_water=True)
        expected = np.divide(st.bins, st.illum + 1)
        np.testing.assert_allclose(st.ccp, expected)
        np.testing.assert_array_equal(st.hits, st.illum)

    def test_min_hits(self):
        st = self.st
        nb = st.bins.shape[0]
        st.bins = np.ones((nb, 851))
        st.illum = np.full((nb, 851), 3)
        # Current semantics: bins with 1 < illum < r are zeroed. Note
        # that illum == 1 survives this filter.
        st.illum[0, :] = 2
        st.illum[1, :] = 1
        st.conclude_ccp(keep_empty=True, keep_water=True, r=3)
        np.testing.assert_array_equal(st.ccp[0], 0)
        np.testing.assert_array_equal(st.hits[0], 0)
        np.testing.assert_allclose(st.ccp[1], 1/2)
        # Illumination above r is untouched (bins/(illum + 1))
        np.testing.assert_allclose(st.ccp[2], 1/4)

    def test_multiple_modes(self):
        st = self.st
        nb = st.bins.shape[0]
        st.bins = np.ones((nb, 851))
        st.bins_m1 = np.full((nb, 851), 2.)
        st.bins_m2 = np.full((nb, 851), 3.)
        st.illum = np.ones((nb, 851))
        st.illumm = np.ones((nb, 851))
        z_multiple = 200
        endi = np.where(st.z == z_multiple)[0][0] + 1
        st.conclude_ccp(
            keep_empty=True, keep_water=True, multiple='linear',
            z_multiple=z_multiple)
        exp = (
            np.divide(st.bins[:, :endi], st.illum[:, :endi] + 1)
            + np.divide(st.bins_m1[:, :endi], st.illumm[:, :endi] + 1)
            + np.divide(st.bins_m2[:, :endi], st.illumm[:, :endi] + 1)) / 3
        np.testing.assert_allclose(st.ccp[:, :endi], exp)
        # Below the multiple depth, the stack is zero
        np.testing.assert_array_equal(st.ccp[:, endi:], 0)

    def test_invalid_multiple_mode(self):
        st = self.st
        with self.assertRaises(ValueError):
            st.conclude_ccp(
                keep_empty=True, keep_water=True, multiple='nonsense')

    def test_invalid_z_multiple(self):
        st = self.st
        with self.assertRaises(ValueError):
            st.conclude_ccp(
                keep_empty=True, keep_water=True, z_multiple=250)


class TestWriteRead(CCPStackTest):
    def _populate(self):
        st = self.st
        nb = st.bins.shape[0]
        rng = np.random.RandomState(13)
        st.bins = rng.rand(nb, 851)
        st.illum = rng.randint(0, 10, (nb, 851))
        st.N = 17
        st.binrad = 2.3
        return st

    def test_pickle_roundtrip(self):
        st = self._populate()
        fn = os.path.join(self.tmpdir, 'ccp')
        st.write(fn, fmt='pickle')
        ccp = read_ccp(fn + '.pkl')
        np.testing.assert_allclose(ccp.bins, st.bins)
        np.testing.assert_array_equal(ccp.illum, st.illum)
        self.assertEqual(ccp.N, 17)
        self.assertEqual(ccp.bingrid.phase, 'P')
        self.assertEqual(ccp.bingrid.edist, 2)
        np.testing.assert_allclose(ccp.z, st.z)

    def test_npz_roundtrip(self):
        st = self._populate()
        fn = os.path.join(self.tmpdir, 'ccp')
        st.write(fn, fmt='npz')
        ccp = read_ccp(fn + '.npz')
        np.testing.assert_allclose(ccp.bins, st.bins)
        np.testing.assert_array_equal(ccp.illum, st.illum)
        self.assertEqual(ccp.N, 17)
        self.assertEqual(ccp.bingrid.phase, 'P')
        self.assertEqual(ccp.bingrid.edist, 2)

    def test_unknown_format(self):
        st = self._populate()
        with self.assertRaises(ValueError):
            st.write(os.path.join(self.tmpdir, 'ccp'), fmt='excel')


if __name__ == "__main__":
    unittest.main()
