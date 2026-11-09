'''
Golden (regression) tests.

These tests pin the numerical output of PyGLImER's core algorithms to
stored reference arrays (``tests/golden/*.npz``). They exist to guarantee
that refactoring and dependency upgrades do not silently change scientific
results.

If an intended change of behaviour alters an output, regenerate the
reference files deliberately with::

    PYGLIMER_REGOLDEN=1 pytest tests/test_golden.py

and review the diff of the ``.npz`` files before committing.

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
import unittest
from unittest import mock

import numpy as np
from obspy import read
from scipy.signal.windows import gaussian

from pyglimer.ccp.compute import bin as ccpbin
from pyglimer.rf import create
from pyglimer.rf.deconvolve import it, multitaper as mt
from pyglimer.rf.moveout import dt_table, dt_table_3D
from pyglimer.utils import createvmodel as cvm
from pyglimer.utils import signalproc as sp
from pyglimer.utils.SphericalNN import SphericalNN
from pyglimer.waveform import rotate

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'golden')
REGOLDEN = os.environ.get('PYGLIMER_REGOLDEN') == '1'

# Tolerances: tight enough to catch a single-sample shift, polarity flip
# or small amplitude change, loose enough to absorb compiler/BLAS level
# noise (~1e-13 relative) between platforms.
RTOL = 1e-7
ATOL = 1e-9


def _golden(name, **arrays):
    """
    Save or compare golden arrays. Integer arrays (e.g. KDTree indices)
    are compared exactly, float arrays with RTOL/ATOL (NaNs equal).
    """
    fname = os.path.join(GOLDEN_DIR, name + '.npz')
    if REGOLDEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        np.savez(fname, **arrays)
        return
    if not os.path.isfile(fname):
        raise FileNotFoundError(
            'Golden file %s is missing. Regenerate with '
            'PYGLIMER_REGOLDEN=1 pytest tests/test_golden.py' % fname)
    with np.load(fname) as ref:
        for key, value in arrays.items():
            expected = ref[key]
            if np.issubdtype(np.asarray(value).dtype, np.integer):
                np.testing.assert_array_equal(
                    value, expected, err_msg='golden array %s' % key)
            else:
                np.testing.assert_allclose(
                    value, expected, rtol=RTOL, atol=ATOL,
                    equal_nan=True, err_msg='golden array %s' % key)


def _wavelet_setup():
    """
    Deterministic synthetic: gaussian wavelet g convolved with an impulse
    response r (same construction as tests/test_deconvolve.py).
    """
    g = gaussian(51, 2.5)
    r = np.zeros_like(g)
    r[0] = 1
    r[15] = .25
    s = np.convolve(g, r)[:len(g)]
    return g, s, r


class TestGoldenMoveout(unittest.TestCase):
    def test_moveout_1d(self):
        tr = create.read_rf()[0]
        z2, mo, RFm1, RFm2 = tr.moveout('iasp91.dat')
        self.assertEqual(mo.stats.type, 'depth')
        _golden(
            'moveout_1d', z2=z2, RF2=mo.data,
            pp_lat=np.asarray(mo.stats.pp_latitude),
            pp_lon=np.asarray(mo.stats.pp_longitude))

    def test_moveout_multiple(self):
        tr = create.read_rf()[0]
        z2, mo, RFm1, RFm2 = tr.moveout('iasp91.dat', multiple=True)
        self.assertEqual(mo.stats.type, 'depth')
        _golden(
            'moveout_multiple', z2=z2, RF2=mo.data,
            RFm1=np.asarray(RFm1.data), RFm2=np.asarray(RFm2.data))


class TestGoldenMoveoutTables(unittest.TestCase):
    """Pin dt_table/dt_table_3D directly (moveout() pins them indirectly,
    but only for a single ray parameter/phase combination)."""

    def test_dt_table_1d(self):
        for tag, args in {
            'p_mult': (6.4, 'iasp91.dat', 'P', 500., True),
            's_nomult': (12.0, 'iasp91.dat', 'S', 100., False),
            'p_raysum': (6.4, 'raysum.dat', 'P', 0., False, True),
        }.items():
            htab, dt, delta, m1, m2 = dt_table(*args)
            arrays = {'htab': htab, 'dt': dt, 'delta': delta}
            if m1 is not None:
                arrays['m1'] = m1
            if m2 is not None:
                arrays['m2'] = m2
            _golden('dt_table_1d_' + tag, **arrays)

    def test_dt_table_3d(self):
        # test_createvmodel caches the full GyPSuM model with
        # flatten=False. load_gyps' cache key ignores the flatten flag,
        # so without this reset the submodel here would silently be
        # derived from the non-flattened model and the table would
        # differ depending on test order. Production (dt_table_3D)
        # always uses the default flatten=True.
        cvm._MODEL_CACHE.clear()
        for tag, args in {
            'p': (6.4, 'P', 10., 20., 30., 500., (0., 30.), (10., 40.), True),
            's': (12.0, 'S', 10., 20., 30., 100., (0., 30.), (10., 40.),
                  False),
        }.items():
            htab, dt, delta, m1, m2 = dt_table_3D(*args)
            arrays = {'htab': htab, 'dt': dt, 'delta': delta}
            if m1 is not None:
                arrays['m1'] = m1
            if m2 is not None:
                arrays['m2'] = m2
            _golden('dt_table_3d_' + tag, **arrays)


class TestGoldenMultitaper(unittest.TestCase):
    def test_multitaper_deconvolution(self):
        g, s, _ = _wavelet_setup()
        rf, lrf, _, _ = mt(g, s, 1, 0, "con")
        _golden('deconvolve_multit', rf=rf, lrf=lrf)


class TestGoldenDeconvolve(unittest.TestCase):
    def test_iterative_deconvolution(self):
        g, s, _ = _wavelet_setup()
        rf, nit, IR = it(g, s, 1, omega_min=0.5)
        _golden('deconvolve_it', rf=rf, nit=np.asarray(nit), IR=IR)


class TestGoldenRotate(unittest.TestCase):
    @mock.patch('pyglimer.waveform.rotate.load_avvmodel')
    def test_rotate_PSV(self, load_model_mock):
        model_mock = mock.MagicMock()
        # Fixed velocities and incidence (the old test drew random values)
        avp, avs = 6.0, 3.5
        model_mock.query.return_value = (avp, avs)
        load_model_mock.return_value = model_mock
        st = read()
        st.rotate('NE->RT', back_azimuth=10)
        for tr in st:
            tr.stats.sampling_rate = 1
        iad = -30
        ia = iad*np.pi/180
        rayp = np.sin(ia) * 1/avp
        _, _, st_rot = rotate.rotate_PSV(0, 0, rayp, st, 'P')
        _golden(
            'rotate_psv',
            R=st_rot[0].data, T=st_rot[1].data, Z=st_rot[2].data)


class TestGoldenSignalProc(unittest.TestCase):
    def test_frequency_domain_operations(self):
        g, s, _ = _wavelet_setup()
        nf = 128
        _golden(
            'signalproc',
            convf=sp.convf(g, s, nf, 1),
            corrf=sp.corrf(g, s, nf),
            sshift=sp.sshift(s, nf, 1, 0.3),
            gaussian=sp.gaussian(51, 1, 2.5))


class TestGoldenSpatial(unittest.TestCase):
    def test_sphericalnn(self):
        lat = np.array([10., 10.5, -45.2, 60., 0.1])
        lon = np.array([20., 20.4, 170., -30., -0.1])
        snn = SphericalNN(lat, lon)
        d, i = snn.query(np.array([10.2, -44.]), np.array([20.1, 169.5]))
        pairs = snn.query_pairs(maximum_distance=90.)
        _golden('sphericalnn', dist=d, idx=i, pairs=pairs)

    def test_bingrid(self):
        lat = np.array([10., 10.5, -45.2, 60., 0.1])
        lon = np.array([20., 20.4, 170., -30., -0.1])
        bg = ccpbin.BinGrid(lat, lon, 1, 'P', verbose=False)
        binlat, binlon = bg.compute_bins()
        bg.bin_tree()
        eucd, idx = bg.query_station_tree(
            np.array([10.2, -44.]), np.array([20.1, 169.5]), 2)
        nb = bg.query_bin_tree(
            np.array([10.2, -44.]), np.array([20.1, 169.5]), 5e4, 3)
        _golden(
            'bingrid', binlat=binlat, binlon=binlon, eucd=eucd, idx=idx,
            nb=nb)


if __name__ == "__main__":
    unittest.main()
