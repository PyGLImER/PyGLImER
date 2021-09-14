'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 25th August 2021 11:54:15 am
Last Modified: Thursday, 26th August 2021 08:49:01 am
'''

import unittest

import numpy as np
from obspy import read
import scipy

from pyglimer.utils import signalproc as sptb


class TestResampleOrDecimate(unittest.TestCase):
    def test_decimate(self):
        st = read()
        freq_new = st[0].stats.sampling_rate//4
        st_filt = sptb.resample_or_decimate(st, freq_new)
        self.assertEqual(st_filt[0].stats.sampling_rate, freq_new)
        self.assertIn('decimate', st_filt[0].stats.processing[-1])
        self.assertIn('filter', st_filt[0].stats.processing[-2])

    def test_resample(self):
        st = read()
        freq_new = st[0].stats.sampling_rate/2.5
        st_filt = sptb.resample_or_decimate(st, freq_new, filter=False)
        self.assertEqual(st_filt[0].stats.sampling_rate, freq_new)
        self.assertIn('resample', st_filt[0].stats.processing[-1])
        self.assertIn('no_filter=True', st_filt[0].stats.processing[-1])

    def test_new_freq_higher_than_native(self):
        st = read()
        freq_new = st[0].stats.sampling_rate+5
        with self.assertRaises(ValueError):
            _ = sptb.resample_or_decimate(st, freq_new, filter=False)


class TestConvf(unittest.TestCase):
    # Note that those functions only compute the positions with full overlap
    # so they have to be padded to compute a "full" convolution
    def test_result(self):
        u = np.random.rand(np.random.randint(50, 100))
        v = np.random.rand(np.random.randint(50, 100))
        exp = np.convolve(u, v)
        res = sptb.convf(u, v, len(u)+len(v)-1, 1)
        self.assertTrue(np.allclose(res, exp))

    def test_zero_len(self):
        with self.assertRaises(ValueError):
            sptb.convf(np.empty(5), np.empty(5), 0, 1)

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            sptb.convf(np.empty(0), np.empty(5), 10, 1)


class TestCorrf(unittest.TestCase):
    def test_result(self):
        u = np.random.rand(np.random.randint(50, 100))
        v = np.random.rand(np.random.randint(50, 100))
        test = scipy.signal.correlate(u, v)
        res = sptb.corrf(u, v, len(u)+len(v)-1)
        # Outputs are phase shifted, so we have to use is in
        self.assertTrue(np.all(np.isin(np.round(test, 6), np.round(res, 6))))

    def test_zero_len(self):
        with self.assertRaises(ValueError):
            sptb.corrf(np.empty(5), np.empty(5), 0)

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            sptb.corrf(np.empty(0), np.empty(5), 10)


class TestGaussian(unittest.TestCase):
    def test_len0(self):
        with self.assertRaises(ZeroDivisionError):
            sptb.gaussian(0, 1, 2.5)

    def test_width0(self):
        with self.assertRaises(ValueError):
            sptb.gaussian(100, 1, 0)


class TestSShift(unittest.TestCase):
    def test_result_forwards(self):
        s = np.random.rand(np.random.randint(100, 150))
        phi = np.random.randint(-50, 50)
        out = sptb.sshift(s, len(s), 1, phi)

        self.assertTrue(np.allclose(out, np.roll(s, phi)))

    def test_shift_0(self):
        s = np.random.rand(np.random.randint(100, 150))
        phi = 0
        out = sptb.sshift(s, len(s), 1, phi)
        self.assertTrue(np.allclose(out, s))

    def test_empty_array(self):
        s = np.array([])
        phi = 0
        out = sptb.sshift(s, len(s), 1, phi)
        with self.assertRaises(ValueError):
            self.assertTrue(np.allclose(out, s))


if __name__ == "__main__":
    unittest.main()
