'''
Tests the pyglimer.rf.deconvolve module

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 12th March 2021 10:13:35 am
Last Modified: Thursday, 21st October 2021 03:28:53 pm
'''
import unittest

import numpy as np
from scipy.signal.windows import gaussian

from pyglimer.rf.deconvolve import it, spectraldivision


class TestIt(unittest.TestCase):
    """
    A class to test the iterative time domain deconvolution.
    """

    def test_decon(self):
        """
        Convolution with subsequent deconvolution.
        """
        # The incoming wavelet
        g = gaussian(51, 2.5)

        # Impulse response
        r = np.zeros_like(g)
        r[0] = 1
        r[15] = .25

        # convolve the two to a signal
        s = np.convolve(g, r)[:len(g)]

        # Deconvolve
        _, _, r2 = it(g, s, 1, omega_min=0.5)

        # test result
        self.assertTrue(np.allclose(r, r2[0:len(r)], atol=0.0001))

    def test_it_max(self):
        """
        Convolution with subsequent deconvolution. One Iteration should
        only recover the largest peak.
        """
        # The incoming wavelet
        g = gaussian(51, 2.5)

        # Impulse response
        r = np.zeros_like(g)
        r[0] = 1
        r[15] = .25

        # convolve the two to a signal
        s = np.convolve(g, r)[:len(g)]

        # Deconvolve
        _, _, r2 = it(g, s, 1, it_max=1, omega_min=0.5)

        # test result
        self.assertFalse(np.allclose(r, r2[0:len(r)], atol=0.1))
        self.assertAlmostEqual(r[0], r2[0], places=4)

    def test_it_max_2(self):
        """
        Convolution with subsequent deconvolution. two Iterations recovers the
        whole reflectivity series.
        """
        # The incoming wavelet
        g = gaussian(51, 2.5)

        # Impulse response
        r = np.zeros_like(g)
        r[0] = 1
        r[15] = .25

        # convolve the two to a signal
        s = np.convolve(g, r)[:len(g)]

        # Deconvolve
        _, _, r2 = it(g, s, 1, it_max=2, omega_min=0.5)

        # test result
        self.assertTrue(np.allclose(r, r2[0:len(r)], atol=0.0001))


class TestSpectralDivision(unittest.TestCase):
    def test_unknown_regul(self):
        with self.assertRaises(ValueError):
            spectraldivision(
                np.empty((25,)), np.empty((25,)), 1, 0, 'blabla', 'P')

    def test_unknown_phase(self):
        with self.assertRaises(ValueError):
            spectraldivision(
                np.empty((25,)), np.empty((25,)), 1, 0, 'wat', 'XX')

    def test_by_self_con_P(self):
        v = np.zeros((200,))
        v[50:] = np.random.random((150,))
        qrf, lrf = spectraldivision(v, v, 0.1, 5, 'con', 'P')
        self.assertTrue(np.allclose(qrf, lrf))
        self.assertTrue(np.argmax(qrf), 50)

    def test_by_self_wat_P(self):
        v = np.zeros((200,))
        v[50:] = np.random.random((150,))
        qrf, lrf = spectraldivision(v, v, 0.1, 5, 'wat', 'P')
        self.assertTrue(np.allclose(qrf, lrf))
        self.assertTrue(np.argmax(qrf), 50)

    def test_by_self_fqd_P(self):
        v = np.zeros((200,))
        v[50:] = np.random.random((150,))
        qrf, _ = spectraldivision(v, v, 0.1, 5, 'fqd', 'P')
        self.assertTrue(np.argmax(qrf), 50)

    def test_by_self_con_S(self):
        v = np.zeros((200,))
        v[50:] = np.random.random((150,))
        qrf, lrf = spectraldivision(v, v, 0.1, 5, 'con', 'S')
        self.assertTrue(np.allclose(qrf, lrf))
        self.assertTrue(np.argmax(qrf), 50)

    def test_by_self_wat_S(self):
        v = np.zeros((200,))
        v[50:] = np.random.random((150,))
        qrf, lrf = spectraldivision(v, v, 0.1, 5, 'wat', 'S')
        self.assertTrue(np.allclose(qrf, lrf))
        self.assertTrue(np.argmax(qrf), 50)

    def test_by_self_fqd_S(self):
        v = np.zeros((200,))
        v[50:] = np.random.random((150,))
        qrf, _ = spectraldivision(v, v, 0.1, 5, 'fqd', 'S')
        self.assertTrue(np.argmax(qrf), 50)


if __name__ == "__main__":
    unittest.main()
