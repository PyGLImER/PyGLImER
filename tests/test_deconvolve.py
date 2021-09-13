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
Last Modified: Monday, 13th September 2021 04:31:07 pm
'''
import unittest

import numpy as np
from numpy import testing
from scipy.signal.windows import gaussian

from pyglimer.rf.deconvolve import it, spectraldivision, multitaper


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


# class TestSpectralDivision(unittest.TestCase):
    


if __name__ == "__main__":
    unittest.main()
