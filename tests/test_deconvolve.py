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
Last Modified: Wednesday, 25th August 2021 12:21:29 pm
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
        self.assertTrue(np.allclose(r, r2[0:len(r)], atol=0.001))


if __name__ == "__main__":
    unittest.main()
