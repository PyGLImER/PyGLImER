'''
Tests the pyglimer.rf.deconvolve module

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 12th March 2021 10:13:35 am
Last Modified: Friday, 12th March 2021 12:57:19 pm
'''
import random
import unittest

import numpy as np
from numpy import testing
from scipy.signal.windows import boxcar,gaussian

from pyglimer.rf.deconvolve import it, spectraldivision, gen_it, multitaper
from pyglimer.test.synthetic import create_R

class TestIt(unittest.TestCase):
    """
    A class to test the iterative time domain deconvolution.
    """

    def test_decon(self):
        """
        Convolution with subsequent deconvolution.
        """
        # The incoming wavelet
        g = gaussian(51,2.5)


        # Impulse response
        r = np.zeros_like(g)
        r[0] = 1
        r[15] = .25
        

        # convolve the two to a signal
        s = np.convolve(g, r)[:len(g)]
        
        # Deconvolve
        _, _, r2 = it(g, s, 1, omega_min=0.5)

        # test result
        self.assertIsNone(testing.assert_allclose(r, r2[0:len(r)],atol=0.001))
        

if __name__ == "__main__":
    unittest.main()