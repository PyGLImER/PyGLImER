'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3 \
       <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 19th October 2021 03:55:33 pm
Last Modified: Tuesday, 19th October 2021 04:04:47 pm
'''

import unittest

import numpy as np

from pyglimer.utils import statistics as stat


class TestStackCaseResampling(unittest.TestCase):
    def test_list_len(self):
        A = np.random.random((30, 40))
        b = np.random.randint(10, 500)
        bs = stat.stack_case_resampling(A, b)
        self.assertEqual(len(bs), b)
        self.assertTupleEqual(bs[0].shape, A[1].shape)


if __name__ == "__main__":
    unittest.main()
