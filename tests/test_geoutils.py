'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 20th August 2021 11:29:54 am
Last Modified: Friday, 20th August 2021 11:51:45 am
'''

import unittest

import numpy as np

from pyglimer.utils import geo_utils as gu


class TestReckon(unittest.TestCase):
    def testsamepoint(self):
        lat = np.random.randint(-90, 90)
        lon = np.random.randint(-180, 180)
        d = 360*np.random.randint(1, 100)
        b = np.random.random()*np.pi*2
        la, lo = gu.reckon(lat, lon, d, b)
        self.assertAlmostEqual(la, lat)
        self.assertAlmostEqual(lo, lon)


if __name__ == "__main__":
    unittest.main()
