'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 20th August 2021 11:29:54 am
Last Modified: Tuesday, 24th August 2021 03:18:45 pm
'''

import unittest

import numpy as np
from obspy.geodetics import locations2degrees

from pyglimer.utils import geo_utils as gu


class TestReckon(unittest.TestCase):
    def testsamepoint(self):
        lat = np.random.randint(-90, 90)
        lon = np.random.randint(-180, 180)
        d = 360*np.random.randint(1, 100)
        b = np.random.random()*360
        la, lo = gu.reckon(lat, lon, d, b)
        self.assertAlmostEqual(la, lat)
        self.assertAlmostEqual(lo, lon)

    def testinverse_dis(self):
        lat = np.random.randint(-90, 90)
        lon = np.random.randint(-180, 180)
        d = np.random.random()*180
        b = np.random.random()*360
        la, lo = gu.reckon(lat, lon, d, b)
        dis = locations2degrees(lat, lon, la, lo)
        self.assertAlmostEqual(dis, d)


class TestGCTrack(unittest.TestCase):
    


if __name__ == "__main__":
    unittest.main()
