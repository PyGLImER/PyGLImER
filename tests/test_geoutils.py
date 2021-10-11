'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 20th August 2021 11:29:54 am
Last Modified: Wednesday, 25th August 2021 11:19:00 am
'''

import unittest

import numpy as np
from obspy.geodetics import locations2degrees

from pyglimer.utils import geo_utils as gu


class TestReckon(unittest.TestCase):
    def testsamepoint(self):
        d = 360*np.random.randint(1, 100)

        lat = np.random.randint(-90, 90)
        lon = np.random.randint(-180, 180)
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


def testspacing():
    d = 10*(np.random.rand(1))/20

    lat = np.random.rand(10)*180-90
    lon = np.random.rand(10)*360-90

    qlat, qlon, qdists, sdists = gu.gctrack(lat, lon, d)
    for ii, (la, lo, qdi, sdi) in enumerate(
            zip(qlat, qlon, qdists, sdists)):
        if ii in (0, len(qlat)-1):
            # naturally unprecise
            continue
        dis = locations2degrees(la, lo, qlat[ii+1], qlon[ii+1])

        # print('Debug')
        # print(d, dis)
        # print(lat)
        # print(lon)
        np.testing.assert_approx_equal(dis, d, significant=1.0)


if __name__ == "__main__":
    unittest.main()
