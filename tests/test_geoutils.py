'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 20th August 2021 11:29:54 am
Last Modified: Thursday, 21st October 2021 03:26:30 pm
'''

import unittest

import numpy as np
from obspy.geodetics import locations2degrees

from pyglimer.utils import geo_utils as gu
from pyglimer.utils.even2Dpoints import even2Dpoints as e2D


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

    def testspacing(self):

        # Least distance between waypoints
        mind = 10

        # Create random new distance rangin from
        # 0.5-1.0 * (a hundreth of the least distance betweem waypoints)
        d = (0.5 * np.random.rand(1) + 0.5)*mind/100
        d = d[0]
        # Create randomly but evenly distirbuted waypoints.
        lon, lat = e2D(10, 340, 160, mind)

        # Compute GCTrack
        qlat, qlon, qdists = gu.gctrack(lat, lon, d)

        # Find locations of waypoints
        pos = []
        for _lat, _lon in zip(lat, lon):
            pos.append(
                np.where(
                    np.isclose(_lat, qlat) & np.isclose(_lon, qlon))[0])
        pos = [[_i-1, _i, _i+1] for _i in pos]
        pos = np.array(pos).flatten().tolist()
        # print("Waypoint idxs", pos)

        for ii, (la, lo, _) in enumerate(
                zip(qlat, qlon, qdists)):
            if ii in pos:
                # naturally unprecise
                continue
            dis = locations2degrees(la, lo, qlat[ii+1], qlon[ii+1])

            np.testing.assert_approx_equal(dis, d, significant=10.0)

    def testspacing_2(self):
        """This tests the accuracy between """

        # Least distance between waypoints
        mind = 10

        # Create random new distance rangin from
        # 0.5-1.0 * (a hundreth of the least distance betweem waypoints)
        d = (0.5 * np.random.rand(1) + 0.5)*mind/100
        d = d[0]
        # Create randomly but evenly distirbuted waypoints.
        lon, lat = e2D(2, 340, 160, mind)

        # Compute GCTrack
        qlat, qlon, qdists, updated_dists = gu.gctrack(
            lat, lon, d, constantdist=False)

        # print("Updated:")
        # print(d, np.max(updated_dists), np.min(updated_dists))
        # print(np.array(updated_dists)-d)

        # Find locations of waypoints
        pos = []
        for _lat, _lon in zip(lat, lon):
            tpos = np.where(np.isclose(_lat, qlat) & np.isclose(_lon, qlon))[0]
            # print(len(tpos))
            if len(tpos) > 1:
                pos.append(tpos)
            else:
                pos.extend(tpos)

        pos = [[_i-1, _i, _i+1] for _i in pos]
        pos = np.array(pos).flatten().tolist()

        for ii, (la, lo, _) in enumerate(
                zip(qlat, qlon, qdists)):
            if ii in pos:
                # naturally unprecise
                continue
            dis = locations2degrees(la, lo, qlat[ii+1], qlon[ii+1])

            np.testing.assert_approx_equal(dis, d, significant=1)


if __name__ == "__main__":
    unittest.main()
