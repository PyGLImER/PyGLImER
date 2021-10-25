'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 25th October 2021 04:16:06 pm
Last Modified: Monday, 25th October 2021 04:24:22 pm
'''
import unittest

import numpy as np
from obspy import read

from pyglimer.rf import moveout as mo
from pyglimer.rf.create import RFStream


class SimpleModelTestCase(unittest.TestCase):

    def setUp(self):
        self.model = mo.load_model()

    def test_moveout_vs_XY(self):
        stream = RFStream(read())[:1]
        for tr in stream:
            tr.stats.slowness = 10.
            tr.stats.onset = tr.stats.starttime + 20.643
            tr.stats.phase = 'P'
            tr.stats.type = 'time'
            tr.stats.station_elevation = 0
            tr.stats.station_longitude = 0
            tr.stats.station_latitude = 0
        stream.decimate(10)
        N = len(stream[0])
        t = np.linspace(0, 20*np.pi, N)
        stream[0].data = np.sin(t)*np.exp(-0.04*t)
        stream[0].stats.slowness = 4.0
        stream1 = stream.copy()
        stream3 = stream.copy()
        stream3[0].stats.slowness = 9.0
        stream9 = stream.copy()
        stream10 = stream.copy()

        stream1.moveout('iasp91.dat')
        stream3.moveout('iasp91.dat')

        np.testing.assert_array_almost_equal(
            stream9[0].data, stream10[0].data, decimal=2)


if __name__ == "__main__":
    unittest.main()
