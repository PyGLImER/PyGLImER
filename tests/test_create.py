'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 25th October 2021 11:17:56 am
Last Modified: Monday, 25th October 2021 03:39:58 pm
'''
from copy import deepcopy
import os
import unittest
from unittest.mock import patch

import numpy as np
from numpy import testing as npt
from obspy import read, UTCDateTime

from pyglimer.rf import create
from pyglimer.data.finddir import finddir


class TestRead(unittest.TestCase):
    def test_example_read(self):
        rf = create.read_rf()
        self.assertIsInstance(rf, create.RFStream)
        self.assertIsInstance(rf[0], create.RFTrace)

    @patch('pyglimer.rf.create.RFTrace.ppoint')
    def test_example_read_depth(self, pp_mock):
        create.read_rf(
            os.path.join(finddir(), 'examples', 'PRF_depth.sac'))
        pp_mock.assert_called_once()


class TestRFTrace(unittest.TestCase):
    def setUp(self):
        self.prft = create.read_rf()[0]
        self.prfz = create.read_rf(
            os.path.join(finddir(), 'examples', 'PRF_depth.sac'))[0]

    def test_str(self):
        exp = "Prf time IU.HRV.00.PRF | -30.0s - 120.0s onset:2018-01-14T09:"\
            + "28:34.789987Z | 10.0 Hz, 1501 samples | mag:7.1 dist:58.1 baz:"\
            + "183.6 slow:7.00"
        self.assertEqual(
            self.prft.__str__(), exp)

    def test_init_empty(self):
        with self.assertRaises(ValueError):
            create.RFTrace()

    def test_init_from_trace(self):
        tr = read()[0]
        rftr = create.RFTrace(trace=tr)
        self.assertDictEqual(dict(tr.stats), dict(rftr.stats))

    def test_init_trace_from_array(self):
        rftr = create.RFTrace(data=np.zeros(5))
        self.assertTrue(np.all(rftr.data == np.zeros(5)))

    def test_seconds_utc(self):
        seconds = [UTCDateTime(0), None, 10]
        out = self.prft._seconds2utc(seconds, UTCDateTime(0))
        exp = [UTCDateTime(0), None, UTCDateTime(0) + 10]
        self.assertListEqual(out, exp)
        self.assertEqual(self.prft._seconds2utc(0, None), 0)

    def test_moveout_already_migrated(self):
        with self.assertRaises(TypeError):
            self.prfz.moveout('iasp91.dat')

    def test_multiple_S(self):
        srft = deepcopy(self.prft)
        srft.stats.phase = 'S'
        with self.assertRaises(NotImplementedError):
            srft.moveout('iasp91.dat', multiple=True)

    def test_moveout_values(self):
        _, prfz2, _, _ = self.prft.moveout('iasp91.dat')
        npt.assert_allclose(prfz2.data, self.prfz.data)
        self.assertEqual(prfz2.stats.type, 'depth')


if __name__ == "__main__":
    unittest.main()
