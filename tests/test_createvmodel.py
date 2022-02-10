'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 19th August 2021 04:27:03 pm
Last Modified: Thursday, 10th February 2022 04:47:15 pm
'''

import unittest
from unittest import mock
from copy import deepcopy
import os

import numpy as np

from pyglimer.utils import createvmodel as cvm


class TestComplexModel(unittest.TestCase):
    def setUp(self):
        self.model = cvm.load_gyps(flatten=False)
        self.subm = self.model.submodel((10, 15), (10, 15))

    def testformat(self):
        self.assertIsInstance(self.model, cvm.ComplexModel)

    def testquery(self):
        # Just test some coordinates here that we know the values for
        exp = (5.836807960000001, 3.2415152000000003)
        vp, vs = self.model.query(-90, -179, 0)
        self.assertAlmostEqual(vp, exp[0])
        self.assertAlmostEqual(vs, exp[1])

    def testquery2(self):
        # Just test some coordinates here that we know the values for
        exp = (5.80521304, 3.6632557049999996)
        vp, vs = self.model.query(0.4, 0.4, 4)
        self.assertAlmostEqual(vp, exp[0])
        self.assertAlmostEqual(vs, exp[1])

    def testsubmodelcreate(self):
        self.assertIsInstance(self.subm, cvm.ComplexModel)

    def testsubmodelquery(self):
        lat = np.random.randint(10, 15) + np.random.randint(-49, 49)/100
        lon = np.random.randint(10, 15)
        z = np.random.randint(0, 750)
        self.assertTupleEqual(
            self.subm.query(lat, lon, z), self.model.query(lat, lon, z))

    def testnocoverage(self):
        # is set to raise this if the distance to the models closest boundary
        # is greater than 1.25 deg
        with self.assertRaises(self.subm.CoverageError):
            self.subm.query(16.3, 15, 1)
        with self.assertRaises(self.subm.CoverageError):
            self.subm.query(90, -80, 1)

    def testflatten(self):
        submf = deepcopy(self.subm)
        submf.vpf, submf.vsf, submf.zf = self.subm.flatten(
            self.subm.vpf, self.subm.vsf)
        lat = np.random.randint(10, 15) + np.random.randint(-49, 49)/100
        lon = np.random.randint(10, 15)
        z = np.random.randint(0, 750)
        vp, vs = self.subm.query(lat, lon, z)
        vpf, vsf = submf.query(lat, lon, z)
        self.assertAlmostEqual(vp*6371/(6371-z), vpf)
        self.assertAlmostEqual(vs*6371/(6371-z), vsf)

    @mock.patch('pyglimer.utils.createvmodel.pickle.dump')
    def test_write(self, pickle_mock):
        # open_mock.return_value = 'blablafile'
        with mock.patch("builtins.open", mock.mock_open()) as mf:
            self.model.write('thisfile')
            pickle_mock.assert_called_once_with(self.model, mf(), mock.ANY)


class TestAVVModel(unittest.TestCase):
    def setUp(self):
        # The compilation is tested manually elsewhere as it can only be tested
        # if Litho1.0 is installed
        self.model = cvm.load_avvmodel()

    def testunknownphase(self):
        with self.assertRaises(ValueError):
            self.model.query(0, 0, 'V')

    def testsgreaterp(self):
        lat = np.random.randint(-90, 90) + np.random.randint(-1, 0)/100
        lon = np.random.randint(-180, 180) + np.random.randint(-1, 0)/100
        vpp, vsp = self.model.query(lat, lon, 'P')
        vps, vss = self.model.query(lat, lon, 'S')
        self.assertGreater(vps, vpp)
        self.assertGreater(vss, vsp)

    @mock.patch('pyglimer.utils.createvmodel.os.makedirs')
    @mock.patch('pyglimer.utils.createvmodel.np.savez')
    def test_write(self, savez_mock, mkdir_mock):
        self.model.write('this_file')
        self.assertIn(
            os.path.join(
                'PyGLImER', 'src', 'pyglimer', 'data', 'velocity_models',
                'this_file'), mkdir_mock.call_args_list[-1][0][0])
        mkdir_mock.assert_called_once_with(mock.ANY, exist_ok=True)


if __name__ == "__main__":
    unittest.main()
