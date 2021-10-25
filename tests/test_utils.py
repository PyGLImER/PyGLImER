'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 19th August 2021 04:01:26 pm
Last Modified: Monday, 25th October 2021 05:40:46 pm
'''
import os
import unittest
from unittest.mock import MagicMock, call, patch, DEFAULT, ANY

import numpy as np
from obspy import read_inventory, Inventory, read
import obspy

from pyglimer.utils import utils as pu


class TestDTString(unittest.TestCase):
    def test_h(self):
        dt = np.random.randint(6001, 9999999)
        h = dt/3600
        expstr = "   Time elapsed: %3.1f h" % h
        self.assertEqual(pu.dt_string(dt), expstr)

    def test_min(self):
        dt = np.random.randint(501, 6000)
        min = dt/60
        expstr = "   Time elapsed: %3.1f min" % min
        self.assertEqual(pu.dt_string(dt), expstr)

    def test_s(self):
        dt = np.random.randint(0, 499)
        expstr = "   Time elapsed: %3.1f s" % dt
        self.assertEqual(pu.dt_string(dt), expstr)


class TestChunks(unittest.TestCase):
    def test_result(self):
        testlist = [1, 2, 3, 4]
        self.assertListEqual(list(pu.chunks(testlist, 2)), [[1, 2], [3, 4]])

    def test_result2(self):
        testlist = [1, 2, 3]
        self.assertListEqual(list(pu.chunks(testlist, 2)), [[1, 2], [3]])

    def test_result3(self):
        testlist = [1, 2, 3, 4]
        self.assertListEqual(list(pu.chunks(testlist, 10)), [testlist])


class TestJoinInv(unittest.TestCase):
    def test_result(self):
        exp_inv = read_inventory()
        inv0 = Inventory([exp_inv[0]])
        inv1 = Inventory([exp_inv[1]])
        invl = [inv0, inv1]
        self.assertEqual(exp_inv, pu.join_inv(invl))
        # Just testing the test
        self.assertNotEqual(inv0, inv1)


class TestDownloadFullInventory(unittest.TestCase):
    @patch('pyglimer.utils.utils.os.listdir')
    @patch('pyglimer.utils.utils.__client__loop__')
    def test_bulk(self, cl_mock, oslist_mock):
        oslist_mock.return_value = [
            'not_a_statxml.txt',
            'net.stat.xml',
            'net2.stat2.xml',
            'also.not.jpg']
        exp = [
            ('net', 'stat', '*', '*', '*', '*'),
            ('net2', 'stat2', '*', '*', '*', '*')
        ]
        pu.download_full_inventory('bla', 'IRIS')
        cl_mock.assert_called_with('IRIS', 'bla', exp)


class TestClientLoop(unittest.TestCase):
    def test_no_valid_fdsn(self):
        self.assertIsNone(pu.__client__loop__('bla', 'some', ['a', 'b']))

    def test_orga(self):
        c = MagicMock(spec=obspy.clients.fdsn.Client)
        bulkl = ['my', 'nonesense']
        statloc = 'should not exist'
        my_inv = read_inventory()
        c.get_stations_bulk.return_value = my_inv
        calls = []
        for net in my_inv:
            for stat in net:
                out = os.path.join(
                    statloc, '%s.%s.xml' % (net.code, stat.code))
                calls.append(call(out, format='STATIONXML'))
        with patch.object(my_inv, 'write') as write_mock:
            inv = pu.__client__loop__(c, statloc, bulkl)
            self.assertEqual(inv, my_inv)
            write_mock.assert_has_calls(calls)


class TestClientLoopWav(unittest.TestCase):
    def test_no_valid_fdsn(self):
        self.assertIsNone(pu.__client__loop_wav__(
            'bla', 'some', ['a', 'b'], 0, 0, 0))

    @patch('pyglimer.utils.utils.save_raw')
    def test_orga(self, save_raw_mock):
        c = MagicMock(spec=obspy.clients.fdsn.Client)
        bulkl = ['my', 'nonesense']
        rawloc = 'should not exist'
        c.get_waveforms_bulk.return_value = 'this could be a stream'
        pu.__client__loop_wav__(c, rawloc, bulkl, {}, False, 'inventory')
        save_raw_mock.assert_called_once_with(
            {}, 'this could be a stream', rawloc, 'inventory', False)


class TestSaveRaw(unittest.TestCase):
    def setUp(self) -> None:
        self.st = read()
        ls = [0, 1, 2]
        self.saved = {
            'event': ls, 'startt': ls, 'endt': ls, 'net': ls, 'stat': ls}

    @patch('pyglimer.utils.utils.save_raw_mseed')
    def test_sav_mseed(self, sm_mock):
        patch.multiple

        with patch.multiple(self.st, select=DEFAULT, slice=DEFAULT) as mocks:
            mocks['select'].return_value = self.st
            mocks['slice'].return_value = self.st
            pu.save_raw(self.saved, self.st, 'rawloc', 'inv', False)
            mocks['select'].assert_has_calls(
                [call(network=ii, station=ii) for ii in range(3)])
            sm_mock.assert_has_calls([
                call(ii, ANY, 'rawloc', ii, ii) for ii in range(3)])


if __name__ == "__main__":
    unittest.main()
