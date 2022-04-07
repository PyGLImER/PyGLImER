'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 19th August 2021 04:01:26 pm
Last Modified: Wednesday, 16th March 2022 04:20:32 pm
'''
import os
import unittest
from unittest.mock import MagicMock, call, patch, DEFAULT, ANY
from unittest import mock
import warnings

import numpy as np
from obspy import read_inventory, Inventory, read, read_events, UTCDateTime,\
    Trace, Stream
from obspy.core import AttribDict
import obspy
from obspy.clients.fdsn.header import URL_MAPPINGS

from pyglimer.utils import utils as pu
from pyglimer.utils.roundhalf import roundhalf


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
    def test_save_mseed(self, sm_mock):
        with patch.multiple(self.st, select=DEFAULT, slice=DEFAULT) as mocks:
            mocks['select'].return_value = self.st
            mocks['slice'].return_value = self.st
            pu.save_raw(self.saved, self.st, 'rawloc', 'inv', False)
            mocks['select'].assert_has_calls(
                [call(network=ii, station=ii) for ii in range(3)])
            sm_mock.assert_has_calls([
                call(ii, ANY, 'rawloc', ii, ii) for ii in range(3)])

    @patch('pyglimer.utils.utils.write_st')
    def test_save_asdf(self, write_st_mock):
        inv = read_inventory()
        with patch.multiple(self.st, select=DEFAULT, slice=DEFAULT) as mocks:
            with patch.object(inv, 'select') as sel_mock:
                mocks['select'].return_value = self.st
                mocks['slice'].return_value = self.st
                sel_mock.return_value = inv
                pu.save_raw(self.saved, self.st, 'rawloc', inv, True)
                mocks['select'].assert_has_calls(
                    [call(network=ii, station=ii) for ii in range(3)])
                write_st_mock.assert_has_calls([
                    call(ANY, ii, 'rawloc', inv) for ii in range(3)])
                sel_mock.assert_has_calls(
                    [call(ii, ii, starttime=ii, endtime=ii) for ii in range(3)]
                )

    def test_exception(self):
        inv = read_inventory()
        with patch.object(self.st, 'select') as sel_mock:
            sel_mock.side_effect = Exception('Test')
            with self.assertLogs("", level='ERROR') as cm:
                pu.save_raw(self.saved, self.st, 'rawloc', inv, True)
                self.assertEqual(
                    cm.output,
                    ['ERROR:root:Test', 'ERROR:root:Test', 'ERROR:root:Test'])


class TestSaveRawMseed(unittest.TestCase):
    def setUp(self):
        self.evt = read_events()[0]
        self.st = read()

    @patch('pyglimer.utils.utils.os.makedirs')
    def test_func(self, mkdir_mock):
        o = (self.evt.preferred_origin() or self.evt.origins[0])
        ot_loc = UTCDateTime(o.time, precision=-1).format_fissures()[:-6]
        evtlat_loc = str(roundhalf(o.latitude))
        evtlon_loc = str(roundhalf(o.longitude))
        folder = os.path.join(
            'rawloc', '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))
        net = 'bla'
        stat = 'blub'
        fn = os.path.join(folder, '%s.%s.mseed' % (net, stat))
        with patch.object(self.st, 'write') as write_mock:
            pu.save_raw_mseed(self.evt, self.st, 'rawloc', net, stat)
            write_mock.assert_called_once_with(fn, fmt='mseed')
        mkdir_mock.assert_called_once_with(folder, exist_ok=True)


class TestGetMultipleFDSNClients(unittest.TestCase):
    def test_with_None(self):
        exp = sorted(dict(URL_MAPPINGS.items()))
        out = pu.get_multiple_fdsn_clients(None)
        self.assertGreater(len(out), 5)
        for el in out:
            self.assertIn(el, exp)

    def test_create_list(self):
        self.assertListEqual(['bla'], pu.get_multiple_fdsn_clients('bla'))


class TestCreateBulkStr(unittest.TestCase):
    def test_netstatstr(self):
        exp = [('bla', 'bla', '00', 'BHZ', '*', '*')]
        self.assertListEqual(
            exp, pu.create_bulk_str('bla', 'bla', '00', 'BHZ', '*', '*'))

    def test_netstatstr2(self):
        exp = [(
            'bla', 'blub', '00', 'BHZ', UTCDateTime(2016, 8, 1, 0, 0, 1),
            UTCDateTime(2015, 8, 1, 0, 0, 1)), (
            'bla', 'blub', '00', 'BHZ', UTCDateTime(2015, 8, 1, 0, 0, 1),
            UTCDateTime(2016, 8, 1, 0, 0, 1))]
        self.assertListEqual(
            exp, pu.create_bulk_str(
                'bla', 'blub', '00', 'BHZ',
                [UTCDateTime(2016, 8, 1, 0, 0, 1),
                    UTCDateTime(2015, 8, 1, 0, 0, 1)],
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)]))

    def test_all_len_x(self):
        exp = [(
            'bla', 'bla', '00', 'BHZ', UTCDateTime(2015, 8, 1, 0, 0, 1),
            UTCDateTime(2016, 8, 1, 0, 0, 1))]*3
        inp = [[
            'bla']*3, ['bla']*3, '00', 'BHZ',
            ['2015-08-1 00:00:01.0']*3, ['2016-08-1 00:00:01.0']*3]
        self.assertListEqual(
            exp, pu.create_bulk_str(*inp))

    def test_all_len_x2(self):
        exp = [(
            'bla', 'blo', '00', 'BHZ', '*', '*')]*2
        self.assertListEqual(
            exp, pu.create_bulk_str(
                ['bla', 'bla'], ['blo', 'blo'], '00', 'BHZ', '*', '*'))

    def test_net_str_stat_list(self):
        exp = [(
            'bla', 'blub', '00', 'BHZ', '*', '*'), (
            'bla', 'blib', '00', 'BHZ', '*', '*')]
        self.assertListEqual(
            exp, pu.create_bulk_str(
                'bla', ['blub', 'blib'], '00', 'BHZ', '*', '*'))

    def test_net_str_stat_list2(self):
        exp = [(
            'bla', 'blub', '00', 'BHZ', UTCDateTime(2015, 8, 1, 0, 0, 1),
            UTCDateTime(2016, 8, 1, 0, 0, 1)), (
            'bla', 'blib', '00', 'BHZ', UTCDateTime(2015, 8, 1, 0, 0, 1),
            UTCDateTime(2016, 8, 1, 0, 0, 1))]
        self.assertListEqual(
            exp, pu.create_bulk_str(
                'bla', ['blub', 'blib'], '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2015, 8, 1, 0, 0, 1)],
                [UTCDateTime(2016, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)]))

    def test_net_list_stat_str(self):
        exp = [(
            'bla', '*', '00', 'BHZ', '*', '*'), (
            'blub', '*', '00', 'BHZ', '*', '*')]
        self.assertListEqual(
            exp, pu.create_bulk_str(
                ['bla', 'blub'], '*', '00', 'BHZ', '*', '*'))

    def test_net_list_stat_str2(self):
        exp = [(
            'bla', '*', '00', 'BHZ', UTCDateTime(2015, 8, 1, 0, 0, 1),
            UTCDateTime(2016, 8, 1, 0, 0, 1)), (
            'blub', '*', '00', 'BHZ', UTCDateTime(2015, 8, 1, 0, 0, 1),
            UTCDateTime(2016, 8, 1, 0, 0, 1))]
        self.assertListEqual(
            exp, pu.create_bulk_str(
                ['bla', 'blub'], '*', '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2015, 8, 1, 0, 0, 1)],
                [UTCDateTime(2016, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)]))

    def test_net_stat_list_diff_len(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                ['bla', 1], ['blub', 'blib', 0], '00', 'BHZ', '*', '*')

    def test_t_lists_diff_len(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                'a', 'b', '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], '*')

    def test_t_lists_diff_len2(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                'a', ['b', 'c'], '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], '*')

    def test_t_lists_diff_len3(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                ['a', 'b'], ['b', 'c'], '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], '*')

    def test_t_lists_diff_len4(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                ['a', 'b'], '*', '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], '*')

    def test_t_lists_diff_len5(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                ['a', 'b'], '*', '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], [
                        UTCDateTime(2015, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1)])

    def test_t_lists_diff_len6(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                ['a', 'b'], ['b', 'c'], '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], [
                        UTCDateTime(2015, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1)])

    def test_t_lists_diff_len7(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                'a', ['b', 'c'], '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], [
                        UTCDateTime(2015, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1)])

    def test_t_lists_diff_len8(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                'a', 'c', '00', 'BHZ',
                [UTCDateTime(2015, 8, 1, 0, 0, 1),
                    UTCDateTime(2016, 8, 1, 0, 0, 1)], [
                        UTCDateTime(2015, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1),
                        UTCDateTime(2016, 8, 1, 0, 0, 1)])

    def test_other_error(self):
        with self.assertRaises(ValueError):
            pu.create_bulk_str(
                ['a', 'b'], 'c', '00', 'BHZ', '*', '*')


class TestCosTaperSt(unittest.TestCase):
    def setUp(self) -> None:
        self.st = read()

    @mock.patch('pyglimer.utils.utils.cos_taper')
    def test_result(self, cos_taper_mock):
        trcs = [
            Trace(np.zeros((10,))), Trace(np.ones((10,))),
            Trace(2*np.ones((10,)))]
        cos_taper_mock.side_effect = trcs
        exp = Stream(trcs)
        out = pu.cos_taper_st(self.st.copy(), 5, True)
        calls = [
            mock.call(self.st[0], 5, True, 'both'),
            mock.call(self.st[1], 5, True, 'both'),
            mock.call(self.st[2], 5, True, 'both')]
        cos_taper_mock.assert_has_calls(calls)
        for tr, tro in zip(exp, out):
            np.testing.assert_array_equal(tr.data, tro.data)

    @mock.patch('pyglimer.utils.utils.cos_taper')
    def test_trace_cannot_be_tapered(self, cos_taper_mock):
        intr = Trace(np.zeros((10,)))
        cos_taper_mock.side_effect = ValueError
        with warnings.catch_warnings(record=True) as w:
            out = pu.cos_taper_st(intr, 5, True)
            self.assertEqual(len(w), 1)
        calls = [
            mock.call(intr, 5, True, 'both')]
        cos_taper_mock.assert_has_calls(calls)
        self.assertEqual(out, Stream([intr]))


class TestCosTaper(unittest.TestCase):
    def setUp(self):
        self.sr = 10  # sampling rate
        st = AttribDict({'sampling_rate': self.sr})
        self.testtr = Trace(np.ones(1000), header=st)
        tl = np.random.randint(1, high=20)
        self.tls = tl * self.sr  # taper len in samples
        self.tr_res = pu.cos_taper(self.testtr.copy(), tl, False)
        self.tr_resl = pu.cos_taper(self.testtr.copy(), tl, False, side='left')
        self.tr_resr = pu.cos_taper(
            self.testtr.copy(), tl, False, side='right')

    def test_in_place(self):
        self.sr = 10  # sampling rate
        st = AttribDict({'sampling_rate': self.sr})
        testtro = Trace(np.ones(1000), header=st)
        testtr = testtro.copy()
        self.assertEqual(testtr, testtro)
        pu.cos_taper(testtr, 5, False)
        self.assertNotEqual(testtr, testtro)

    def test_ends(self):
        # Check that ends reduce to 0
        self.assertAlmostEqual(self.tr_res.data[0], 0)
        self.assertAlmostEqual(self.tr_resl.data[0], 0)
        self.assertAlmostEqual(self.tr_res.data[-1], 0)
        self.assertAlmostEqual(self.tr_resr.data[-1], 0)
        # one-sided taper
        self.assertAlmostEqual(self.tr_resl.data[-1], 1)
        self.assertAlmostEqual(self.tr_resr.data[0], 1)

    def test_middle(self):
        # Assert that the rest (in the middle) stayed the same
        self.assertTrue(np.array_equal(
            self.testtr[self.tls:-self.tls], self.tr_res[self.tls:-self.tls]))
        self.assertTrue(np.array_equal(
            self.testtr[self.tls:-self.tls], self.tr_resr[self.tls:-self.tls]))
        self.assertTrue(np.array_equal(
            self.testtr[self.tls:-self.tls], self.tr_resl[self.tls:-self.tls]))

    def test_up_down(self):
        # Everything else should be between 1 and 0
        # up
        self.assertTrue(np.all(self.tr_res[1:-1] > 0))
        self.assertTrue(np.all(self.tr_resr[1:-1] > 0))
        self.assertTrue(np.all(self.tr_resl[1:-1] > 0))
        self.assertTrue(np.all(self.tr_res[1:self.tls] < 1))
        self.assertTrue(np.all(self.tr_resl[1:self.tls] < 1))
        # down
        self.assertTrue(np.all(self.tr_res[-self.tls:-1] < 1))
        self.assertTrue(np.all(self.tr_resr[-self.tls:-1] < 1))

    def test_empty_trace(self):
        testtr = Trace(np.array([]), header=self.testtr.stats)
        with self.assertRaises(ValueError):
            pu.cos_taper(testtr, 10, False)

    def test_invalid_taper_len(self):
        with self.assertRaises(ValueError):
            pu.cos_taper(
                self.testtr.copy(), np.random.randint(-100, 0), False)
        with self.assertRaises(ValueError):
            pu.cos_taper(self.testtr.copy(), 501*self.sr, False)

    def test_masked_value(self):
        tr0 = read()[0]
        tr1 = tr0.copy()
        tr1.stats.starttime += 240
        st = Stream([tr0, tr1])
        tr = st.merge()[0]
        tl = np.random.randint(1, high=5)
        ttr = pu.cos_taper(tr, tl, True)
        # Check that ends reduce to 0
        self.assertAlmostEqual(ttr.data[0], 0)
        self.assertAlmostEqual(ttr.data[-1], 0)
        self.assertAlmostEqual(ttr.data[tr0.count()-1], 0)
        self.assertAlmostEqual(ttr.data[-tr1.count()], 0)
        # Also the mask should be retained
        self.assertEqual(
            len(ttr.data[ttr.data.mask]), ttr.count()-tr0.count()-tr1.count())


if __name__ == "__main__":
    unittest.main()
