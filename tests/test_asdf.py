'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 20th October 2021 03:50:35 pm
Last Modified: Friday, 22nd October 2021 05:59:00 pm
'''
import os
import unittest
from unittest.mock import patch, call

from obspy import read, read_events, read_inventory
from obspy.core.utcdatetime import UTCDateTime

from pyglimer.database import asdf
from pyglimer.utils.roundhalf import roundhalf


class TestWriteRaw(unittest.TestCase):
    @patch('pyglimer.database.asdf.os.listdir')
    @patch('pyglimer.database.asdf.read')
    @patch('pyglimer.database.asdf.read_inventory')
    @patch('pyglimer.database.asdf.write_st')
    def test_works(self, wst_mock, ri_mock, read_mock, listdir_mock):
        read_mock.return_value = read()
        ri_mock.return_value = read_inventory()
        listdir_mock.return_value = ['1.item.mseed']

        evt = read_events()[0]
        rawdir = os.path.join('path', 'to', 'waveforms')
        statdir = os.path.join('path', 'to', 'response')

        asdf.writeraw(evt, rawdir, statdir, False, False)

        listdir_mock.assert_called_once_with(rawdir)
        ri_mock.assert_called_once_with(os.path.join(statdir, '1.item.xml'))
        read_mock.assert_called_once_with(os.path.join(rawdir, '1.item.mseed'))
        wst_mock.assert_called_once_with(
            read(), evt, os.path.join(rawdir, os.path.pardir),
            read_inventory(), False)

    @patch('pyglimer.database.asdf.read_inventory')
    @patch('pyglimer.database.asdf.os.listdir')
    def test_errorhandler(self, listdir_mock, ri_mock):
        ri_mock.side_effect = Exception('Test')
        listdir_mock.return_value = ['1.item.mseed']

        evt = read_events()[0]
        rawdir = os.path.join('path', 'to', 'waveforms')
        statdir = os.path.join('path', 'to', 'response')

        with self.assertLogs(
                "obspy.clients.fdsn.mass_downloader", level='ERROR') as cm:
            asdf.writeraw(evt, rawdir, statdir, False, False)
            self.assertEqual(cm.output, [
                'ERROR:obspy.clients.fdsn.mass_downloader:Test'])
        listdir_mock.assert_called_once_with(rawdir)


class TestWriteSt(unittest.TestCase):
    def setUp(self):
        self.st = read()
        self.evt = read_events()
        self.inv = read_inventory()

    @patch('pyglimer.database.asdf.ASDFDataSet', name='ds_mock')
    def test_add_data(self, ds_mock):
        ds_mock.name = 'blas'
        asdf.write_st(self.st, self.evt[0], 'bla', self.inv, False)
        ds_mock.assert_called_once_with(os.path.join('bla', '%s.%s.h5' % (
            self.st[0].stats.network, self.st[0].stats.station)))
        ds_mock().__enter__().add_quakeml.assert_called_once_with(self.evt[0])
        ds_mock().__enter__().add_waveforms.assert_called_once_with(
            self.st, tag='raw_recording', event_id=self.evt[0].resource_id
        )
        ds_mock().__enter__().add_stationxml.assert_called_once_with(
            self.inv
        )

    @patch('pyglimer.database.asdf.ASDFDataSet', name='ds_mock')
    def test_missing_component(self, ds_mock):
        asdf.write_st(self.st[:-1], self.evt[0], 'bla', self.inv, False)
        ds_mock().__enter__().add_quakeml.assert_not_called()
        ds_mock().__enter__().add_waveforms.assert_called_once_with(
            self.st[:-1], tag='raw_recording', event_id=self.evt[0].resource_id
        )
        ds_mock().__enter__().add_stationxml.assert_called_once_with(
            self.inv)


class TestRewriteToHDF5(unittest.TestCase):
    def setUp(self):
        self.evtcat = read_events()
        self.rawdir = os.path.join('path', 'to', 'waveforms')
        self.calls = []
        self.writeraw_calls = []
        for event in self.evtcat:
            origin_time = event.origins[0].time
            ot_loc = UTCDateTime(origin_time, precision=-1).format_fissures()[:-6]
            evtlat = event.origins[0].latitude
            evtlon = event.origins[0].longitude
            evtlat_loc = str(roundhalf(evtlat))
            evtlon_loc = str(roundhalf(evtlon))
            evtdir = os.path.join(
                self.rawdir, '%s_%s_%s' % (ot_loc, evtlat_loc, evtlon_loc))
            self.calls.append(call(evtdir))
            self.writeraw_calls.append(call(
                event, evtdir, os.path.join('path', 'to', 'response'),
                False, True))

    @patch('pyglimer.database.asdf.obspy.core.event.catalog.Catalog.write')
    @patch('pyglimer.database.asdf.os.path.isdir')
    @patch('pyglimer.database.asdf.writeraw')
    @patch('pyglimer.database.asdf.shutil.copyfile')
    @patch('pyglimer.database.asdf.read_events')
    def test_no_waveform(
            self, re_mock, copy_mock, writeraw_mock, isdir_mock, cat_mock):
        re_mock.return_value = self.evtcat
        isdir_mock.return_value = False
        catfile = os.path.join('path', 'to', 'catalog')
        statdir = os.path.join('path', 'to', 'response')

        asdf.rewrite_to_hdf5(catfile, self.rawdir, statdir)
        copy_mock.assert_called_once_with(catfile, '%s_bac' % catfile)
        writeraw_mock.assert_not_called()
        # expected calls
        isdir_mock.assert_has_calls(self.calls)
        isdir_mock.assert_called_with
        assert 3 == isdir_mock.call_count
        cat_mock.assert_called_with(catfile, format='QUAKEML')

    @patch('pyglimer.database.asdf.os.rmdir')
    @patch('pyglimer.database.asdf.os.listdir')
    @patch('pyglimer.database.asdf.obspy.core.event.catalog.Catalog.write')
    @patch('pyglimer.database.asdf.os.path.isdir')
    @patch('pyglimer.database.asdf.writeraw')
    @patch('pyglimer.database.asdf.shutil.copyfile')
    @patch('pyglimer.database.asdf.read_events')
    def test_remove_empty_dir(
        self, re_mock, copy_mock, writeraw_mock, isdir_mock, cat_mock,
            listdir_mock, rmdir_mock):
        re_mock.return_value = self.evtcat
        isdir_mock.return_value = True
        listdir_mock.return_value = []
        catfile = os.path.join('path', 'to', 'catalog')
        statdir = os.path.join('path', 'to', 'response')

        asdf.rewrite_to_hdf5(catfile, self.rawdir, statdir)
        copy_mock.assert_called_once_with(catfile, '%s_bac' % catfile)
        writeraw_mock.assert_not_called()
        # expected calls
        isdir_mock.assert_has_calls(self.calls)
        assert 3 == isdir_mock.call_count
        listdir_mock.assert_has_calls(self.calls)
        assert 3 == listdir_mock.call_count
        rmdir_mock.assert_has_calls(self.calls)
        assert 3 == rmdir_mock.call_count
        cat_mock.assert_called_with(catfile, format='QUAKEML')

    @patch('pyglimer.database.asdf.os.rmdir')
    @patch('pyglimer.database.asdf.os.listdir')
    @patch('pyglimer.database.asdf.obspy.core.event.catalog.Catalog.write')
    @patch('pyglimer.database.asdf.os.path.isdir')
    @patch('pyglimer.database.asdf.writeraw')
    @patch('pyglimer.database.asdf.shutil.copyfile')
    @patch('pyglimer.database.asdf.read_events')
    def test_waveform_found(
        self, re_mock, copy_mock, writeraw_mock, isdir_mock, cat_mock,
            listdir_mock, rmdir_mock):
        re_mock.return_value = self.evtcat
        isdir_mock.return_value = True
        listdir_mock.return_value = [
            'EG.IE.mseed', 'ETC.PP.mseed']
        catfile = os.path.join('path', 'to', 'catalog')
        statdir = os.path.join('path', 'to', 'response')

        asdf.rewrite_to_hdf5(catfile, self.rawdir, statdir)
        copy_mock.assert_called_once_with(catfile, '%s_bac' % catfile)

        # expected calls
        assert 3 == writeraw_mock.call_count
        writeraw_mock.assert_has_calls(self.writeraw_calls)
        isdir_mock.assert_has_calls(self.calls)
        assert 3 == isdir_mock.call_count
        listdir_mock.assert_has_calls(self.calls)
        assert 3 == listdir_mock.call_count
        rmdir_mock.assert_not_called
        cat_mock.assert_called_with(catfile, format='QUAKEML')


if __name__ == "__main__":
    unittest.main()
