'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 20th October 2021 03:50:35 pm
Last Modified: Thursday, 8th September 2022 04:16:10 pm
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
        ds_mock().__enter__().add_waveforms.assert_called_once_with(
            self.st, tag='raw_recording')
        ds_mock().__enter__().add_stationxml.assert_called_once_with(
            self.inv
        )

    @patch('pyglimer.database.asdf.ASDFDataSet', name='ds_mock')
    def test_missing_component(self, ds_mock):
        asdf.write_st(self.st[:-1], self.evt[0], 'bla', self.inv, False)
        ds_mock().__enter__().add_quakeml.assert_not_called()
        ds_mock().__enter__().add_waveforms.assert_called_once_with(
            self.st[:-1], tag='raw_recording')
        ds_mock().__enter__().add_stationxml.assert_called_once_with(
            self.inv)


if __name__ == "__main__":
    unittest.main()
