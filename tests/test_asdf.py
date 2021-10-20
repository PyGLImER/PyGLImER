'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3 \
       <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 20th October 2021 03:50:35 pm
Last Modified: Wednesday, 20th October 2021 04:52:42 pm
'''
import os
import unittest
from unittest.mock import patch

from obspy import read, read_events, read_inventory

from pyglimer.database import asdf


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


if __name__ == "__main__":
    unittest.main()
