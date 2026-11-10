'''
Characterization tests for :mod:`pyglimer.database.stations` (the
`StationDB` pandas database built from the shelve info files of the
preprocessed receiver functions).

Runs entirely offline on a synthetic directory tree in a temporary
folder.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import os
import shelve
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

from pyglimer.database.stations import StationDB


def _write_info(folder: str, network: str, station: str, num: int,
                numret: int = None, lat=42.6, lon=-71.6, el=200.):
    """Create a shelve info file as written by the RF creation."""
    os.makedirs(folder, exist_ok=True)
    infof = os.path.join(folder, 'info')
    with shelve.open(infof) as db:
        db['network'] = network
        db['station'] = station
        db['statlat'] = lat
        db['statlon'] = lon
        db['statel'] = el
        db['num'] = num
        if numret is not None:
            db['numret'] = numret


class StationDBTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.datadir = os.path.join(self.tmpdir, 'database')
        # Two P stations, one of them without a saved numret
        _write_info(
            os.path.join(self.datadir, 'P', 'by_station', 'AA.BB1'),
            'AA', 'BB1', num=100, numret=80)
        _write_info(
            os.path.join(self.datadir, 'P', 'by_station', 'XX.YY2'),
            'XX', 'YY2', num=50)
        # One S station that shares the network.code with a P station
        _write_info(
            os.path.join(self.datadir, 'S', 'by_station', 'AA.BB1'),
            'AA', 'BB1', num=40, numret=30)
        # One S-only station
        _write_info(
            os.path.join(self.datadir, 'S', 'by_station', 'SS.TT3'),
            'SS', 'TT3', num=10, numret=5, lat=10., lon=20., el=100.)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make(self, **kwargs):
        defaults = dict(
            dir=self.datadir, hdf5=False, logdir=self.tmpdir)
        defaults.update(kwargs)
        return StationDB(**defaults)


class TestCreateFromInfo(StationDBTest):
    def test_single_phase(self):
        db = self._make(phase='P')
        self.assertIsInstance(db.db, pd.DataFrame)
        self.assertEqual(list(db.db['code']), ['AA.BB1', 'XX.YY2'])
        self.assertEqual(list(db.db['NP']), [100, 50])
        # missing numret is counted as 0
        self.assertEqual(list(db.db['NPret']), [80, 0])
        # opposing phase columns exist but are empty for phase-specific run
        self.assertEqual(list(db.db['NS']), [0, 0])
        self.assertEqual(list(db.db['NSret']), [0, 0])
        self.assertEqual(list(db.db['lat']), [42.6, 42.6])
        self.assertEqual(list(db.db['lon']), [-71.6, -71.6])

    def test_invalid_phase(self):
        with self.assertRaises(ValueError):
            self._make(phase='Q')

    def test_merge_both_phases(self):
        # phase=None would save database.csv into the package's data
        # folder -- redirect finddir to the temporary directory
        with mock.patch(
                'pyglimer.database.stations.finddir',
                return_value=self.tmpdir):
            db = self._make()
        row = db.db[db.db['code'] == 'AA.BB1'].iloc[0]
        self.assertEqual(row['NP'], 100)
        self.assertEqual(row['NPret'], 80)
        self.assertEqual(row['NS'], 40)
        self.assertEqual(row['NSret'], 30)
        # S-only station added with NP=0
        row = db.db[db.db['code'] == 'SS.TT3'].iloc[0]
        self.assertEqual(row['NP'], 0)
        self.assertEqual(row['NPret'], 0)
        self.assertEqual(row['NS'], 10)
        self.assertEqual(row['NSret'], 5)
        # P-only station keeps NS=0
        row = db.db[db.db['code'] == 'XX.YY2'].iloc[0]
        self.assertEqual(row['NS'], 0)
        # database.csv was written to the (patched) data folder
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir,
                                                    'database.csv')))


class TestGeoBoundary(StationDBTest):
    def _db(self):
        return self._make(phase='P')

    def test_filter(self):
        db = self._db()
        subset = db.geo_boundary((40, 45), (-75, -70))
        self.assertEqual(len(subset), 2)
        subset = db.geo_boundary((50, 55), (-75, -70))
        self.assertEqual(len(subset), 0)

    def test_phase_requirement(self):
        db = self._db()
        # stations with retained P RFs only
        subset = db.geo_boundary((40, 45), (-75, -70), phase='P')
        self.assertEqual(len(subset), 1)
        self.assertEqual(subset.iloc[0]['code'], 'AA.BB1')

    def test_find_stations(self):
        db = self._db()
        nets, stats = db.find_stations((40, 45), (-75, -70), phase='P')
        self.assertEqual(nets, ['AA'])
        self.assertEqual(stats, ['BB1'])


if __name__ == "__main__":
    unittest.main()
