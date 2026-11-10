'''
Regression tests for the Phase 2 bug fixes of the 2026 overhaul.
Each test pins one fixed defect.

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
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import h5py
from obspy import read_inventory

from pyglimer import tmp
from pyglimer.ccp.io import load_velocity_model
from pyglimer.utils.utils import join_inv
from pyglimer.waveform import errorhandler


class TestLoadVelocityModel(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_dataset_raises(self):
        """
        Used to print the error and then raise a NameError on undefined
        return values, hiding the real cause. Now the h5py error
        propagates.
        """
        fn = os.path.join(self.tmpdir, 'vmodel.h5')
        with h5py.File(fn, 'w') as f:
            f.create_dataset('lat', data=np.zeros(3))
            # 'lon', 'dep', 'vp', 'vs' deliberately missing
        with self.assertRaises(KeyError):
            load_velocity_model(fn)

    def test_valid_file(self):
        fn = os.path.join(self.tmpdir, 'vmodel.h5')
        with h5py.File(fn, 'w') as f:
            f.create_dataset('lat', data=np.zeros((2, 2)))
            f.create_dataset('lon', data=np.ones((2, 2)))
            f.create_dataset('dep', data=np.arange(4).reshape(2, 2))
            f.create_dataset('vp', data=np.full((2, 2), 6.))
            f.create_dataset('vs', data=np.full((2, 2), 3.5))
        vlat, vlon, vdep, vp, vs = load_velocity_model(fn)
        np.testing.assert_array_equal(vlat, np.zeros(4))
        np.testing.assert_array_equal(vlon, np.ones(4))
        np.testing.assert_array_equal(vp, np.full((2, 2), 6.))
        np.testing.assert_array_equal(vs, np.full((2, 2), 3.5))


class TestJoinInv(unittest.TestCase):
    def _inv(self):
        return read_inventory()

    def test_no_default_argument(self):
        # The old signature used typing.List[Inventory] as the default
        # value (a type object, not an inventory).
        with self.assertRaises(TypeError):
            join_inv()

    def test_merge(self):
        inv1 = read_inventory()
        inv2 = read_inventory()
        # Networks with distinct codes to avoid obspy's extend() merging
        # same-code networks
        from obspy.core.inventory import Inventory, Network
        inv1 = Inventory(networks=[Network('AA')], source='test')
        inv2 = Inventory(networks=[Network('BB')], source='test')
        out = join_inv([inv1, inv2])
        self.assertEqual(
            {net.code for net in out}, {'AA', 'BB'})

    def test_empty_list_warns_and_returns_none(self):
        with self.assertWarns(UserWarning):
            out = join_inv([])
        self.assertIsNone(out)


class TestTmpSetClients(unittest.TestCase):
    def setUp(self):
        self._old = tmp.re_client

    def tearDown(self):
        tmp.re_client = self._old

    def test_str(self):
        tmp.set_clients('IRIS')
        self.assertEqual(tmp.re_client, ['IRIS'])

    def test_list_copied(self):
        clients = ['IRIS', 'ORFEUS']
        tmp.set_clients(clients)
        clients.append('RESIF')
        self.assertEqual(tmp.re_client, ['IRIS', 'ORFEUS'])

    def test_none_is_noop(self):
        tmp.set_clients('IRIS')
        tmp.set_clients(None)
        self.assertEqual(tmp.re_client, ['IRIS'])


class TestErrorHandlerRetry(unittest.TestCase):
    """
    tmp.re_client used to stay empty forever, silently disabling all
    error handlers. These tests verify that the handlers iterate over the
    registered clients once it is populated.
    """
    def setUp(self):
        self._old = tmp.re_client

    def tearDown(self):
        tmp.re_client = self._old

    @mock.patch('pyglimer.waveform.errorhandler.Client')
    def test_redownload_uses_registered_clients(self, client_mock):
        tmp.set_clients(['A', 'B'])
        st = mock.MagicMock()
        st.__len__.return_value = 2  # < 3 channels
        st.__getitem__().stats.channel = 'BHZ'
        # First client returns another incomplete stream, second returns
        # a 3-channel stream
        first = mock.MagicMock()
        first.__len__.return_value = 2
        second = mock.MagicMock()
        second.__len__.return_value = 3
        bad, good = mock.MagicMock(), mock.MagicMock()
        bad.get_waveforms.return_value = first
        good.get_waveforms.return_value = second
        client_mock.side_effect = [bad, good]
        out = errorhandler.redownload('IU', 'HRV', 0, 10, st)
        self.assertEqual(client_mock.call_count, 2)
        self.assertEqual(len(out), 3)

    @mock.patch('pyglimer.waveform.errorhandler.Client')
    def test_redownload_statxml(self, client_mock):
        tmp.set_clients(['A'])
        client = client_mock.return_value
        out = errorhandler.redownload_statxml(None, 'IU', 'HRV', 'f.xml')
        client.get_stations.assert_called_once()
        self.assertEqual(out, client.get_stations.return_value)


class TestInitCCPStationFilter(unittest.TestCase):
    """
    init_ccp built its station fnmatch pattern as ``'*' or station``, so a
    string station was silently ignored (the pattern always matched every
    station of the network). These tests pin the fix (``station or '*'``).
    """
    def setUp(self):
        # rfloc two levels below the tmpdir so that init_ccp's logdir
        # (dirname(dirname(rfloc))) stays inside the tmpdir
        self.tmpdir = tempfile.mkdtemp()
        self.rfloc = os.path.join(self.tmpdir, 'waveforms', 'rf')
        os.makedirs(os.path.join(self.tmpdir, 'logs'))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _db():
        import pandas as pd
        return pd.DataFrame({
            'code': ['YP.NE1', 'YP.NE2', 'YP.NE3'],
            'network': ['YP', 'YP', 'YP'],
            'station': ['NE1', 'NE2', 'NE3'],
            'lat': [42.6, 42.7, 42.8],
            'lon': [-71.6, -71.5, -71.4],
            'elevation': [0., 0., 0.]})

    def _init(self, **kwargs):
        from pyglimer.ccp.ccp import init_ccp
        db = mock.Mock(db=self._db())
        with mock.patch('pyglimer.ccp.ccp.StationDB', return_value=db), \
                mock.patch('pyglimer.ccp.ccp.CCPStack') as ccpstack:
            init_ccp(self.rfloc, 2, 'iasp91', 'P', verbose=False, **kwargs)
        # (lats, lons, spacing) positional args of the CCPStack call
        args = ccpstack.call_args[0]
        return np.asarray(args[0]), np.asarray(args[1])

    def test_station_str_filters(self):
        lats, lons = self._init(network='YP', station='NE2')
        np.testing.assert_allclose(lats, [42.7])
        np.testing.assert_allclose(lons, [-71.5])

    def test_station_none_matches_all(self):
        lats, _ = self._init(network='YP', station=None)
        np.testing.assert_allclose(lats, [42.6, 42.7, 42.8])

    def test_no_filter_matches_all(self):
        lats, _ = self._init(network=None, station=None)
        np.testing.assert_allclose(lats, [42.6, 42.7, 42.8])

    def test_station_list_filters(self):
        # Note: init_ccp's list branches intersect Python sets, so the
        # station order is nondeterministic (pre-existing behaviour) --
        # compare sorted.
        lats, _ = self._init(network='YP', station=['NE1', 'NE3'])
        np.testing.assert_allclose(sorted(lats), [42.6, 42.8])


class TestAverageVelModelWrite(unittest.TestCase):
    """
    AverageVelModel.write contained ``filename + '.' - x[-1]`` (str minus
    str -> TypeError), so saving a model with a dotted filename always
    crashed after the ~1 h compilation. The fix strips the extension like
    the sibling ComplexModel.write does.
    """
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model = mock.Mock(spec=['latv', 'lonv', 'avpP', 'avsP',
                                     'avpS', 'avsS'])

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, filename):
        from pyglimer.utils.createvmodel import AverageVelModel
        with mock.patch('pyglimer.utils.createvmodel.finddir',
                        return_value=self.tmpdir), \
                mock.patch('pyglimer.utils.createvmodel.np.savez') as savez:
            AverageVelModel.write(self.model, filename)
        return savez.call_args[0][0]

    def test_dotted_filename(self):
        oloc = self._write('mymodel.npz')
        self.assertEqual(
            oloc, os.path.join(self.tmpdir, 'velocity_models', 'mymodel'))

    def test_plain_filename(self):
        oloc = self._write('mymodel')
        self.assertEqual(
            oloc, os.path.join(self.tmpdir, 'velocity_models', 'mymodel'))

    def test_multiple_dots(self):
        oloc = self._write('my.model.v2.npz')
        self.assertEqual(
            oloc, os.path.join(self.tmpdir, 'velocity_models', 'my.model.v2'))


if __name__ == "__main__":
    unittest.main()
