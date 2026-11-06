'''
End-to-end pipeline tests on synthetic data.

Covers the full chain raw hdf5 waveforms -> preprocessing (rotation,
quality control, deconvolution) -> receiver function database -> CCP
stack, plus the RFStream case-resampling and quick-stack paths. The
waveforms are synthetic teleseismic P arrivals computed with iasp91, so
the whole test runs offline.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.core.event import Catalog, Event, Magnitude, Origin
from obspy.core.inventory import Channel, Inventory, Network, Station
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

from pyglimer.ccp.ccp import CCPStack
from pyglimer.database.raw import RawDatabase
from pyglimer.database.rfh5 import RFDataBase
from pyglimer.rf import create
from pyglimer.utils import utils as pu
from pyglimer.utils.geo_utils import reckon
from pyglimer.waveform.preprocessh5 import _preprocessh5_single

SLAT, SLON = 42.6, -71.6
DT = 0.1
DUR = 800.0  # length of the synthetic waveforms [s after origin]


def _events():
    """Two events at teleseismic distances around the station."""
    events = []
    for i, (dist, baz, depth) in enumerate(
            [(60., 10., 100.), (65., 130., 150.)]):
        lat, lon = reckon(SLAT, SLON, dist, baz)
        ot = UTCDateTime(2018, 1, 14, 9, 30) + i * 86400
        ev = Event()
        o = Origin(time=ot, latitude=lat, longitude=lon, depth=depth * 1000)
        ev.origins = [o]
        ev.preferred_origin_id = o.resource_id
        ev.magnitudes = [Magnitude(magnitude=6.5, magnitude_type='mww')]
        events.append(ev)
    return Catalog(events=events)


def _inventory():
    chan = []
    for c, az, dip in (('HHZ', 0, -90), ('HHN', 0, 0), ('HHE', 90, 0)):
        chan.append(Channel(
            c, '00', SLAT, SLON, 200, depth=0, dip=dip, azimuth=az,
            sample_rate=1 / DT))
    stat = Station('SYNT', SLAT, SLON, 200)
    stat.channels = chan
    return Inventory(networks=[Network('XX', stations=[stat])])


def _synthetic_stream(model, origin, dist_deg, baz):
    """3-component stream with a P onset at the theoretical arrival and
    a Ps conversion on the radial component 5 s later."""
    arrival = model.get_travel_times(
        source_depth_in_km=origin.depth / 1000,
        distance_in_degree=dist_deg, phase_list=['P'])[0]
    t_arr = arrival.time

    n = int(DUR / DT)
    rng = np.random.RandomState(int(origin.time.timestamp) % 2**31)
    i0 = int(t_arr / DT)
    w = np.exp(-0.5 * (np.arange(-40, 41) / 8.0) ** 2)
    signal_z = np.zeros(n)
    signal_r = np.zeros(n)
    signal_z[i0 - 40:i0 + 41] += 10.0 * w
    signal_r[int((t_arr + 5) / DT) - 40:int((t_arr + 5) / DT) + 41] \
        += 3.0 * w

    st = Stream()
    for cha, sig in (
            ('HHZ', signal_z),
            ('HHN', signal_r * np.cos(baz * np.pi / 180)),
            ('HHE', signal_r * np.sin(baz * np.pi / 180))):
        tr = Trace(data=(sig + rng.normal(0, 0.02, n)).astype(np.float64))
        tr.stats.sampling_rate = 1 / DT
        tr.stats.channel = cha
        tr.stats.station = 'SYNT'
        tr.stats.network = 'XX'
        tr.stats.starttime = UTCDateTime(origin.time)
        st += tr
    return st


def _build_raw_database(rawdir):
    """Create a synthetic raw hdf5 database with 2 events."""
    os.makedirs(rawdir, exist_ok=True)
    model = TauPyModel('iasp91')
    cat = _events()
    inv = _inventory()
    content = {}
    rawf = os.path.join(rawdir, 'XX.SYNT.h5')
    with RawDatabase(rawf, mode='w') as rdb:
        for ev in cat:
            o = ev.origins[0]
            dist_m, baz, _ = gps2dist_azimuth(
                SLAT, SLON, o.latitude, o.longitude)
            st = _synthetic_stream(
                model, o, dist_m / 1000 / 111.19, baz)
            rdb.add_waveform(st, evt_id=o.time)
            for cha in ('HHZ', 'HHN', 'HHE'):
                content.setdefault(cha, [])
                content[cha].append(pu.utc_save_str(o.time))
        rdb.add_response(inv)
        rdb._define_content(content)
    return rawf, cat, model


class TestSyntheticPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.rawdir = os.path.join(cls.tmpdir, 'raw')
        cls.rfdir = os.path.join(cls.tmpdir, 'rf')
        # compute_stack looks for RF databases under <rfloc>/<phase>/,
        # mirroring the production folder layout
        rfloc_phase = os.path.join(cls.rfdir, 'P')
        os.makedirs(rfloc_phase, exist_ok=True)
        rawf, cat, model = _build_raw_database(cls.rawdir)

        # quiet loggers
        logging.basicConfig(level=logging.CRITICAL)
        logger = logging.getLogger('test_e2e.pyglimer')
        rflogger = logging.getLogger('test_e2e.rf')
        for lg in (logger, rflogger):
            lg.setLevel(logging.CRITICAL)
            lg.addHandler(logging.NullHandler())

        _preprocessh5_single(
            'P', 'RT', 'v', 0.05, model, 'cosine', 30, 150, rfloc_phase,
            'it', False, logger, rflogger, rawf, cat, False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_rfs_created(self):
        """Both synthetic events survive QC and are written as RFs."""
        with RFDataBase(
                os.path.join(self.rfdir, 'P', 'XX.SYNT.h5'),
                mode='r') as rfdb:
            rfs = list(rfdb.walk('rf', 'XX', 'SYNT', 'P'))
        self.assertEqual(len(rfs), 2)
        for tr in rfs:
            self.assertGreater(tr.stats.distance, 30)
            self.assertLess(tr.stats.distance, 90)
            self.assertGreater(np.abs(tr.data).max(), 0)

    def test_compute_stack_end_to_end(self):
        """CCPStack.compute_stack ingests the synthetic hdf5 database."""
        st = CCPStack(
            [42.4, 42.8], [-71.8, -71.4], 2, 'P', verbose=False,
            logdir=self.tmpdir)
        st.compute_stack(
            'iasp91.dat', self.rfdir, in_format='hdf5', network='XX',
            station='SYNT')
        # both RFs were binned and illuminated the stack
        self.assertGreater(st.illum.sum(), 0)
        # N counts the RFs that reached the surface depth index
        self.assertEqual(st.N, 2)
        self.assertTrue(np.isfinite(st.ccp).all())
        # conclude_ccp ran inside compute_stack
        self.assertTrue(hasattr(st, 'ccp'))


class TestRFStreamResampling(unittest.TestCase):
    def setUp(self):
        self.rfs = create.read_rf()

    def test_bootstrap(self):
        out = self.rfs.bootstrap(b=5)
        self.assertEqual(len(out), 5)
        for arr in out:
            self.assertEqual(arr.shape, out[0].shape)
            self.assertTrue(np.isfinite(arr).all())
        # resampled stacks scatter around the deterministic stack
        self.assertGreater(np.std(out), 0)

    def test_dirty_ccp_stack(self):
        latc, lonc, zc, ccpi, illum = self.rfs.dirty_ccp_stack(
            dlon=2.0, maxz=200)
        self.assertGreater(illum.sum(), 0)
        # the mesh is ordered (lon, lat, depth)
        self.assertEqual(ccpi.shape, (len(lonc), len(latc), len(zc)))
        self.assertEqual(ccpi.shape, illum.shape)
        self.assertTrue(np.isfinite(ccpi).all())


if __name__ == "__main__":
    unittest.main()
