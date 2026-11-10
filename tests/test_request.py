'''
Characterization tests for :class:`pyglimer.waveform.request.Request`.

Pins the object's initialisation behaviour (directory layout, attribute
normalisation, format handling, logging) entirely offline in temporary
directories. No downloads are attempted.

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
import logging
import shutil
import tempfile
import unittest
from unittest import mock

from obspy import Catalog, UTCDateTime
from obspy.core.event import Event
from obspy.taup import TauPyModel

from pyglimer.waveform.request import Request


class RequestTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # An existing event catalogue prevents Request.__init__ from
        # downloading one from IRIS (tests must run offline)
        evtfile = os.path.join(self.tmpdir, 'events', 'catalog.xml')
        os.makedirs(os.path.dirname(evtfile))
        Catalog([Event()]).write(evtfile, format='QUAKEML')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Keep the module logger clean for the next test
        log = logging.getLogger('pyglimer.request')
        while log.handlers:
            log.removeHandler(log.handlers[0])

    def _make(self, **kwargs):
        defaults = dict(
            proj_dir=self.tmpdir, raw_subdir='waveforms', rf_subdir='RF',
            statloc_subdir='stations', evt_subdir='events',
            log_subdir='logs', phase='P', rot='RTZ', deconmeth='time',
            starttime=UTCDateTime(0), endtime=UTCDateTime(1),
            evtcat='catalog.xml')
        defaults.update(kwargs)
        return Request(**defaults)


class TestRequestInit(RequestTest):
    def test_directory_layout(self):
        req = self._make(prepro_subdir='preprocessed')
        self.assertEqual(
            req.logdir, os.path.join(self.tmpdir, 'logs'))
        self.assertEqual(
            req.rawloc, os.path.join(self.tmpdir, 'waveforms', 'P'))
        self.assertEqual(
            req.preproloc,
            os.path.join(self.tmpdir, 'preprocessed', 'P'))
        self.assertEqual(req.rfloc, os.path.join(self.tmpdir, 'RF', 'P'))
        self.assertTrue(
            os.path.isfile(os.path.join(req.logdir, 'request.log')))

    def test_attributes(self):
        req = self._make(prepro_subdir='preprocessed')
        self.assertIsInstance(req.model, TauPyModel)
        self.assertTrue(req.h5)
        # last character of the phase is upper-cased (a single-char
        # phase is hence fully upper-cased)
        req2 = self._make(phase='s', prepro_subdir='preprocessed')
        self.assertEqual(req2.phase, 'S')
        req3 = self._make(rot='rtz', prepro_subdir='preprocessed')
        self.assertEqual(req3.rot, 'RTZ')
        # phases not in constants are rejected
        with self.assertRaises(NotImplementedError):
            self._make(phase='pP', prepro_subdir='preprocessed')

    def test_no_preproloc_for_hdf5(self):
        req = self._make()
        self.assertIsNone(req.preproloc)

    def test_format_h5(self):
        # 'h5' used to fall through the `== ('hdf5' or 'h5')` bug and
        # raised NotImplementedError
        for fmt in ('hdf5', 'HDF5', 'h5'):
            req = self._make(format=fmt)
            self.assertTrue(req.h5)

    def test_format_sac_removed(self):
        # format='sac' used to select the mseed pipeline with SAC RF
        # output; receiver functions are now always written to the hdf5
        # database, so 'sac' raises with a pointer to the valid options
        with self.assertRaises(NotImplementedError):
            self._make(format='sac')
        req = self._make(format='mseed', prepro_subdir='prepro')
        self.assertFalse(req.h5)

    def test_unknown_format(self):
        with self.assertRaises(NotImplementedError):
            self._make(format='segy')

    def test_no_handler_accumulation(self):
        # Creating several Request objects must not duplicate handlers
        # on the shared 'pyglimer.request' logger (every message used to
        # be logged once per instantiation)
        self._make()
        self._make()
        self._make()
        log = logging.getLogger('pyglimer.request')
        self.assertEqual(len(log.handlers), 2)  # file + console


class TestRequestPreprocess(RequestTest):
    def test_n_jobs_forwarded(self):
        # Request.preprocess must expose n_jobs and forward it to
        # pyglimer.waveform.preprocess.preprocess
        with mock.patch('pyglimer.waveform.request.preprocess') as pp:
            req = self._make()
            req.preprocess(n_jobs=4)
            self.assertEqual(pp.call_args.kwargs['n_jobs'], 4)
            req.preprocess()
            self.assertEqual(pp.call_args.kwargs['n_jobs'], -1)


if __name__ == "__main__":
    unittest.main()
