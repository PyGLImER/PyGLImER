'''
Characterization tests for :mod:`pyglimer.waveform.qc` (SNR quality
control for downloaded waveforms).

The synthetic streams follow the window layout of the production code:
a strong teleseismic arrival near the theoretical onset and a noisy
preceding window. The assertions pin the accept/reject decision, the
returned filter frequency and whether the stream is overwritten with the
successfully filtered traces.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 6th November 2025
'''
import unittest

import numpy as np
from obspy import Stream, Trace

from pyglimer.waveform.qc import qcp, qcs, SNR_criteriaP, SNR_criteriaS


def _stream(burst_r=1.0, burst_z=0.8, seed=1, comps=('R', 'T', 'Z'),
            n=600, extra_bursts=None, burst_time=31.0):
    """
    Synthetic 3-component stream sampled at 10 Hz. `extra_bursts` is a
    list of (component, time_s, amplitude) tuples.
    """
    sr = 10.0
    t = np.arange(n)/sr
    rng = np.random.RandomState(seed)
    vert = '3' if '3' in comps else 'Z'
    amps = {'R': burst_r, vert: burst_z}
    traces = []
    for c in comps:
        data = 0.001*rng.randn(n)
        if c in amps and amps[c]:
            data = data + amps[c]*np.exp(-0.5*((t-burst_time)/0.5)**2)
        for comp, tc, amp in (extra_bursts or []):
            if comp == c:
                data = data + amp*np.exp(-0.5*((t-tc)/0.5)**2)
        tr = Trace(data=data.astype(np.float64))
        tr.stats.sampling_rate = sr
        tr.stats.component = c
        traces.append(tr)
    return Stream(traces)


class TestQcp(unittest.TestCase):
    def test_accept(self):
        st = _stream()
        orig = [tr.data.copy() for tr in st]
        out, crit, f, noisemat = qcp(st, 0.1, 10.0, 30.)
        self.assertIs(out, st)
        self.assertTrue(crit)
        # accepted with the broadest filter (first lowcut)
        self.assertEqual(f, 0.03)
        self.assertEqual(noisemat.shape, (3, 3))
        # SNR criteria are met at the accepted filter
        self.assertGreater(noisemat[0, 0], SNR_criteriaP[0])
        self.assertLess(noisemat[0, 1], SNR_criteriaP[1])
        self.assertGreater(noisemat[0, 2], SNR_criteriaP[2])
        # stream was overwritten with the filtered traces
        for tr, before in zip(st, orig):
            self.assertFalse(np.array_equal(tr.data, before))

    def test_reject(self):
        st = _stream(burst_r=0.0, burst_z=0.0)
        orig = [tr.data.copy() for tr in st]
        out, crit, f, noisemat = qcp(st, 0.1, 10.0, 30.)
        self.assertFalse(crit)
        # all filters failed -> last lowcut
        self.assertEqual(f, 0.5)
        # unfiltered stream is returned untouched
        for tr, before in zip(st, orig):
            np.testing.assert_array_equal(tr.data, before)

    def test_component_3(self):
        """Streams using channel '3' for the vertical are supported."""
        st = _stream(comps=('R', 'T', '3'))
        out, crit, f, _ = qcp(st, 0.1, 10.0, 30.)
        self.assertTrue(crit)


class TestQcs(unittest.TestCase):
    def test_accept(self):
        # For SRFs the Sp conversion is strong on R *before* the S onset,
        # while the criterion requires vertical energy to dominate there.
        st = _stream(
            burst_r=1.0, burst_z=0.0, n=1000, burst_time=60.,
            extra_bursts=[('Z', 40., 1.0)])
        orig = [tr.data.copy() for tr in st]
        out, crit, f, noisemat = qcs(st, 0.1, 10.0, 60.)
        self.assertIs(out, st)
        self.assertTrue(crit)
        # broadest highcut first
        self.assertEqual(f, 0.33)
        self.assertEqual(noisemat.shape, (4, 3))
        self.assertGreater(noisemat[0, 0], SNR_criteriaS[0])
        self.assertLess(noisemat[0, 1], SNR_criteriaS[1])
        self.assertLess(noisemat[0, 2], SNR_criteriaS[2])
        for tr, before in zip(st, orig):
            self.assertFalse(np.array_equal(tr.data, before))

    def test_reject(self):
        st = _stream(burst_r=0.0, burst_z=0.0, n=1000, burst_time=60.)
        out, crit, f, noisemat = qcs(st, 0.1, 10.0, 60.)
        self.assertFalse(crit)
        # last highcut after all failures
        self.assertEqual(f, 0.175)


if __name__ == "__main__":
    unittest.main()
