'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 21st October 2021 10:24:02 am
Last Modified: Thursday, 21st October 2021 11:18:04 am
'''

import unittest

import numpy as np
from obspy import read

from pyglimer.waveform import rotate


class TestRotateLQTmin(unittest.TestCase):
    def setUp(self):
        self.st = read()
        self.st.rotate('NE->RT', back_azimuth=10)
        for tr in self.st:
            tr.stats.sampling_rate = 1

    def test_rot_P(self):
        L = np.ones((60,))
        Q = np.zeros((60,))
        T = np.zeros((60,))
        # L[28:40] = 1
        LQ = np.array([L, Q])
        # Rotation in QL/RZ plane
        iad = np.random.randint(-90, 0)
        ia = iad*np.pi/180
        A_rot = np.array([
            [np.cos(ia), np.sin(ia)],
            [-np.sin(ia), np.cos(ia)]])
        RZ = np.dot(A_rot, LQ)
        self.st[0].data = RZ[1]
        self.st[1].data = RZ[0]
        self.st[2].data = T
        st_rot, iag = rotate.rotate_LQT_min(self.st, 'P')
        self.assertEqual(round(iad+90), round(iag))
        self.assertTrue(np.allclose(st_rot[0].data, L))
        self.assertTrue(np.allclose(st_rot[1].data, Q))
        self.assertTrue(np.allclose(st_rot[2].data, T))

    def test_rot_S(self):
        Q = np.ones((200,))
        L = np.zeros((200,))
        T = np.zeros((200,))
        # L[28:40] = 1
        LQ = np.array([L, Q])
        # Rotation in QL/RZ plane
        iad = np.random.randint(90, 180)
        ia = iad*np.pi/180
        A_rot = np.array([
            [np.cos(ia), np.sin(ia)],
            [-np.sin(ia), np.cos(ia)]])
        RZ = np.dot(A_rot, LQ)
        self.st[0].data = RZ[1]
        self.st[1].data = RZ[0]
        self.st[2].data = T
        st_rot, iag = rotate.rotate_LQT_min(self.st, 'S')
        self.assertEqual(round(iad-90), round(iag))
        self.assertTrue(np.allclose(st_rot[0].data, L))
        self.assertTrue(np.allclose(st_rot[1].data, Q))
        self.assertTrue(np.allclose(st_rot[2].data, T))


if __name__ == "__main__":
    unittest.main()
