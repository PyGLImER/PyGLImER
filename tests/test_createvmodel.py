'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 19th August 2021 04:27:03 pm
Last Modified: Thursday, 19th August 2021 04:47:24 pm
'''

import unittest

from pyglimer.utils import createvmodel as cvm


class TestComplexModel(unittest.TestCase):
    def setUp(self):
        self.model = cvm.load_gyps()

    def testformat(self):
        self.assertIsInstance(self.model, cvm.ComplexModel)

    # def testquery(self):
    #     exp = (0.94528, 


if __name__ == "__main__":
    unittest.main()
