'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3
   <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 19th August 2021 04:01:26 pm
Last Modified: Thursday, 19th August 2021 04:23:29 pm
'''

import unittest
from unittest.mock import patch

from pyglimer.utils import utils as pu


class TestDownloadFullInventory(unittest.TestCase):
    @patch('pyglimer.utils.utils.os.listdir')
    @patch('pyglimer.utils.utils.__client__loop__')
    def test_bulk(self, cl_mock, oslist_mock):
        oslist_mock.return_value = [
            'not_a_statxml.txt',
            'net.stat.xml',
            'net2.stat2.xml',
            'also.not.jpg']
        exp = [
            ('net', 'stat', '*', '*', '*', '*'),
            ('net2', 'stat2', '*', '*', '*', '*')
        ]
        pu.download_full_inventory('bla', 'IRIS')
        cl_mock.assert_called_with('IRIS', 'bla', exp)


if __name__ == "__main__":
    unittest.main()
