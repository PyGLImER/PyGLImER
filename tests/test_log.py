'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 14th September 2021 08:39:45 am
Last Modified: Tuesday, 14th September 2021 09:17:50 am
'''

import unittest
from unittest.mock import patch, mock_open
import os

import numpy as np

from pyglimer.utils import log


m = mock_open()


class TestStartLoggerIfNecessary(unittest.TestCase):
    @patch('builtins.open', m)
    def test_full_creation(self):
        loglvl = np.random.randint(0, 4)*10
        logger = log.start_logger_if_necessary('mylogger', 'myfile', loglvl)
        self.assertEqual(logger.name, 'mylogger')
        self.assertEqual(logger.level, loglvl)
        self.assertEqual(len(logger.handlers), 2)
        for h in logger.handlers:
            if hasattr(h, 'baseFilename'):
                self.assertEqual(os.path.basename(h.baseFilename), 'myfile')

    @patch('builtins.open', m)
    def test_already_existing_handler(self):
        loglvl = np.random.randint(0, 4)*10
        log.start_logger_if_necessary('mylogger2', 'myfile', loglvl)
        logger = log.start_logger_if_necessary(
            'mylogger2', 'notmyfile', loglvl-10)
        self.assertEqual(logger.name, 'mylogger2')
        self.assertEqual(logger.level, loglvl)
        self.assertEqual(len(logger.handlers), 2)
        for h in logger.handlers:
            if hasattr(h, 'baseFilename'):
                self.assertEqual(os.path.basename(h.baseFilename), 'myfile')


class TestCreateMPILogger(unittest.TestCase):
    @patch('builtins.open', m)
    def test_names(self):
        loglvl = np.random.randint(0, 4)*10
        logger = log.start_logger_if_necessary('mylogger3', 'myfile', loglvl)
        rank = np.random.randint(0, 999)
        rankstr = str(rank).zfill(3)
        mpilog = log.create_mpi_logger(logger, rank)
        exp_name = 'mylogger3rank%s' % rankstr
        exp_fname = 'myfilerank%s' % rankstr
        self.assertEqual(mpilog.name, exp_name)
        self.assertEqual(mpilog.level, loglvl)
        self.assertEqual(len(logger.handlers), 2)
        for h in mpilog.handlers:
            if hasattr(h, 'baseFilename'):
                self.assertEqual(os.path.basename(h.baseFilename), exp_fname)


if __name__ == "__main__":
    unittest.main()
