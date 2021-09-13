'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3\
        <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 13th September 2021 10:38:53 am
Last Modified: Monday, 13th September 2021 12:16:53 pm
'''

import logging

import multiprocessing


def start_logger_if_necessary(
        name: str, logfile: str, loglvl: int) -> logging.Logger:
    """
    Initialise a logger.

    :param name: The logger's name
    :type name: str
    :param logfile: File to log to
    :type logfile: str
    :param loglvl: Log level
    :type loglvl: int
    :return: the logger
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(loglvl)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler(logfile, mode='w')
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


def create_mpi_logger(logger: logging.Logger, rank: int) -> logging.Logger:
    """
    Creates a very similar logger to the input logger, but with name and
    filehandler dependent on mpi rank.

    :param logger: [description]
    :type logger: logging.Logger
    :return: [description]
    :rtype: logging.Logger
    """
    lvl = logger.level
    rankstr = str(rank).zfill(3)
    name = '%srank%s' % (logger.name, rankstr)
    for h in logger.handlers:
        if hasattr(h, 'baseFilename'):
            fn = h.baseFilename
    return start_logger_if_necessary(name, fn, lvl)
